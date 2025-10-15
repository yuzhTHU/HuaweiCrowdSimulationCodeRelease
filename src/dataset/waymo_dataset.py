import os
import sys
import torch
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from argparse import Namespace
from PIL import Image, ImageOps
from scipy.spatial import cKDTree
from .base_dataset import BaseDataset, RasterizedMap
from ..utils.homography import calc_homography_mat, affine_transformation, image_to_world
from typing import List

_logger = logging.getLogger(__name__)

class WayMoDataset(BaseDataset):
    raw_fps = 10

    @classmethod
    def load_data(cls, args: Namespace, data_path: str) -> "WayMoDataset":
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data path {data_path} not found.")
        name = data_path.parent.name

        ## 检查缓存
        cache_path = cls._make_cache_path(args, str(data_path), name)
        if args.cache_dataset and os.path.exists(cache_path):
            _logger.info(f"Loading cached dataset from {cache_path}")
            return cls.load_cache(cache_path)

        ## 读取数据
        df_data = pd.read_csv(
            data_path, # center_x center_y center_z length width height heading velocity_x velocity_y f id type
            usecols=["center_x", "center_y", "f", "id", "type"],
        )
        df_data = df_data.rename(columns={"center_x": "x", "center_y": "y"})  # 第一维向右，第二维向上
        df_data['type'] = df_data['type'].replace({ # UNSET, VEHICLE, PEDESTRIAN, CYCLIST, OTHER
            'PEDESTRIAN': 'pedestrian',
            'VEHICLE': 'vehicle',
            'CYCLIST': 'vehicle',
            'UNSET': 'vehicle',
            'OTHER': 'vehicle',
        })

        ## 清除离行人太远的车辆
        df_data = cls.filter_vehicle_trajectories(df_data, distance_threshold=5.0)

        ## 数据重采样
        df_data = cls.resample_dataframe(df_data, raw_fps=cls.raw_fps, target_fps=args.fps)
        
        ## 读取地图
        map_path = data_path.parent / "map.png"
        image = Image.open(map_path).convert('L')
        # 将黑白反转，变成白色障碍物，黑色道路
        image = ImageOps.invert(image)
        # 缩小图片，防止 image_to_world 内存爆炸
        h, w = image.size
        total_pixels = h * w
        max_pixels = 1e5
        if total_pixels > max_pixels:
            scale = (max_pixels / total_pixels) ** 0.5
            image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        image = np.array(image)  # (H, W)  第一维向下，第二维向右
        h, w = image.shape
        xmin0, xmax0, ymin0, ymax0 = np.loadtxt(data_path.parent / 'map_range.txt')
        H = calc_homography_mat(
            np.array([[0, 0], [h, 0], [0, w], [h, w]]),
            np.array([[xmin0, ymax0], [xmin0, ymin0], [xmax0, ymax0], [xmax0, ymin0]]),
        )
        map, xmin, xmax, ymin, ymax = image_to_world(image, H, dot_per_meter=args.dot_per_meter)  # 第一维向右，第二维向上，即 xy 坐标
        map_data = RasterizedMap(map=map, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

        ## 标准化坐标
        df_data, map_data = cls.normalize_xy(df_data, map_data)

        ## 处理数据集
        dataset = cls(name=name, args=args, df_data=df_data, map_data=map_data)

        ## 保存缓存
        cache_path = cls._make_cache_path(args, str(data_path), name)
        _logger.info(f"Caching dataset to {cache_path}")
        cls.save_cache(dataset, cache_path)

        return dataset

    @classmethod
    def load_data_batch(cls, args: Namespace, data_path: str, show_tqdm=True, total=200) -> List["WayMoDataset"]:
        ## 检查缓存
        name = '-'.join(Path(data_path).relative_to('./data').parts)
        cache_path = Path('./data/.cache') / Path(name).with_suffix(".pkl")
        if args.cache_dataset and os.path.exists(cache_path):
            _logger.info(f"Loading cached dataset-list from {cache_path}")
            files = cls.load_cache(cache_path)
        else:
            data_path = Path(data_path)
            if data_path.is_dir():
                files = list(sorted(data_path.glob("**/data.csv.gz")))
            elif "*" in str(data_path):
                if data_path.is_absolute():
                    data_path = data_path.relative_to(".")
                files = list(sorted(Path(".").glob(data_path)))
            else:
                files = [data_path]
            _logger.info(f"Caching dataset-list to {cache_path}")
            cls.save_cache(files, cache_path)

        datasets = []
        pbar = tqdm(files, disable=not show_tqdm, desc="Loading WayMo datasets")
        for file in pbar:
            if len(datasets) == total: break
            try:
                pbar.set_postfix_str(file.parent.parent.name + "/" + file.parent.name)
                dataset = cls.load_data(args, file)
                if len(dataset.samples) == 0:
                    raise ValueError(f"Dataset {file} has no samples, skipping.")
                datasets.append(dataset)
            except Exception as e:
                _logger.error(f"Failed to load {file}: {e}")
                continue
        return datasets

    @staticmethod
    def filter_vehicle_trajectories(df_data, distance_threshold=5.0):
        """ 过滤掉与所有行人轨迹距离超过指定阈值的车辆轨迹。 """
        if df_data.groupby('id')['type'].nunique().max() > 1:
            raise ValueError("Each id should correspond to a single type.")

        ped_points = df_data[df_data['type'] == 'pedestrian'][['x', 'y']].values  # (N, 2)
        kd_tree = cKDTree(ped_points)

        drop_id = []
        for pid, group in df_data[df_data['type'] == 'vehicle'].groupby('id'):
            veh_traj = group[['x', 'y']].values  # (M, 2)
            min_dists, _ = kd_tree.query(veh_traj, k=1)
            if np.min(min_dists) > distance_threshold:
                drop_id.append(pid)
        df_data = df_data[~df_data['id'].isin(drop_id)].reset_index(drop=True)
        return df_data

    @staticmethod
    def filter_short_trajectories(df_data, distance_threshold=3):
        """ 过滤掉长度小于指定阈值的轨迹。 """
        drop_id = []
        for pid, group in df_data.sort_values(['id', 'f']).groupby('id'):
            dist = np.linalg.norm((group.iloc[0][['x', 'y']] - group.iloc[-1][['x', 'y']]).values)
            if dist < distance_threshold:
                drop_id.append(pid)
        df_data = df_data[~df_data['id'].isin(drop_id)].reset_index(drop=True)
        return df_data
