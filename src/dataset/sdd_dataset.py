import os
import sys
import torch
import logging
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from argparse import Namespace
from .base_dataset import BaseDataset, RasterizedMap
from ..utils.homography import calc_homography_mat, affine_transformation, image_to_world
from typing import List

_logger = logging.getLogger(__name__)


class SDDDataset(BaseDataset):
    raw_fps = 30

    @classmethod
    def load_data(cls, args: Namespace, data_path: str) -> "SDDDataset":
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data path {data_path} not found.")
        name = (
            data_path.parent.parent.name
            + "-"
            + data_path.parent.name.removeprefix("video")
        )

        ## 检查缓存
        cache_path = cls._make_cache_path(args, str(data_path), name)
        if args.cache_dataset and os.path.exists(cache_path):
            _logger.info(f"Loading cached dataset from {cache_path}")
            return cls.load_cache(cache_path)

        ## 读取数据
        df_data = pd.read_csv(
            data_path,
            sep=" ",
            names=[
                "id", "xmin", "ymin", "xmax", "ymax",
                "frame", "lost", "occluded", "generated", "label",
            ],
            usecols=["id", "xmin", "ymin", "xmax", "ymax", "frame", "label"],
        )
        df_data["x"] = (df_data["xmin"] + df_data["xmax"]) / 2
        df_data["y"] = (df_data["ymin"] + df_data["ymax"]) / 2
        df_data = df_data.rename(columns={"frame": "f", "label": "type"})
        df_data['type'] = df_data['type'].replace({
            'Pedestrian': 'pedestrian',
            'Skater': 'vehicle',
            'Biker': 'vehicle',
            'Cart': 'vehicle',
            'Car': 'vehicle',
            'Bus': 'vehicle',
        })
        df_data = df_data[['f', 'id', 'x', 'y', 'type']]  # 第一维向右，第二维向下

        ## 仿射变换
        H = cls.get_homography_mat(data_path=data_path)
        df_data[['x', 'y']] = affine_transformation(df_data[['x', 'y']].values, H)

        ## 数据重采样
        df_data = cls.resample_dataframe(df_data, raw_fps=cls.raw_fps, target_fps=args.fps)
        
        ## 读取地图
        map_path = data_path.parent / f"map.png"
        if map_path.exists():
            image = np.array(Image.open(map_path).convert('L'))  # (H, W)  第一维向下，第二维向右
            image_ = image.T # 转置，使得第一维向右，第二维向下，与 df_data 中的坐标系对齐
            map, xmin, xmax, ymin, ymax = image_to_world(image_, H, dot_per_meter=5)  # 第一维向右，第二维向上，即 xy 坐标
        else:
            dot_per_meter = args.dot_per_meter
            xmin, xmax = df_data['x'].min(), df_data['x'].max()
            ymin, ymax = df_data['y'].min(), df_data['y'].max()
            len_x = len(np.arange(xmin, xmax+1/dot_per_meter, 1/dot_per_meter)[:-1])
            len_y = len(np.arange(ymin, ymax+1/dot_per_meter, 1/dot_per_meter)[:-1])
            map = np.full((len_x, len_y), np.nan)
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
    def load_data_batch(cls, args: Namespace, data_path: str, show_tqdm=True) -> List["SDDDataset"]:
        name = '-'.join(Path(data_path).relative_to('./data').parts)
        cache_path = Path('./data/.cache') / f"{name}.pkl"
        if args.cache_dataset and os.path.exists(cache_path):
            _logger.info(f"Loading cached dataset-list from {cache_path}")
            files = cls.load_cache(cache_path)
        else:
            data_path = Path(data_path)
            if data_path.is_dir():
                files = list(sorted(data_path.glob("**/annotations.txt")))
            elif "*" in str(data_path):
                if data_path.is_absolute():
                    data_path = data_path.relative_to(".")
                files = list(sorted(Path(".").glob(data_path)))
            else:
                files = [data_path]
            _logger.info(f"Caching dataset-list to {cache_path}")
            cls.save_cache(files, cache_path)

        datasets = []
        pbar = tqdm(files, disable=not show_tqdm, desc="Loading SDD datasets")
        for file in pbar:
            pbar.set_postfix_str(file.parent.parent.name + "/" + file.parent.name)
            datasets.append(cls.load_data(args, file))
        return datasets

    @staticmethod
    def get_homography_mat(data_path):
        image0 = np.array(Image.open(data_path.parent / 'reference.jpg'))
        meter_per_pixel_dict = {
            'bookstore': {
                'video1': 36 / 1080, 'video2': 36 / 1080, 'video3': 36 / 1080,
                'video4': 36 / 1080, 'video5': 36 / 1080, 'video6': 36 / 1080,
            },
            'coupa': {
                'video1': 30 / 1080, 'video2': 31 / 1080,
                'video3': 33 / 1080, 'video4': 33 / 1080,
            },
            'deathCircle': {
                'video0': 60 / 904, 'video1': 77 / 1080, 'video2': 82 / 1080,
                'video3': 55 / 1080, 'video4': 77 / 1080,
            },
            'gates': {
                'video0': 32 / 728, 'video1': 35 / 780, 'video2': 43 / 728,
                'video3': 31 / 772, 'video4': 40 / 784, 'video5': 51 / 1080,
                'video6': 32 / 712, 'video7': 37 / 728, 'video8': 32 / 728,
            },
            'hyang': {
                'video0': 45 / 816, 'video1': 60 / 780, 'video2': 43 / 842,
                'video3': 70 / 1434, 'video4': 41 / 836, 'video5': 40 / 788,
                'video6': 71 / 1416, 'video7': 46 / 808, 'video8': 41 / 752,
                'video9': 62 / 1080, 'video10': 36 / 748, 'video11': 36 / 748,
                'video12': 44 / 848, 'video13': 39 / 748, 'video14': 39 / 748,
            },
            'little': {
                'video0': 42/1080, 'video1': 40/1080, 
                'video2': 40/1080, 'video3': 40/1080,
            },
            'nexus': {
                'video0': 60 / 740, 'video1': 45 / 796, 'video2': 43 / 716,
                'video3': 37 / 716, 'video4': 39 / 740, 'video5': 38 / 728,
                'video6': 42 / 788, 'video7': 38 / 728, 'video8': 36 / 732,
                'video9': 37 / 788, 'video10': 33 / 732, 'video11': 42 / 772,
            },
            'quad': {
                'video0': 56 / 1968, 'video1': 58 / 1968, 
                'video2': 58 / 1968, 'video3': 60 / 1968,
            }
        }
        ratio = meter_per_pixel_dict[data_path.parent.parent.name][data_path.parent.name]
        h, w, _ = image0.shape
        H = calc_homography_mat(
            np.array([[0, h], [w, h], [0, 0], [w, 0]]),
            np.array([[0, 0], [w, 0], [0, h], [w, h]]) * ratio,
        )
        return H
