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
        cache_path = cls._make_cache_path(args, str(data_path), name)
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
        h, w, _ = image0.shape
        # 计算仿射矩阵以将图像中最长边缩放到 10 米
        H = calc_homography_mat(
            np.array([[0, h], [w, h], [0, 0], [w, 0]]),
            np.array([[0, 0], [w, 0], [0, h], [w, h]]) / max(h, w) * 10,
        )
        return H
