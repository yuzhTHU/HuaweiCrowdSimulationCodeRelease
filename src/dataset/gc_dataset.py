import os
import sys
import torch
import logging
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from typing import List
from io import StringIO
from pathlib import Path
from argparse import Namespace
from scipy.interpolate import griddata
from .base_dataset import BaseDataset, RasterizedMap
from ..utils.homography import calc_homography_mat, affine_transformation, image_to_world

_logger = logging.getLogger(__name__)


class GCDataset(BaseDataset):
    raw_fps = 25

    @classmethod
    def load_data(cls, args: Namespace, data_path: str) -> "GCDataset":
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data path {data_path} not found.")
        name = f"GC"

        ## 检查缓存
        cache_path = cls._make_cache_path(args, str(data_path), name)
        if args.cache_dataset and cache_path.exists():
            _logger.note(f"Loading cached dataset from {cache_path}")
            return cls.load_cache(cache_path)
        
        ## 读取数据
        H = cls.get_homography_mat()  # (3, 3)
        df_data = []
        for file in tqdm(sorted(data_path.glob('*.txt')), desc="Load GC Data"):
            xyf = np.loadtxt(file).reshape(-1, 3)
            df = pd.DataFrame(xyf, columns=['x', 'y', 'f'])
            df[['x', 'y']] = affine_transformation(df[['x', 'y']].values, H)
            df['abnormal'] = cls.get_abnormal(df, min_abnormal_speed=5.0, min_abnormal_whis=3.0)
            df['id'] = int(file.stem)
            df['type'] = 'pedestrian'
            df_data.append(df)
        df_data = pd.concat(df_data, ignore_index=True)

        ## 移除异常点
        df_data = df_data[~df_data['abnormal']]

        ## 数据重采样
        df_data = cls.resample_dataframe(df_data, raw_fps=cls.raw_fps, target_fps=args.fps)

        ## 创建地图
        image = np.array(Image.open(data_path.parent / f"map.png").convert('L')) # (H, W) 第一维向下，第二维向右
        image = image.T # 转置，使得第一维向右，第二维向下，与 df_data 中的坐标系对齐
        map, xmin, xmax, ymin, ymax = image_to_world(image, H, dot_per_meter=args.dot_per_meter) # 第一维向右，第二维向上，即 xy 坐标
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
    def get_homography_mat(cls) -> np.ndarray:
        H = np.array([
            [3.54477751e-02,  1.73477252e-02, -1.82112170e+01],
            [6.03523702e-04, -5.58259424e-02,  5.12654156e+01],
            [1.00205219e-05,  1.25487966e-03,  1.00000000e+00],
        ])
        return H
    
    @classmethod
    def get_abnormal(cls, df: pd.DataFrame, min_abnormal_speed=5.0, min_abnormal_whis=3.0) -> pd.Series:
        df_ = df.copy()
        while True:
            df_.sort_values(by='f').reset_index(drop=True)
            spd1 = df_[['x', 'y']].diff(axis=0).pow(2).sum(axis=1).pow(0.5).div(df_['f'].diff() / cls.raw_fps)
            spd2 = -df_[['x', 'y']].diff(-1, axis=0).pow(2).sum(axis=1).pow(0.5).div(df_['f'].diff(-1) / cls.raw_fps)
            spd = pd.concat([spd1, spd2], axis=1).fillna(0.0).min(axis=1)
            quantiles = spd.quantile([0.25, 0.75])
            IQR = quantiles.loc[0.75] - quantiles.loc[0.25]
            upper_bound = quantiles.loc[0.75] + min_abnormal_whis * IQR
            abnormal = (spd > upper_bound) & (spd > min_abnormal_speed)
            if not abnormal.any(): 
                spd = pd.concat([spd1, spd2], axis=1).min(axis=1)
                quantiles = spd.quantile([0.25, 0.75])
                IQR = quantiles.loc[0.75] - quantiles.loc[0.25]
                upper_bound = quantiles.loc[0.75] + min_abnormal_whis * IQR
                abnormal = (spd > upper_bound) & (spd > min_abnormal_speed)
            if abnormal.any():
                df_ = df_[~abnormal]
            else:
                break
        return ~df.index.isin(df_.index)
