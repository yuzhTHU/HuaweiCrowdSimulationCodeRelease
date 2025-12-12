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
from .base_dataset import BaseDataset, EmptyDatasetError, RasterizedMap
from ..utils.homography import image_to_world

_logger = logging.getLogger(__name__)


class ETHDataset(BaseDataset):
    """
    ETH 行人数据集加载器。
    
    处理 BIWI Hotel (ETH) 和 ETH Univ 等场景的数据。
    原始数据格式通常为观察矩阵 (obsmat.txt) 或类似格式。
    """

    raw_fps = 25  # 官方 README 说是 25 fps，但是看视频感觉走起路来太快了不像真的

    @classmethod
    def load_data(cls, args: Namespace, data_path: str) -> "ETHDataset":
        """
        加载单个 ETH 场景数据。

        支持读取缓存。如果无缓存，则读取原始 txt/csv 文件，
        加载地图图像和单应性矩阵 (H matrix)，进行坐标映射和重采样。

        Args:
            args (Namespace): 全局参数。
            data_path (str): 数据文件路径 (通常是包含位置信息的 txt 文件)。

        Returns:
            ETHDataset: 初始化后的数据集实例。
        """
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data path {data_path} not found.")
        name = data_path.parent.name.removeprefix("seq_")

        ## 检查缓存
        cache_path = cls._make_cache_path(args, str(data_path), name)
        if args.cache_dataset and os.path.exists(cache_path):
            _logger.info(f"Loading cached dataset from {cache_path}")
            try:
                dataset = cls.load_cache(cache_path)
                if len(dataset) == 0: # 如果能读取但却是空的，重新生成一次也会是空的，不如直接报错通知这个用不了
                    raise EmptyDatasetError(f"Cached dataset {cache_path} is empty.")
                cls.collate_fn([dataset[0]]) # 测试能否正常使用
                return dataset
            except Exception as e:
                _logger.error(f"Failed to use cached dataset {cache_path}: {e}")
        
        ## 读取数据
        df_data = pd.read_csv(
            data_path,
            sep=' ',
            names=['f', 'id', 'x', 'z', 'y', 'vx', 'vz', 'vy'],
            usecols=['f', 'id', 'x', 'y'],
            skipinitialspace=True,
        )
        df_data['type'] = 'pedestrian'

        ## 数据重采样
        df_data = cls.resample_dataframe(df_data, raw_fps=cls.raw_fps, target_fps=args.fps)

        ## 创建地图
        H = np.loadtxt(data_path.parent / "H.txt")  # (3, 3)
        image = np.array(Image.open(data_path.parent / 'map.png').convert('L')) / 255.0 # (H, W)
        map, xmin, xmax, ymin, ymax = image_to_world(image, H, dot_per_meter=args.dot_per_meter)
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
    def load_data_batch(cls, args: Namespace, data_path: str, show_tqdm=True) -> List["ETHDataset"]:
        """
        批量加载指定目录下的所有 ETH 场景数据。

        Args:
            args (Namespace): 全局参数。
            data_path (str): 根目录路径或 glob 模式字符串。
            show_tqdm (bool, optional): 是否显示进度条。默认为 True。

        Returns:
            List[ETHDataset]: 数据集实例列表。
        """
        name = '-'.join(Path(data_path).relative_to('./data').parts)
        cache_path = Path('./data/.cache') / f"{name}.pkl"
        if args.cache_dataset and os.path.exists(cache_path):
            _logger.info(f"Loading cached dataset-list from {cache_path}")
            files = cls.load_cache(cache_path)
        else:
            data_path = Path(data_path)
            if data_path.is_dir():
                files = list(sorted(data_path.glob("**/obsmat.txt")))
            elif "*" in str(data_path):
                if data_path.is_absolute():
                    data_path = data_path.relative_to(".")
                files = list(sorted(Path(".").glob(data_path)))
            else:
                files = [data_path]
            _logger.info(f"Caching dataset-list to {cache_path}")
            cls.save_cache(files, cache_path)

        datasets = []
        pbar = tqdm(files, disable=not show_tqdm, desc="Loading ETH datasets")
        for file in pbar:
            pbar.set_postfix_str(file.parent.name)
            datasets.append(cls.load_data(args, file))
            datasets[-1].path = str(file)
        return datasets
