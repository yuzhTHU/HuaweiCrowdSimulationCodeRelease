import os
import sys
import torch
import logging
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from io import StringIO
from pathlib import Path
from argparse import Namespace
from .base_dataset import BaseDataset, EmptyDatasetError, RasterizedMap
from ..utils.homography import calc_homography_mat, affine_transformation, image_to_world
from typing import List

_logger = logging.getLogger(__name__)


class UCYDataset(BaseDataset):
    """
    UCY (University of Cyprus) 行人数据集加载器。
    
    包含 Zara01, Zara02, University Students 等场景。
    原始数据通常为 .vsp 格式或包含样条控制点的文本格式。
    """
    raw_fps = 25

    @classmethod
    def load_data(cls, args: Namespace, data_path: str) -> "UCYDataset":
        """
        加载单个 UCY 场景数据。

        解析特定的文本格式（包含行人数、样条控制点等），
        应用特定的坐标偏移（如 x+=360, y+=288）和单应性变换。

        Args:
            args (Namespace): 全局参数。
            data_path (str): .vsp 或数据文件路径。

        Returns:
            UCYDataset: 初始化后的数据集实例。
        """
        data_path = Path(data_path)
        name = (
            data_path.parent.name.removeprefix("data_")
            + "-"
            + data_path.name.removesuffix(".vsp")
        )

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

        if not data_path.exists():
            raise FileNotFoundError(f"Data path {data_path} not found.")

        ## 读取数据
        df_list = []
        with open(data_path, 'r') as f:
            P = int(f.readline().split(' ')[0])  # 行人数
            for p in range(P):
                S = int(f.readline().split(' ')[0])  # 控制柄数
                csv = [f.readline().removesuffix(' - (2D point, m_id)') for _ in range(S)]
                df = pd.read_csv(
                    StringIO('\n'.join(csv)),
                    sep=" ",
                    nrows=S,
                    usecols=[0, 1, 2, 3],
                    names=["x", "y", "f", "direction"],
                )
                df["id"] = p
                df['type'] = 'pedestrian'
                df_list.append(df)
        df_data = pd.concat(df_list, ignore_index=True)
        df_data = df_data[['f', 'id', 'x', 'y', 'type']]
        df_data['x'] += 360
        df_data['y'] += 288

        ## 仿射变换
        H = cls.get_homography_mat(mat_name=data_path.stem)
        df_data[['x', 'y']] = affine_transformation(df_data[['x', 'y']].values, H)

        ## 数据重采样
        df_data = cls.resample_dataframe(df_data, raw_fps=cls.raw_fps, target_fps=args.fps)

        ## 创建地图
        image_path = data_path.parent / f"{data_path.stem}.png"
        if image_path.exists():
            image = np.array(Image.open(image_path).convert('L')) / 255.0 # (H, W)
            image = image[::-1].T # 转置并上下翻转，使得第一维向右，第二维向上，与 df_data 中的坐标系对齐
            map, xmin, xmax, ymin, ymax = image_to_world(image, H, dot_per_meter=args.dot_per_meter)
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
    def load_data_batch(cls, args: Namespace, data_path: str, show_tqdm=True) -> List["UCYDataset"]:
        """批量加载 UCY 数据集。"""
        ## 检查缓存
        name = '-'.join(Path(data_path).relative_to('./data').parts)
        cache_path = Path('./data/.cache') / f"{name}.pkl"
        try:
            assert args.cache_dataset, f"Cache disabled"
            assert cache_path.exists(), f"Cache {cache_path} not found"
            _logger.info(f"Loading cached dataset-list from {cache_path}")
            files = cls.load_cache(cache_path)
        except Exception as e:
            _logger.info(f"Failed to load cached dataset-list from {cache_path} since: {e}")
            data_path = Path(data_path)
            if data_path.is_dir():
                files = list(sorted(data_path.glob("**/*.vsp")))
            elif "*" in str(data_path):
                if data_path.is_absolute():
                    data_path = data_path.relative_to(".")
                files = list(sorted(Path(".").glob(data_path)))
            else:
                files = [data_path]
            _logger.info(f"Caching dataset-list to {cache_path}")
            cls.save_cache(files, cache_path)

        datasets = []
        pbar = tqdm(files, disable=not show_tqdm, desc="Loading UCY datasets")
        for file in pbar:
            pbar.set_postfix_str(file.parent.name + '/' + file.stem)
            datasets.append(cls.load_data(args, file))
            datasets[-1].path = str(file)
        return datasets

    @staticmethod
    def get_homography_mat(mat_name):
        """
        获取指定场景的预定义单应性矩阵。

        UCY 数据集通常需要特定的 H 矩阵将像素坐标转换为世界坐标。

        Args:
            mat_name (str): 场景名称 (e.g., 'students003', 'crowds_zara01')。

        Returns:
            np.ndarray: 3x3 单应性矩阵。
        """
        if mat_name in ['students001', 'students003', 'uni_examples']:
            length, width = 13, 12.6
            post1 = np.array([[132, 148], [602, 143], [166, 473], [561, 458]]) # - np.array([[360, 288]])
            post2 = np.array([[0, 0], [width, 0], [0, length], [width, length]])
            H = calc_homography_mat(post1, post2)
        elif mat_name in ['arxiepiskopi1']:
            H = np.eye(3)
        elif mat_name in ['crowds_zara01', 'crowds_zara02', 'crowds_zara03']:
            length, width = 5., 8.
            post1 = np.array([[32, 73], [630, 80], [78, 341], [593, 342]]) # - np.array([[360, 288]])
            post2 = np.array([[0, 0], [width, 0], [0, length], [width, length]])
            H = calc_homography_mat(post1, post2)
        # elif mat_name == 'students003': # https://github.com/erichhhhho/DataExtraction/blob/master/univ/H.txt
        #     H = np.array([
        #         [-2.3002776e-02,   5.3741914e-04,   8.6657256e+00],
        #         [-5.2753792e-04,   1.9565153e-02,  -6.0889188e+00],
        #         [0.0000000e+00,  -0.0000000e+00,   1.0000000e+00],
        #     ])
        # elif mat_name == 'crowds_zara01':
        #     H = np.array([
        #         [-2.5956517e-02,  -5.1572804e-18,   7.8388681e+00],
        #         [-1.0953874e-03,   2.1664330e-02,  -1.0032272e+01],
        #         [ 1.9540125e-20,   4.2171410e-19,   1.0000000e+00],
        #     ])
        # elif mat_name == 'crowds_zara02':
        #     H = np.array([
        #         [-2.5956517e-02,  -5.1572804e-18,   7.8388681e+00],
        #         [-1.0953874e-03,   2.1664330e-02,  -1.0032272e+01],
        #         [ 1.9540125e-20,   4.2171410e-19,   1.0000000e+00],
        #     ])
        else:
            raise ValueError(f"Unknown mat_name: {mat_name}")
        return H
