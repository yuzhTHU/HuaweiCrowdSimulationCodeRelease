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
from .base_dataset import BaseDataset, RasterizedMap
from ..utils.homography import calc_homography_mat, affine_transformation, image_to_world
from typing import List

_logger = logging.getLogger(__name__)


class UCYDataset(BaseDataset):
    raw_fps = 25

    @classmethod
    def load_data(cls, args: Namespace, data_path: str) -> "UCYDataset":
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data path {data_path} not found.")
        name = (
            data_path.parent.name.removeprefix("data_")
            + "-"
            + data_path.name.removesuffix(".vsp")
        )

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

        ## 数据重采样
        df_data = cls.resample_dataframe(df_data, raw_fps=cls.raw_fps, target_fps=args.fps)

        ## 仿射变换
        H = cls.get_homography_mat(mat_name=data_path.stem)
        df_data[['x', 'y']] = affine_transformation(df_data[['x', 'y']].values, H)

        ## 创建地图
        image_path = data_path.parent / f"{data_path.stem}.png"
        if image_path.exists():
            image = np.array(Image.open(image_path).convert('L'))  # (H, W)
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

        ## 返回数据集
        return cls(name=name, args=args, df_data=df_data, map_data=map_data)

    @classmethod
    def load_data_batch(self, args: Namespace, data_path: str, show_tqdm=True) -> List["UCYDataset"]:
        data_path = Path(data_path)
        if data_path.is_dir():
            files = list(sorted(data_path.glob("**/*.vsp")))
        elif "*" in data_path:
            if data_path.is_absolute():
                data_path = data_path.relative_to(".")
            files = list(sorted(Path(".").glob(data_path)))
        else:
            files = [data_path]

        datasets = []
        for file in tqdm(files, disable=not show_tqdm):
            datasets.append(self.load_data(args, file))
        return datasets

    @staticmethod
    def get_homography_mat(mat_name):
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
        else:
            raise ValueError(f"Unknown mat_name: {mat_name}")
        return H

if False:
    data_path = Path('data/UCY/data/data_zara/crowds_zara01.vsp')

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

    ## 数据重采样
    df_data = BaseDataset.resample_dataframe(df_data, raw_fps=25, target_fps=2.5)

    ## 仿射变换
    H = UCYDataset.get_homography_mat(mat_name=data_path.stem)
    df_data[['raw_x', 'raw_y']] = df_data[['x', 'y']]
    df_data[['x', 'y']] = affine_transformation(df_data[['x', 'y']].values, H)

    ## 创建地图
    image_path = data_path.parent / f"{data_path.stem}.png"
    image = np.array(Image.open(image_path).convert('L'))  # (H, W)
    map, xmin, xmax, ymin, ymax = image_to_world(image[::-1].T, H, dot_per_meter=5)

    ## 可视化
    fig, axes = plt.subplots(1, 2, figsize=(8, 5), dpi=300)
    h, w = image.shape[:2]
    axes[0].imshow(image, extent=(0, w, 0, h), cmap='gray_r')
    for pid, group in df_data.groupby('id'):
        cmap = plt.cm.get_cmap('viridis')
        axes[0].plot(group['raw_x'], group['raw_y'], color=cmap(group['f'].max() / df_data['f'].max()))
    axes[1].imshow(map, origin='lower', extent=(xmin, xmax, ymin, ymax), cmap='gray_r')
    for pid, group in df_data.groupby('id'):
        cmap = plt.cm.get_cmap('viridis')
        axes[1].plot(group['x'], group['y'], color=cmap(group['f'].max() / df_data['f'].max()))
    axes[1].set_facecolor('gray')
