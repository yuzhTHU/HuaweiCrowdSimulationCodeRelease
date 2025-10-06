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
from ..utils.homography import image_to_world

_logger = logging.getLogger(__name__)


class ETHDataset(BaseDataset):
    raw_fps = 2.5

    @classmethod
    def load_data(cls, args: Namespace, data_path: str) -> "ETHDataset":
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data path {data_path} not found.")
        name = data_path.parent.name.removeprefix("seq_")

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
        image = np.array(Image.open('data/ETH/seq_hotel/map.png').convert('L')) # (H, W)
        map, xmin, xmax, ymin, ymax = image_to_world(image, H, dot_per_meter=args.dot_per_meter)
        map_data = RasterizedMap(map=map, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

        # # 转换到世界坐标系
        # h, w = image.shape
        # ii, jj = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        # image_coord = np.stack([ii, jj], axis=-1).astype(np.float32)  # (H, W, 2)
        # xy1 = np.concatenate(
        #     [image_coord, np.ones((*image_coord.shape[:2], 1), dtype=np.float32)],
        #     axis=-1,
        # )  # (H, W, 3)
        # tmp = xy1 @ H.T  # (H, W, 3)
        # world_coord = tmp[..., :2] / tmp[..., 2:3]  # 除以最后一维以归一化 (H, W, 2)
        # df_map = pd.DataFrame({
        #     "x": world_coord[..., 0].reshape(-1),  # (H*W,)
        #     "y": world_coord[..., 1].reshape(-1),  # (H*W,)
        #     "value": image.reshape(-1),  # (H*W,)
        # })
        # # 双线性插值
        # dot_per_meter = 5  # 每米多少个点
        # xmin, xmax = df_map["x"].min(), df_map["x"].max()
        # ymin, ymax = df_map["y"].min(), df_map["y"].max()
        # xx, yy = np.meshgrid(
        #     np.linspace(xmin, xmax, dot_per_meter),
        #     np.linspace(ymin, ymax, dot_per_meter),
        # )
        # map = griddata(
        #     points=df_map[["x", "y"]].values,
        #     values=df_map["value"].values,
        #     xi=(xx, yy),
        #     method="linear",  # 'linear' 对应双线性插值
        # )
        # map_data = RasterizedMap(
        #     map=map,
        #     xmin=xmin,
        #     xmax=xmax,
        #     ymin=ymin,
        #     ymax=ymax,
        # )

        ## 标准化坐标
        df_data, map_data = cls.normalize_xy(df_data, map_data)

        ## 返回数据集
        return cls(name=name, args=args, df_data=df_data, map_data=map_data)

    @classmethod
    def load_data_batch(self, args: Namespace, data_path: str, show_tqdm=True) -> List["ETHDataset"]:
        data_path = Path(data_path)
        if data_path.is_dir():
            files = list(sorted(data_path.glob("**/obsmat.txt")))
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


if False:
    data_path = Path('data/ETH/seq_hotel/obsmat.txt')
    name = data_path.parent.name.removeprefix("seq_")

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
    df_data = BaseDataset.resample_dataframe(df_data, raw_fps=ETHDataset.raw_fps, target_fps=2.5)

    ## 创建地图
    H = np.loadtxt(data_path.parent / "H.txt")  # (3, 3)
    image = np.array(Image.open('data/ETH/seq_hotel/map.png').convert('L')) # (H, W)
    map, xmin, xmax, ymin, ymax = image_to_world(image, H, dot_per_meter=5.0)

    ## 可视化
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), dpi=300)
    # image0 = np.array(Image.open('data/UCY/data/Arxiepiskopi_flock.jpg'))  # (H, W)
    # h, w = image0.shape[:2]
    # # axes[0].imshow(image0, extent=(0, w, 0, h))
    # for pid, group in df_data.groupby('id'):
    #     cmap = plt.cm.get_cmap('viridis')
    #     axes[0].plot(group['raw_x'], group['raw_y'], color=cmap(group['f'].max() / df_data['f'].max()))
    # axes[0].axis('equal')
    h, w = image.shape[:2]
    axes[1].imshow(image, extent=(0, w, 0, h), cmap='gray_r')
    # for pid, group in df_data.groupby('id'):
    #     cmap = plt.cm.get_cmap('viridis')
    #     axes[1].plot(group['raw_x'], group['raw_y'], color=cmap(group['f'].max() / df_data['f'].max()))
    axes[1].axis('equal')
    axes[2].imshow(map, origin='lower', extent=(xmin, xmax, ymin, ymax), cmap='gray_r')
    for pid, group in df_data.groupby('id'):
        cmap = plt.cm.get_cmap('viridis')
        axes[2].plot(group['x'], group['y'], color=cmap(group['f'].max() / df_data['f'].max()))
    axes[2].set_facecolor('gray')
    axes[2].axis('equal')