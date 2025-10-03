import os
import sys
import torch
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from io import StringIO
from pathlib import Path
from argparse import Namespace
from .base_dataset import BaseDataset, RasterizedMap
from typing import List

_logger = logging.getLogger(__name__)


class UCYDataset(BaseDataset):
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

        df_data_list = []
        for id, group in df_data.groupby('id'):
            if group['type'].nunique() > 1:
                raise ValueError(f"ID {id} has multiple types: {group['type'].unique()}")
            new_group = cls.resample_dataframe(group, raw_fps=30, target_fps=args.fps)
            new_group['id'] = id
            new_group['type'] = group['type'].iloc[0]
            df_data_list.append(new_group)
        df_data = pd.concat(df_data_list, ignore_index=True)

        map_data = RasterizedMap(
            map=np.zeros((150, 150)),
            xmin=0,
            xmax=30,
            ymin=0,
            ymax=30,
        )

        x_mean = df_data['x'].mean()
        x_std = df_data['x'].std()
        df_data['x'] = (df_data['x'] - x_mean) / x_std
        _logger.info(f"Normalized x with mean={x_mean:.4f}, std={x_std:.4f}")
        y_mean = df_data['y'].mean()
        y_std = df_data['y'].std()
        df_data['y'] = (df_data['y'] - y_mean) / y_std
        _logger.info(f"Normalized y with mean={y_mean:.4f}, std={y_std:.4f}")
        map_data.xmin = (map_data.xmin - x_mean) / x_std
        map_data.xmax = (map_data.xmax - x_mean) / x_std
        map_data.ymin = (map_data.ymin - y_mean) / y_std
        map_data.ymax = (map_data.ymax - y_mean) / y_std
        _logger.info(
            f"Normalized map into xmin={map_data.xmin:.4f}, xmax={map_data.xmax:.4f}, "
            f"ymin={map_data.ymin:.4f}, ymax={map_data.ymax:.4f}"
        )

        return cls(name=name, args=args, df_data=df_data, map_data=map_data)
