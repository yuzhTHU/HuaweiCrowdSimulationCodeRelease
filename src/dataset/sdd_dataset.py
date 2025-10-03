import os
import sys
import torch
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from argparse import Namespace
from .base_dataset import BaseDataset, RasterizedMap
from typing import List

_logger = logging.getLogger(__name__)

class SDDDataset(BaseDataset):
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

        df_data = pd.read_csv(
            data_path,
            sep=" ",
            names=[
                "id",
                "xmin",
                "ymin",
                "xmax",
                "ymax",
                "frame",
                "lost",
                "occluded",
                "generated",
                "label",
            ],
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

    @classmethod
    def load_data_batch(self, args: Namespace, data_path: str, show_tqdm=True) -> List["SDDDataset"]:
        data_path = Path(data_path)
        if data_path.is_dir():
            files = list(sorted(data_path.glob("**/annotations.txt")))
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

