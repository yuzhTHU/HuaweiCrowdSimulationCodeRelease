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
from ..utils.homography import calc_homography_mat, affine_transformation, image_to_world

_logger = logging.getLogger(__name__)


class GCDataset(BaseDataset):
    """
    Grand Central Station (GC) dataset loader.

    This is a high-density crowd dataset.
    """
    raw_fps = 25

    @classmethod
    def load_data(cls, args: Namespace, data_path: str) -> "GCDataset":
        """
        Load the GC dataset.

        Read all txt files in the directory, apply the homography transform,
        and remove trajectory points with abnormal speeds. Load and transpose
        the map image so the coordinate system aligns correctly.

        Args:
            args (Namespace): Global arguments.
            data_path (str): Dataset directory path.

        Returns:
            GCDataset: Initialized dataset instance.
        """
        data_path = Path(data_path)
        name = f"GC"

        ## Check cache.
        cache_path = cls._make_cache_path(args, str(data_path), name)
        if args.cache_dataset and cache_path.exists():
            _logger.info(f"Loading cached dataset from {cache_path}")
            try:
                dataset = cls.load_cache(cache_path)
                if len(dataset) == 0: # If it can be read but is empty, regenerating it would still be empty, so fail early.
                    raise EmptyDatasetError(f"Cached dataset {cache_path} is empty.")
                cls.collate_fn([dataset[0]]) # Sanity-check that the dataset is usable.
                return dataset
            except Exception as e:
                _logger.error(f"Failed to use cached dataset {cache_path}: {e}")

        if not data_path.exists():
            raise FileNotFoundError(f"Data path {data_path} not found.")

        ## Read raw data.
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

        ## Remove abnormal points.
        df_data = df_data[~df_data['abnormal']]

        ## Resample data.
        df_data = cls.resample_dataframe(df_data, raw_fps=cls.raw_fps, target_fps=args.fps)

        ## Build map.
        image = np.array(Image.open(data_path.parent / f"map.png").convert('L')) / 255.0 # (H, W), first dimension downward and second dimension rightward.
        image = image.T # Transpose so the first dimension points right and the second points down, aligning with the dataframe coordinates.
        map, xmin, xmax, ymin, ymax = image_to_world(image, H, dot_per_meter=args.dot_per_meter) # First dimension rightward and second upward, i.e. xy coordinates.
        map_data = RasterizedMap(map=map, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

        ## Normalize coordinates.
        df_data, map_data = cls.normalize_xy(df_data, map_data)

        ## Build dataset object.
        dataset = cls(name=name, args=args, df_data=df_data, map_data=map_data)

        ## Save cache.
        cache_path = cls._make_cache_path(args, str(data_path), name)
        _logger.info(f"Caching dataset to {cache_path}")
        cls.save_cache(dataset, cache_path)
        
        return dataset

    @classmethod
    def get_homography_mat(cls) -> np.ndarray:
        """
        Get the fixed homography matrix for the GC dataset.

        Returns:
            np.ndarray: 3x3 homography matrix.
        """
        H = np.array([
            [3.54477751e-02,  1.73477252e-02, -1.82112170e+01],
            [6.03523702e-04, -5.58259424e-02,  5.12654156e+01],
            [1.00205219e-05,  1.25487966e-03,  1.00000000e+00],
        ])
        return H
    
    @classmethod
    def get_abnormal(cls, df: pd.DataFrame, min_abnormal_speed=5.0, min_abnormal_whis=3.0) -> pd.Series:
        """
        Detect and mark abnormal trajectory points, such as points with
        excessively high speeds.

        Uses an IQR-based outlier detection algorithm.

        Args:
            df (pd.DataFrame): DataFrame containing trajectory data.
            min_abnormal_speed (float): Minimum abnormal-speed threshold.
            min_abnormal_whis (float): IQR multiplier.

        Returns:
            pd.Series: Boolean series where `True` means the row is normal and
            `False` means it is abnormal and should be removed.
        """
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
