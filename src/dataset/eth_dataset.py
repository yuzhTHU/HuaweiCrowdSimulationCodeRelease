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
    ETH pedestrian dataset loader.

    Handles scenes such as BIWI Hotel (ETH) and ETH Univ. The raw data usually
    comes in `obsmat.txt` or a similar observation-matrix format.
    """

    raw_fps = 25  # The official README says 25 fps, although the video subjectively looks faster than real walking.

    @classmethod
    def load_data(cls, args: Namespace, data_path: str) -> "ETHDataset":
        """
        Load a single ETH scene.

        If no cache is available, read the raw txt/csv file, load the map image
        and homography matrix, then apply coordinate mapping and resampling.

        Args:
            args (Namespace): Global arguments.
            data_path (str): Data file path, usually a txt file containing positions.

        Returns:
            ETHDataset: Initialized dataset instance.
        """
        data_path = Path(data_path)
        name = data_path.parent.name.removeprefix("seq_")

        ## Check cache.
        cache_path = cls._make_cache_path(args, str(data_path), name)
        if args.cache_dataset and os.path.exists(cache_path):
            _logger.info(f"Loading cached dataset from {cache_path}")
            try:
                dataset = cls.load_cache(cache_path)
                if len(dataset) == 0: # If it can be read but is empty, regenerating it would still be empty; fail early instead.
                    raise EmptyDatasetError(f"Cached dataset {cache_path} is empty.")
                cls.collate_fn([dataset[0]]) # Sanity-check that the dataset is usable.
                return dataset
            except Exception as e:
                _logger.error(f"Failed to use cached dataset {cache_path}: {e}")

        if not data_path.exists():
            raise FileNotFoundError(f"Data path {data_path} not found.")

        ## Read raw data.
        df_data = pd.read_csv(
            data_path,
            sep=' ',
            names=['f', 'id', 'x', 'z', 'y', 'vx', 'vz', 'vy'],
            usecols=['f', 'id', 'x', 'y'],
            skipinitialspace=True,
        )
        df_data['type'] = 'pedestrian'

        ## Resample data.
        df_data = cls.resample_dataframe(df_data, raw_fps=cls.raw_fps, target_fps=args.fps)

        ## Build map.
        H = np.loadtxt(data_path.parent / "H.txt")  # (3, 3)
        image = np.array(Image.open(data_path.parent / 'map.png').convert('L')) / 255.0 # (H, W)
        map, xmin, xmax, ymin, ymax = image_to_world(image, H, dot_per_meter=args.dot_per_meter)
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
    def load_data_batch(cls, args: Namespace, data_path: str, show_tqdm=True) -> List["ETHDataset"]:
        """
        Batch-load all ETH scenes under the given directory.

        Args:
            args (Namespace): Global arguments.
            data_path (str): Root directory path or glob pattern.
            show_tqdm (bool, optional): Whether to show a progress bar.

        Returns:
            List[ETHDataset]: List of dataset instances.
        """
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
