import os
import sys
import torch
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from argparse import Namespace
from PIL import Image, ImageOps
from scipy.spatial import cKDTree
from .base_dataset import BaseDataset, EmptyDatasetError, RasterizedMap
from ..utils.homography import calc_homography_mat, affine_transformation, image_to_world
from typing import List

_logger = logging.getLogger(__name__)

class WayMoDataset(BaseDataset):
    """
    Waymo Open Motion Dataset loader.

    Handles autonomous-driving scenes containing both pedestrians and vehicles.
    """
    raw_fps = 10

    @classmethod
    def load_data(cls, args: Namespace, data_path: str, with_shape=False) -> "WayMoDataset":
        """
        Load a single Waymo scene segment.

        Read the processed csv data, rename coordinate columns, map type labels,
        filter out trajectories that are too short and unrelated vehicles that
        are too far from pedestrians, and process the map image by inverting
        colors, resizing, and projecting it into world coordinates.

        Args:
            args (Namespace): Global arguments.
            data_path (str): Path to `data.csv.gz`.
            with_shape (bool, optional): Whether to load object-shape
                information. This is not fully implemented yet.

        Returns:
            WayMoDataset: Initialized dataset instance.
        """
        data_path = Path(data_path)
        name = data_path.parent.name

        ## Check cache.
        cache_path = cls._make_cache_path(args, str(data_path), name)
        if args.cache_dataset and os.path.exists(cache_path):
            _logger.info(f"Loading cached dataset from {cache_path}")
            try:
                dataset = cls.load_cache(cache_path)
                if len(dataset) == 0: 
                    raise Exception(f"Cached dataset is empty.")
                cls.collate_fn([dataset[0]]) # Sanity-check that the dataset is usable.
                if True: # Check whether the map aspect ratio is correct.
                    map_data = dataset.map_data
                    delta_x = map_data.xmax - map_data.xmin
                    delta_y = map_data.ymax - map_data.ymin
                    w, h = map_data.map.shape
                    if not (0.8 < (ratio := (delta_x / w) / (delta_y / h)) < 1.2):
                        raise Exception(
                            f"Map aspect ratio of {dataset.name} mismatch: "
                            f"data ratio={ratio:.4f} (xrange={delta_x:.4f}, yrange={delta_y:.4f}, "
                            f"map shape={map_data.map.shape}), may cause distortion."
                        )
                return dataset
            except Exception as e:
                if 'Cached dataset is empty.' in str(e): 
                    raise e from e  # If it can be read but is empty, regenerating it would still be empty, so fail early.
                _logger.error(f"Failed to use cached dataset {cache_path}: {e}")

        if not data_path.exists():
            raise FileNotFoundError(f"Data path {data_path} not found.")

        ## Read raw data.
        df_data = pd.read_csv(
            data_path, # center_x center_y center_z length width height heading velocity_x velocity_y f id type
            usecols=["center_x", "center_y", "f", "id", "type"],
        )
        df_data = df_data.rename(columns={"center_x": "x", "center_y": "y"})  # First dimension points right, second dimension points up.
        df_data['type'] = df_data['type'].replace({ # UNSET, VEHICLE, PEDESTRIAN, CYCLIST, OTHER
            'PEDESTRIAN': 'pedestrian',
            'VEHICLE': 'vehicle',
            'CYCLIST': 'vehicle',
            'UNSET': 'vehicle',
            'OTHER': 'vehicle',
        })

        ## Remove trajectories whose start-to-end distance is too short.
        df_data = cls.filter_short_trajectories(df_data, distance_threshold=3.0)

        ## Remove vehicles that are too far from pedestrians.
        df_data = cls.filter_vehicle_trajectories(df_data, distance_threshold=5.0)

        ## Resample data.
        df_data = cls.resample_dataframe(df_data, raw_fps=cls.raw_fps, target_fps=args.fps)
        
        ## Load map.
        map_path = data_path.parent / "map.png"
        image = Image.open(map_path).convert('L')
        # Invert black and white so obstacles become white and roads become black.
        image = ImageOps.invert(image)
        # Downscale the image to avoid excessive memory usage inside `image_to_world`.
        h, w = image.size
        total_pixels = h * w
        max_pixels = 1e5
        if total_pixels > max_pixels:
            scale = (max_pixels / total_pixels) ** 0.5
            image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        image = np.array(image) / 255.0  # (H, W), first dimension downward and second dimension rightward.
        h, w = image.shape
        xmin0, xmax0, ymin0, ymax0 = np.loadtxt(data_path.parent / 'map_range.txt')
        H = calc_homography_mat(
            np.array([[0, 0], [h, 0], [0, w], [h, w]]),
            np.array([[xmin0, ymax0], [xmin0, ymin0], [xmax0, ymax0], [xmax0, ymin0]]),
        )
        map, xmin, xmax, ymin, ymax = image_to_world(image, H, dot_per_meter=args.dot_per_meter)  # First dimension rightward and second upward, i.e. xy coordinates.
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
    def load_data_batch(cls, args: Namespace, data_path: str, show_tqdm=True, total=200) -> List["WayMoDataset"]:
        """
        Batch-load Waymo datasets.

        Supports sorting scenes by pedestrian count via `summary.csv` and
        selecting the top `total` scenes.

        Args:
            args (Namespace): Global arguments.
            data_path (str): Dataset root directory.
            total (int, optional): Maximum number of scenes to load.

        Returns:
            List[WayMoDataset]: Dataset list.
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
            df = pd.read_csv('data/WayMo/summary.csv', sep=',')
            df = df.sort_values('num_pedestrians', ascending=False)
            files = []
            for idx, row in df.iterrows():
                a = row['filename'].split('-')[1]
                b = row['id']
                c = row['scenario_id']
                file = Path('./data/WayMo/Processed') / f"{a}_{b}_{c}" / "data.csv.gz"
                files.append(file)
            _logger.info(f"Caching dataset-list to {cache_path}")
            cls.save_cache(files, cache_path)

        datasets = []
        pbar = tqdm(files, disable=not show_tqdm, desc="Loading WayMo datasets")
        for file in pbar:
            if len(datasets) == total: break
            try:
                pbar.set_postfix_str(file.parent.parent.name + "/" + file.parent.name)
                dataset = cls.load_data(args, file)
                if len(dataset.samples) == 0:
                    raise ValueError(f"Dataset {file} has no samples, skipping.")
                datasets.append(dataset)
                datasets[-1].path = str(file)
            except Exception as e:
                _logger.error(f"Failed to load {file}: {e}")
                continue
        return datasets

    @staticmethod
    def filter_vehicle_trajectories(df_data, distance_threshold=5.0):
        """
        Filter out vehicle trajectories whose distance to every pedestrian
        trajectory exceeds the threshold, reducing irrelevant data.

        Args:
            df_data (pd.DataFrame): Raw data.
            distance_threshold (float): Distance threshold in meters.

        Returns:
            pd.DataFrame: Filtered data.
        """
        if df_data.groupby('id')['type'].nunique().max() > 1:
            raise ValueError("Each id should correspond to a single type.")

        ped_points = df_data[df_data['type'] == 'pedestrian'][['x', 'y']].values  # (N, 2)
        kd_tree = cKDTree(ped_points)

        drop_id = []
        for pid, group in df_data[df_data['type'] == 'vehicle'].groupby('id'):
            veh_traj = group[['x', 'y']].values  # (M, 2)
            min_dists, _ = kd_tree.query(veh_traj, k=1)
            if np.min(min_dists) > distance_threshold:
                drop_id.append(pid)
        df_data = df_data[~df_data['id'].isin(drop_id)].reset_index(drop=True)
        return df_data

    @staticmethod
    def filter_short_trajectories(df_data, distance_threshold=3):
        """
        Filter out short trajectories whose displacement from start to end is
        below the threshold.

        Args:
            df_data (pd.DataFrame): Raw data.
            distance_threshold (float): Minimum displacement threshold in meters.

        Returns:
            pd.DataFrame: Filtered data.
        """
        drop_id = []
        for pid, group in df_data.sort_values(['id', 'f']).groupby('id'):
            start_position = group.iloc[0][['x', 'y']].values
            stop_position = group.iloc[-1][['x', 'y']].values
            dist = np.linalg.norm(start_position - stop_position)
            if dist < distance_threshold:
                drop_id.append(pid)
        df_data = df_data[~df_data['id'].isin(drop_id)].reset_index(drop=True)
        return df_data
