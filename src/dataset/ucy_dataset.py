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
    UCY (University of Cyprus) pedestrian dataset loader.

    Covers scenes such as Zara01, Zara02, and University Students.
    The raw data is typically stored in `.vsp` format or in a text format with
    spline control points.
    """
    raw_fps = 25

    @classmethod
    def load_data(cls, args: Namespace, data_path: str) -> "UCYDataset":
        """
        Load a single UCY scene.

        Parse the dataset-specific text format, including the pedestrian count
        and spline control points, then apply the required coordinate offsets
        such as `x += 360` and `y += 288`, followed by the homography transform.

        Args:
            args (Namespace): Global arguments.
            data_path (str): Path to the `.vsp` or data file.

        Returns:
            UCYDataset: Initialized dataset instance.
        """
        data_path = Path(data_path)
        name = (
            data_path.parent.name.removeprefix("data_")
            + "-"
            + data_path.name.removesuffix(".vsp")
        )

        ## Check cache.
        cache_path = cls._make_cache_path(args, str(data_path), name)
        if args.cache_dataset and os.path.exists(cache_path):
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
        df_list = []
        with open(data_path, 'r') as f:
            P = int(f.readline().split(' ')[0])  # Number of pedestrians.
            for p in range(P):
                S = int(f.readline().split(' ')[0])  # Number of control points.
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

        ## Apply the affine transformation.
        H = cls.get_homography_mat(mat_name=data_path.stem)
        df_data[['x', 'y']] = affine_transformation(df_data[['x', 'y']].values, H)

        ## Resample data.
        df_data = cls.resample_dataframe(df_data, raw_fps=cls.raw_fps, target_fps=args.fps)

        ## Build map.
        image_path = data_path.parent / f"{data_path.stem}.png"
        if image_path.exists():
            image = np.array(Image.open(image_path).convert('L')) / 255.0 # (H, W)
            image = image[::-1].T # Transpose and flip vertically so the first dimension points right and the second points up, aligning with the dataframe coordinates.
            map, xmin, xmax, ymin, ymax = image_to_world(image, H, dot_per_meter=args.dot_per_meter)
        else:
            dot_per_meter = args.dot_per_meter
            xmin, xmax = df_data['x'].min(), df_data['x'].max()
            ymin, ymax = df_data['y'].min(), df_data['y'].max()
            len_x = len(np.arange(xmin, xmax+1/dot_per_meter, 1/dot_per_meter)[:-1])
            len_y = len(np.arange(ymin, ymax+1/dot_per_meter, 1/dot_per_meter)[:-1])
            map = np.full((len_x, len_y), np.nan)
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
    def load_data_batch(cls, args: Namespace, data_path: str, show_tqdm=True) -> List["UCYDataset"]:
        """Batch-load UCY datasets."""
        ## Check cache.
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
        Get the predefined homography matrix for the specified scene.

        The UCY dataset typically requires a scene-specific `H` matrix to
        transform pixel coordinates into world coordinates.

        Args:
            mat_name (str): Scene name, e.g. `'students003'` or `'crowds_zara01'`.

        Returns:
            np.ndarray: 3x3 homography matrix.
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
