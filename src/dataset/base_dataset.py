import os
import torch
import pickle
import hashlib
import logging
import numpy as np
import pandas as pd
import torch.utils.data as D
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from functools import reduce
from argparse import Namespace
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence

_logger = logging.getLogger(__name__)

@dataclass
class RasterizedMap:
    """
    Rasterized map data structure.

    Attributes:
        map (np.array): 2D raster map array, where 0 usually means walkable
            area and 1 means obstacle.
        xmin (float): Minimum x value of the map in world coordinates.
        ymin (float): Minimum y value of the map in world coordinates.
        xmax (float): Maximum x value of the map in world coordinates.
        ymax (float): Maximum y value of the map in world coordinates.
    """
    map: np.array = None
    xmin: float = None
    ymin: float = None
    xmax: float = None
    ymax: float = None


class EmptyDatasetError(BaseException):
    pass


class BaseDataset(D.Dataset):
    """
    Base class for all pedestrian-trajectory prediction datasets.

    Provides shared logic for sample splitting, resampling, coordinate
    normalization, cache management, and the PyTorch DataLoader `collate_fn`.
    Dataset-specific loading logic is implemented by subclasses through
    `load_data`.
    """
    def __init__(
        self, 
        name: str,
        args: Namespace, 
        df_data: pd.DataFrame,
        map_data: RasterizedMap=None,
    ):
        """
        Initialize the dataset.

        Args:
            name (str): Dataset name, e.g. `"eth"` or `"zara01"`.
            args (Namespace): Global configuration containing `fps`,
                `hist_step`, `pred_step`, and related fields.
            df_data (pd.DataFrame): DataFrame containing all trajectory data.
                Required columns: `['f', 'id', 'x', 'y', 'type']`.
                - f: 帧号
                - id: 轨迹 ID
                - x, y: 坐标
                - type: 'pedestrian' 或 'vehicle'
            map_data (RasterizedMap, optional): Scene map data.
        """
        self.args = args
        self.name = name
        self.df_data = df_data
        self.map_data = map_data
        self.samples = self.split_samples(df_data)

        delta_x = map_data.xmax - map_data.xmin
        delta_y = map_data.ymax - map_data.ymin
        w, h = map_data.map.shape
        if not (0.8 < (ratio := (delta_x / w) / (delta_y / h)) < 1.2):
            _logger.warning(
                f"Map aspect ratio of {name} mismatch: "
                f"data ratio={ratio:.4f} (xrange={delta_x:.4f}, yrange={delta_y:.4f}, "
                f"map shape={map_data.map.shape}), may cause distortion."
            )
    
    @classmethod
    def load_data(cls, args) -> 'BaseDataset':
        """
        [抽象方法] 从文件路径加载数据并返回数据集实例。

        子类必须实现此方法以处理特定的原始数据格式。

        Args:
            args (Namespace): Global configuration.

        Returns:
            BaseDataset: Loaded dataset instance.
        
        Raises:
            NotImplementedError: Raised when a subclass does not implement this method.
        """
        raise NotImplementedError
        df_data = ...
        map_data = ...
        return cls(name="unknown", args=args, df_data=df_data, map_data=map_data)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, index):
        """Fetch the sample at the given index."""
        return self.samples[index]

    def split_samples(self, df_data, use_tqdm=True):
        """
        Split continuous trajectory data into sliding-window samples for training and evaluation.

        Samples are generated according to `args.hist_step`, `args.pred_step`,
        and `args.skip_step`. Each sample contains the pedestrian and vehicle
        history, future labels, and context for the current scene.

        Args:
            df_data (pd.DataFrame): DataFrame containing complete trajectories.
            use_tqdm (bool, optional): Whether to show a progress bar.

        Returns:
            List[dict]: List of sample dictionaries, including:
                - pos: 当前时刻位置 (#ped, 2)
                - vel: 当前时刻速度 (#ped, 2)
                - des: 目的地 (#ped, 2)
                - spd: 期望速度 (#ped, 1)
                - hst: 历史轨迹 (#ped, hist_step, 2)
                - veh: 车辆历史 (#veh, hist_step+1, 2)
                - future_acc: 未来加速度标签 (#ped, pred_step, 2)
                - ...
        """
        hist_step = self.args.hist_step
        pred_step = self.args.pred_step * self.args.roll_step
        skip_step = self.args.skip_step
        fps = self.args.fps

        df_data = df_data.sort_values(by=['f', 'id']).reset_index(drop=True)

        df_data['f'] = df_data['f'].astype(int)
        if not df_data['type'].isin(['pedestrian', 'vehicle']).all():
            raise ValueError(f"Data type must be 'pedestrian' or 'vehicle', found {df_data['type'].unique()}")

        samples = []
        f_min, f_max = df_data['f'].min(), df_data['f'].max()
        for f in tqdm(range(f_min + hist_step, f_max - pred_step + 1, skip_step), disable=not use_tqdm):
            df = df_data[df_data['f'].ge(f - hist_step) & df_data['f'].lt(f + pred_step + 1)]

            ## Organize pedestrian data.
            ped_data = df[df['type'].eq('pedestrian')]
            ped_list = ped_data['id'].unique().tolist()
            ped_table = (
                ped_data
                .pivot_table(index='f', columns='id', values=['x', 'y'])
                .reindex(index=range(f - hist_step, f + pred_step + 1), 
                         columns=pd.MultiIndex.from_product([['x', 'y'], ped_list]))
                .swaplevel(axis='columns')
                .sort_index(axis='columns')
            )

            # Interpolation.
            ped_table = ped_table.interpolate(method='linear', limit_area='inside', axis=0)

            # Exclude pedestrians not present at frame `f`.
            ped_list = ped_table.loc[f].unstack().notna().all(axis='columns')
            ped_list = ped_list[ped_list].index.tolist()
            ped_table = ped_table[ped_list]
            if len(ped_list) == 0:
                _logger.debug(f"No pedestrian at frame {f} in dataset {self.name}, skip.")
                continue

            # Fill NaN if needed.
            # ped_table = ped_table.ffill().bfill()

            # Current state.
            pos = ped_table.loc[f].values.reshape(len(ped_list), 2)  # (#ped, 2)
            assert pos.shape == (len(ped_list), 2)
            vel = ped_table.diff().loc[f].mul(fps).fillna(0).values.reshape(len(ped_list), 2)  # (#ped, 2)
            assert vel.shape == (len(ped_list), 2)

            # Future acceleration as the label.
            future_acc = (
                ped_table
                .diff().mul(fps)
                .diff().mul(fps)
                .iloc[-pred_step:]
                .fillna(0)
                .values
                .reshape(pred_step, len(ped_list), 2)
                .transpose(1, 0, 2)
            )  # (#ped, pred_step, 2)
            assert future_acc.shape == (len(ped_list), pred_step, 2)
            # Future trajectory.
            future_pos = (
                ped_table
                .iloc[-pred_step:]
                .values
                .reshape(pred_step, len(ped_list), 2)
                .transpose(1, 0, 2)
            )  # (#ped, pred_step, 2)
            assert future_pos.shape == (len(ped_list), pred_step, 2)

            # History trajectory.
            hst = (
                ped_table
                .iloc[:hist_step]
                .values
                .reshape(hist_step, len(ped_list), 2)
                .transpose(1, 0, 2)
            )  # (#ped, hist_step, 2)
            assert hst.shape == (len(ped_list), hist_step, 2)

            # Mean speed over the next 5 seconds.
            future_5s = (
                df_data[
                    df_data['f'].ge(f - 1) &  # Include the current frame so the next-step speed can be computed.
                    df_data['f'].lt(f + 5 * fps) &
                    df_data['id'].isin(ped_list)
                ]
                .pivot_table(index='f', columns='id', values=['x', 'y'])
                .reindex(index=range(f-1, int(f + 5 * fps) + 1), 
                         columns=pd.MultiIndex.from_product([['x', 'y'], ped_list]))
                .swaplevel(axis='columns').sort_index(axis='columns')
                .interpolate(method='linear', limit_area='inside', axis='rows')
                # .ffill().bfill()
            )
            spd = (
                future_5s
                .diff().mul(fps).iloc[1:]  # Drop the first NaN row, which corresponds to the speed at the current frame.
                .swaplevel(axis='columns').stack(future_stack=True) # Keep `(NaN, NaN)` instead of dropping it.
                .pow(2).sum(axis='columns', min_count=2).pow(0.5) # `min_count` prevents `(NaN, NaN)` from being interpreted as speed=0.
                .unstack()
                .mean(axis='rows').values
                [..., np.newaxis]
            )  # (#ped, 1)
            assert spd.shape == (len(ped_list), 1), "You may need to replace `future_stack=True` above with `dropna=False`, or use the recommended pandas version 2.3.3."
            
            # Destination condition: the final observed position.
            des = (
                df_data[df_data['id'].isin(ped_list)]
                .groupby('id').tail(1)
                .set_index('id').reindex(index=ped_list)
                [['x', 'y']].values
            )  # (#ped, 2)
            assert des.shape == (len(ped_list), 2)

            # Vehicle context.
            veh_data = df[df['type'].eq('vehicle') & df['f'].ge(f - hist_step) & df['f'].lt(f + pred_step + 1)]
            veh_list = veh_data['id'].unique().tolist()
            veh_table = (
                veh_data
                .pivot_table(index='f', columns='id', values=['x', 'y'])
                .reindex(index=range(f - hist_step, f + pred_step + 1),
                         columns=pd.MultiIndex.from_product([['x', 'y'], veh_list]))
                .swaplevel(axis='columns').sort_index(axis='columns')
                .interpolate(method='linear', limit_area='inside', axis='rows')
            )
            veh = (
                veh_table
                .iloc[:hist_step + 1]
                .values
                .reshape(hist_step + 1, len(veh_list), 2)
                .transpose(1, 0, 2)
            )  # (#vehicle, hist_step + 1, 2)
            assert veh.shape == (len(veh_list), hist_step+1, 2)
            future_veh = (
                veh_table
                .iloc[-pred_step:]
                .values
                .reshape(pred_step, len(veh_list), 2)
                .transpose(1, 0, 2)
            )  # (#ped, pred_step, 2)
            assert future_veh.shape == (len(veh_list), pred_step, 2)

            samples.append({
                'pos': pos, # (#ped, 2)
                'vel': vel, # (#ped, 2)
                'des': des, # (#ped, 2)
                'spd': spd, # (#ped, 1)
                'hst': hst, # (#ped, hist_step, 2)
                'veh': veh, # (#veh, hist_step + 1, 2)
                'future_acc': future_acc, # (#ped, pred_step, 2)
                'future_pos': future_pos, # (#ped, pred_step, 2)
                'future_veh': future_veh, # (#veh, pred_step, 2)
                'f': f, # int
                'ped_id': ped_list, # (#ped,)
                'veh_id': veh_list, # (#veh,)
            })

        return samples

    @staticmethod
    def collate_fn(batch):
        """
        Custom DataLoader collate function for padding variable-length sequences.

        Args:
            batch (List[dict]): Sample list returned by `__getitem__`.

        Returns:
            dict: Batched tensors after padding and stacking.
                包含 'pos', 'vel', 'ped_length', 'veh_length' 等键。
                Padding 值通常为 0 (对于坐标) 或 -1 (对于 ID)。
        """
        pos = pad_sequence([torch.from_numpy(item['pos']).float() for item in batch], batch_first=True, padding_value=0.0)
        vel = pad_sequence([torch.from_numpy(item['vel']).float() for item in batch], batch_first=True, padding_value=0.0)
        des = pad_sequence([torch.from_numpy(item['des']).float() for item in batch], batch_first=True, padding_value=0.0)
        spd = pad_sequence([torch.from_numpy(item['spd']).float() for item in batch], batch_first=True, padding_value=0.0)
        hst = pad_sequence([torch.from_numpy(item['hst']).float() for item in batch], batch_first=True, padding_value=0.0)
        veh = pad_sequence([torch.from_numpy(item['veh']).float() for item in batch], batch_first=True, padding_value=0.0)
        future_acc = pad_sequence([torch.from_numpy(item['future_acc']).float() for item in batch], batch_first=True, padding_value=0.0)
        future_pos = pad_sequence([torch.from_numpy(item['future_pos']).float() for item in batch], batch_first=True, padding_value=0.0)
        future_veh = pad_sequence([torch.from_numpy(item['future_veh']).float() for item in batch], batch_first=True, padding_value=0.0)
        ped_length = torch.LongTensor([item['pos'].shape[0] for item in batch])
        veh_length = torch.LongTensor([item['veh'].shape[0] for item in batch])
        f = torch.LongTensor([item['f'] for item in batch])
        ped_id = pad_sequence([torch.LongTensor(item['ped_id']) for item in batch], batch_first=True, padding_value=-1)
        veh_id = pad_sequence([torch.LongTensor(item['veh_id']) for item in batch], batch_first=True, padding_value=-1)

        return {
            'pos': pos,
            'vel': vel,
            'des': des,
            'spd': spd,
            'hst': hst,
            'veh': veh,
            'future_acc': future_acc,
            'future_pos': future_pos,
            'future_veh': future_veh,
            'ped_length': ped_length,
            'veh_length': veh_length,
            'f': f,
            'ped_id': ped_id,
            'veh_id': veh_id,
        }

    @staticmethod
    def resample_dataframe(df_data, raw_fps=30, target_fps=2.5):
        """
        Resample trajectory data to the target frame rate expected by the model.

        Args:
            df_data (pd.DataFrame): Raw trajectory data.
            raw_fps (float): Original frame rate.
            target_fps (float): Target frame rate.

        Returns:
            pd.DataFrame: Resampled DataFrame with interpolated coordinates and updated frame indices.
        """
        if raw_fps == target_fps:
            return df_data.copy()

        new_df_data = []
        for id, group in df_data.groupby('id'):
            if group['type'].nunique() > 1:
                _logger.warning(f"ID {id} has multiple types: {group['type'].unique()}, use the first one.")
            t_raw = group['f'] / raw_fps
            f_new = np.arange(np.ceil(t_raw.min() * target_fps), np.floor(t_raw.max() * target_fps)).astype(int)
            t_new = f_new / target_fps
            x_new = np.interp(t_new, t_raw, group['x'])
            y_new = np.interp(t_new, t_raw, group['y'])
            new_group = pd.DataFrame({
                'f': f_new,
                'x': x_new,
                'y': y_new
            })
            new_group['id'] = id
            new_group['type'] = group['type'].iloc[0]
            new_df_data.append(new_group)
        new_df_data = pd.concat(new_df_data, ignore_index=True)
        if new_df_data.duplicated(subset=['f', 'id']).any():
            if df_data.duplicated(subset=['f', 'id']).any():
                raise ValueError("Input df_data has duplicate (f, id) entries.")
            raise ValueError("Resampling resulted in duplicate (f, id) entries.")
        return new_df_data


    @staticmethod
    def normalize_xy(df_data, map_data):
        """
        Apply Z-score-style normalization to coordinate data.

        The current implementation uses `std = 1.0`, so it effectively centers
        only. The commented code shows how full standard-deviation scaling would work.

        Args:
            df_data (pd.DataFrame): Trajectory data.
            map_data (RasterizedMap): Map data.

        Returns:
            tuple: `(normalized_df_data, updated_map_data)`
        """
        x_mean = 0.0 # df_data['x'].mean()
        x_std = 1.0 # df_data['x'].std()
        df_data['x'] = (df_data['x'] - x_mean) / x_std
        _logger.info(f"Normalized x with mean={x_mean:.4f}, std={x_std:.4f}")
        y_mean = 0.0 # df_data['y'].mean()
        y_std = 1.0 # df_data['y'].std()
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
        return df_data, map_data

    @staticmethod
    def _make_cache_path(args, data_path, name: str, cache_dir: str = "./data/.cache") -> Path:
        """
        Generate a unique dataset cache path.

        The cache filename includes the dataset name and key parameters to avoid
        loading stale caches after argument changes.

        Args:
            args (Namespace): Configuration arguments.
            data_path (str): Raw data path, unused except for signature compatibility.
            name (str): Dataset name.
            cache_dir (str, optional): Cache directory.

        Returns:
            Path: Full cache file path.
        """
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_name = f"{name}_{args.fps}_{args.hist_step}_{args.pred_step}_{args.skip_step}_{args.dot_per_meter}.pkl"
        # key = f"{data_path}_{args.fps}_{args.hist_step}_{args.pred_step}_{args.skip_step}.pkl"
        # hash_key = hashlib.md5(key.encode()).hexdigest()[:10]
        # cache_name = f"{name}_{hash_key}.pkl"
        cache_path = cache_dir / cache_name
        return cache_path

    @staticmethod
    def save_cache(obj, cache_path):
        """Serialize and save a dataset object to disk cache."""
        with open(cache_path, "wb") as f:
            pickle.dump(obj, f)

    @staticmethod
    def load_cache(cache_path):
        """Load a dataset object from disk cache."""
        with open(cache_path, "rb") as f:
            return pickle.load(f)
