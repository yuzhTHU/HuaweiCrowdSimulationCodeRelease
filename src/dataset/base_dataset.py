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
    栅格化地图数据结构。
    
    Attributes:
        map (np.array): 二维栅格地图数组，通常 0 表示可通行区域，1 表示障碍物。
        xmin (float): 地图在世界坐标系下的 x 轴最小值。
        ymin (float): 地图在世界坐标系下的 y 轴最小值。
        xmax (float): 地图在世界坐标系下的 x 轴最大值。
        ymax (float): 地图在世界坐标系下的 y 轴最大值。
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
    所有行人轨迹预测数据集的基类。
    
    提供了通用的样本切分、数据重采样、坐标标准化、缓存管理以及 PyTorch DataLoader 的 collate_fn。
    具体的加载逻辑由子类通过实现 `load_data` 类方法来完成。
    """
    def __init__(
        self, 
        name: str,
        args: Namespace, 
        df_data: pd.DataFrame,
        map_data: RasterizedMap=None,
    ):
        """
        初始化数据集。

        Args:
            name (str): 数据集名称（例如 "eth", "zara01"）。
            args (Namespace): 全局参数配置，需包含 fps, hist_step, pred_step 等。
            df_data (pd.DataFrame): 包含所有轨迹数据的 DataFrame。
                必须包含列: ['f', 'id', 'x', 'y', 'type']。
                - f: 帧号
                - id: 轨迹 ID
                - x, y: 坐标
                - type: 'pedestrian' 或 'vehicle'
            map_data (RasterizedMap, optional): 对应的场景地图数据。默认为 None。
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
            args (Namespace): 全局参数配置。

        Returns:
            BaseDataset: 加载好的数据集实例。
        
        Raises:
            NotImplementedError: 如果子类未实现此方法。
        """
        raise NotImplementedError
        df_data = ...
        map_data = ...
        return cls(name="unknown", args=args, df_data=df_data, map_data=map_data)

    def __len__(self):
        """返回数据集中的样本数量。"""
        return len(self.samples)

    def __getitem__(self, index):
        """获取指定索引的样本数据。"""
        return self.samples[index]

    def split_samples(self, df_data, use_tqdm=True):
        """
        将连续的轨迹数据切分为用于训练/测试的滑动窗口样本。

        根据 args.hist_step (历史步长) 和 args.pred_step (预测步长) 
        以及 args.skip_step (滑窗步长) 生成样本。每个样本包含当前场景下的
        所有行人和车辆的历史轨迹、未来轨迹标签以及相关的上下文信息。

        Args:
            df_data (pd.DataFrame): 包含完整轨迹的 DataFrame。
            use_tqdm (bool, optional): 是否显示进度条。默认为 True。

        Returns:
            List[dict]: 样本列表，每个样本是一个字典，包含：
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

            ## 行人数据整理
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

            # 插值
            ped_table = ped_table.interpolate(method='linear', limit_area='inside', axis=0)

            # 排除 f 时刻不在场的行人
            ped_list = ped_table.loc[f].unstack().notna().all(axis='columns')
            ped_list = ped_list[ped_list].index.tolist()
            ped_table = ped_table[ped_list]
            if len(ped_list) == 0:
                _logger.debug(f"No pedestrian at frame {f} in dataset {self.name}, skip.")
                continue

            # 填充 NaN
            # ped_table = ped_table.ffill().bfill()

            # 当前状态
            pos = ped_table.loc[f].values.reshape(len(ped_list), 2)  # (#ped, 2)
            assert pos.shape == (len(ped_list), 2)
            vel = ped_table.diff().loc[f].mul(fps).fillna(0).values.reshape(len(ped_list), 2)  # (#ped, 2)
            assert vel.shape == (len(ped_list), 2)

            # 未来加速度作为标签
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
            # 未来轨迹
            future_pos = (
                ped_table
                .iloc[-pred_step:]
                .values
                .reshape(pred_step, len(ped_list), 2)
                .transpose(1, 0, 2)
            )  # (#ped, pred_step, 2)
            assert future_pos.shape == (len(ped_list), pred_step, 2)

            # 历史轨迹
            hst = (
                ped_table
                .iloc[:hist_step]
                .values
                .reshape(hist_step, len(ped_list), 2)
                .transpose(1, 0, 2)
            )  # (#ped, hist_step, 2)
            assert hst.shape == (len(ped_list), hist_step, 2)

            # 未来 5s 平均速度
            future_5s = (
                df_data[
                    df_data['f'].ge(f - 1) &  # 包含当前帧，以允许计算下一步的速度
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
                .diff().mul(fps).iloc[1:]  # 去掉第一行 NaN（对应于当前第 f 帧的速度），只剩未来 5s
                .swaplevel(axis='columns').stack(future_stack=True) # dropna 避免 (NaN, NaN) 被丢弃
                .pow(2).sum(axis='columns', min_count=2).pow(0.5) # min_count 避免 (NaN, NaN) 被识别为 speed=0
                .unstack()
                .mean(axis='rows').values
                [..., np.newaxis]
            )  # (#ped, 1)
            assert spd.shape == (len(ped_list), 1), "您可能需要将这里上方的 future_stack=True 改成 dropna=False 再试一试，或者用我们推荐的 pandas 版本 2.3.3"
            
            # 目的地 (最后出现位置) 作为条件
            des = (
                df_data[df_data['id'].isin(ped_list)]
                .groupby('id').tail(1)
                .set_index('id').reindex(index=ped_list)
                [['x', 'y']].values
            )  # (#ped, 2)
            assert des.shape == (len(ped_list), 2)

            # 车辆信息作为条件
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
        DataLoader 的自定义整理函数，用于处理变长序列的 Padding。

        Args:
            batch (List[dict]): 由 __getitem__ 返回的样本列表。

        Returns:
            dict: 整理后的批次数据，所有张量已 Padding 并堆叠。
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
        对轨迹数据进行重采样，以匹配模型所需的目标帧率。

        Args:
            df_data (pd.DataFrame): 原始轨迹数据。
            raw_fps (float): 原始数据的帧率。默认为 30。
            target_fps (float): 目标帧率。默认为 2.5。

        Returns:
            pd.DataFrame: 重采样后的 DataFrame，包含插值后的坐标和更新的帧号。
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
        对坐标数据进行 Z-Score 标准化（归一化）。

        计算轨迹数据的均值和标准差，并将轨迹数据和地图边界同时进行标准化。
        注意：目前实现中 std 默认为 1.0 (仅去均值)，注释掉的代码为标准差归一化。

        Args:
            df_data (pd.DataFrame): 轨迹数据。
            map_data (RasterizedMap): 地图数据。

        Returns:
            tuple: (标准化后的 df_data, 更新后的 map_data)
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
        生成唯一的数据集缓存文件路径。

        缓存文件名包含数据集名称及关键参数 (fps, steps)，以避免参数变更后读取旧缓存。

        Args:
            args (Namespace): 参数配置。
            data_path (str): 原始数据路径 (未使用，仅作为签名参考)。
            name (str): 数据集名称。
            cache_dir (str, optional): 缓存目录。默认为 "./data/.cache"。

        Returns:
            Path: 缓存文件的完整路径。
        """
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_name = f"{name}_{args.fps}_{args.hist_step}_{args.pred_step}_{args.skip_step}.pkl"
        # key = f"{data_path}_{args.fps}_{args.hist_step}_{args.pred_step}_{args.skip_step}.pkl"
        # hash_key = hashlib.md5(key.encode()).hexdigest()[:10]
        # cache_name = f"{name}_{hash_key}.pkl"
        cache_path = cache_dir / cache_name
        return cache_path

    @staticmethod
    def save_cache(obj, cache_path):
        """将数据集对象序列化保存到磁盘缓存。"""
        with open(cache_path, "wb") as f:
            pickle.dump(obj, f)

    @staticmethod
    def load_cache(cache_path):
        """从磁盘缓存加载数据集对象。"""
        with open(cache_path, "rb") as f:
            return pickle.load(f)
