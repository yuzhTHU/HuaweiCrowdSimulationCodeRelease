import os
import torch
import logging
import numpy as np
import pandas as pd
import torch.utils.data as D
import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import reduce
from argparse import Namespace
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence

_logger = logging.getLogger(__name__)

@dataclass
class RasterizedMap:
    """栅格化地图数据"""
    map: np.array = None
    xmin: float = None
    ymin: float = None
    xmax: float = None
    ymax: float = None


class BaseDataset(D.Dataset):
    def __init__(
            self, 
            name: str,
            args: Namespace, 
            df_data: pd.DataFrame,
            map_data: RasterizedMap=None,
        ):
        self.args = args
        self.name = name
        self.df_data = df_data
        self.map_data = map_data
        self.samples = self.split_samples(df_data)
    
    @classmethod
    def load_data(cls, args) -> 'BaseDataset':
        raise NotImplementedError
        df_data = ...
        map_data = ...
        return cls(name="unknown", args=args, df_data=df_data, map_data=map_data)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    def split_samples(self, df_data):
        hist_step = self.args.hist_step
        pred_step = self.args.pred_step
        skip_step = self.args.skip_step
        fps = self.args.fps

        df_data = df_data.sort_values(by=['f', 'id']).reset_index(drop=True)

        df_data['f'] = df_data['f'].astype(int)
        if not df_data['type'].isin(['pedestrian', 'vehicle']).all():
            raise ValueError(f"Data type must be 'pedestrian' or 'vehicle', found {df_data['type'].unique()}")

        samples = []
        f_min, f_max = df_data['f'].min(), df_data['f'].max()
        for f in range(f_min + hist_step, f_max - pred_step + 1, skip_step):
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
                _logger.warning(f"No pedestrian at frame {f} in dataset {self.name}, skip.")
                continue

            # 当前状态
            pos = ped_table.loc[f].values.reshape(len(ped_list), 2)  # (#ped, 2)
            vel = ped_table.diff().loc[f].mul(fps).fillna(0).values.reshape(len(ped_list), 2)  # (#ped, 2)

            # 未来加速度作为标签
            acc = (
                ped_table
                .diff().mul(fps)
                .diff().mul(fps)
                .iloc[-pred_step:]
                .fillna(0)
                .values
                .reshape(pred_step, len(ped_list), 2)
                .transpose(1, 0, 2)
            )  # (#ped, pred_step, 2)
            # 未来轨迹
            future = (
                ped_table
                .iloc[-pred_step:]
                .values
                .reshape(pred_step, len(ped_list), 2)
                .transpose(1, 0, 2)
            )  # (#ped, pred_step, 2)

            # 历史轨迹
            hst = (
                ped_table
                .iloc[:hist_step]
                .values
                .reshape(hist_step, len(ped_list), 2)
                .transpose(1, 0, 2)
            )  # (#ped, hist_step, 2)

            # 未来 5s 平均速度
            future_5s = df_data[
                df_data['f'].ge(f) & 
                df_data['f'].lt(f + 5 * fps) &
                df_data['id'].isin(ped_list)
            ]
            future_5s = (
                future_5s
                .pivot_table(index='f', columns='id', values=['x', 'y'])
                .reindex(index=range(f, future_5s['f'].max() + 1), 
                         columns=pd.MultiIndex.from_product([['x', 'y'], ped_list]))
                .swaplevel(axis='columns').sort_index(axis='columns')
                .interpolate(method='linear', limit_area='inside', axis='rows')
                .diff().mul(fps)
            )
            spd = (
                future_5s
                .swaplevel(axis='columns').stack(future_stack=True)
                .pow(2).sum(axis='columns').pow(0.5)
                .unstack()
                .mean(axis='rows').values
                [..., np.newaxis]
            )  # (#ped, 1)
            
            # 目的地 (最后出现位置) 作为条件
            des = (
                df_data[df_data['id'].isin(ped_list)]
                .groupby('id').tail(1)
                .set_index('id').reindex(index=ped_list)
                [['x', 'y']].values
            )  # (#ped, 2)

            # 车辆信息作为条件
            veh_hist_step = hist_step
            veh_data = df[df['type'].eq('vehicle') & df['f'].ge(f - veh_hist_step) & df['f'].lt(f + 1)]
            veh_list = veh_data['id'].unique().tolist()
            veh_table = (
                veh_data
                .pivot_table(index='f', columns='id', values=['x', 'y'])
                .reindex(index=range(f - veh_hist_step, f + 1),
                         columns=pd.MultiIndex.from_product([['x', 'y'], veh_list]))
                .swaplevel(axis='columns').sort_index(axis='columns')
                .interpolate(method='linear', limit_area='inside', axis='rows')
            )
            veh = (
                veh_table
                .values
                .reshape(veh_hist_step + 1, len(veh_list), 2)
                .transpose(1, 0, 2)
            )  # (#vehicle, veh_hist_step, 2)

            samples.append({
                'pos': torch.FloatTensor(pos), # (#ped, 2)
                'vel': torch.FloatTensor(vel), # (#ped, 2)
                'hst': torch.FloatTensor(hst), # (#ped, hist_step, 2)
                'des': torch.FloatTensor(des), # (#ped, 2)
                'spd': torch.FloatTensor(spd), # (#ped, 1)
                'veh': torch.FloatTensor(veh), # (#vehicle, veh_hist_step, 2)
                'acc': torch.FloatTensor(acc), # (#ped, pred_step, 2)
                'future': torch.FloatTensor(future), # (#ped, pred_step, 2)
            })

        return samples

    @staticmethod
    def collate_fn(batch):
        # pos = torch.stack([item['pos'] for item in batch], dim=0)
        # vel = torch.stack([item['vel'] for item in batch], dim=0)
        # acc = torch.stack([item['acc'] for item in batch], dim=0)
        # hst = torch.stack([item['hst'] for item in batch], dim=0)
        # des = torch.stack([item['des'] for item in batch], dim=0)
        # spd = torch.stack([item['spd'] for item in batch], dim=0)
        # veh = torch.stack([item['veh'] for item in batch], dim=0)
        pos = pad_sequence([item['pos'] for item in batch], batch_first=True, padding_value=0.0)
        vel = pad_sequence([item['vel'] for item in batch], batch_first=True, padding_value=0.0)
        acc = pad_sequence([item['acc'] for item in batch], batch_first=True, padding_value=0.0)
        hst = pad_sequence([item['hst'] for item in batch], batch_first=True, padding_value=0.0)
        des = pad_sequence([item['des'] for item in batch], batch_first=True, padding_value=0.0)
        spd = pad_sequence([item['spd'] for item in batch], batch_first=True, padding_value=0.0)
        veh = pad_sequence([item['veh'] for item in batch], batch_first=True, padding_value=0.0)
        future = pad_sequence([item['future'] for item in batch], batch_first=True, padding_value=0.0)
        ped_length = torch.LongTensor([item['pos'].shape[0] for item in batch])
        veh_length = torch.LongTensor([item['veh'].shape[0] for item in batch])

        return {
            'pos': pos,
            'vel': vel,
            'acc': acc,
            'hst': hst,
            'des': des,
            'spd': spd,
            'veh': veh,
            'future': future,
            'ped_length': ped_length,
            'veh_length': veh_length,
        }

    @staticmethod
    def resample_dataframe(df_data, raw_fps=30, target_fps=2.5):
        t_raw = df_data['f'] / raw_fps
        t_new = np.arange(t_raw.min(), t_raw.max(), 1/target_fps)
        x_new = np.interp(t_new, t_raw, df_data['x'])
        y_new = np.interp(t_new, t_raw, df_data['y'])
        f_new = (t_new * target_fps).astype(int)
        df_new = pd.DataFrame({
            'f': f_new,
            'x': x_new,
            'y': y_new
        })
        return df_new
