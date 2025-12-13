import os
import sys
import torch
import logging
import numpy as np
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from argparse import Namespace
from scipy.signal import medfilt, savgol_filter
from scipy.interpolate import interp1d
from .base_dataset import BaseDataset, EmptyDatasetError, RasterizedMap
from ..utils.homography import calc_homography_mat, affine_transformation, image_to_world
from typing import List

_logger = logging.getLogger(__name__)


class SDDDataset(BaseDataset):
    """
    Stanford Drone Dataset (SDD) 数据集加载器。
    
    该数据集包含俯视视角的无人机航拍视频，包含行人、自行车、滑板、车辆等多种类别。
    """

    raw_fps = 30

    @classmethod
    def load_data(cls, args: Namespace, data_path: str) -> "SDDDataset":
        """
        加载单个 SDD 视频场景的数据。

        读取 annotations.txt，进行像素到米的坐标变换。
        包含复杂的数据清洗逻辑（如去除异常速度、平滑轨迹）。
        如果存在 map.png 则加载，否则创建空白地图。

        Args:
            args (Namespace): 全局参数。
            data_path (str): annotations.txt 文件路径。

        Returns:
            SDDDataset: 初始化后的数据集实例。
        """
        data_path = Path(data_path)
        name = (
            data_path.parent.parent.name
            + "-"
            + data_path.parent.name.removeprefix("video")
        )

        ## 检查缓存
        cache_path = cls._make_cache_path(args, str(data_path), name)
        if args.cache_dataset and os.path.exists(cache_path):
            _logger.info(f"Loading cached dataset from {cache_path}")
            try:
                dataset = cls.load_cache(cache_path)
                if len(dataset) == 0: # 如果能读取但却是空的，重新生成一次也会是空的，不如直接报错通知这个用不了
                    raise EmptyDatasetError(f"Cached dataset {cache_path} is empty.")
                cls.collate_fn([dataset[0]]) # 测试能否正常使用
                return dataset
            except Exception as e:
                _logger.error(f"Failed to use cached dataset {cache_path}: {e}")

        if not data_path.exists():
            raise FileNotFoundError(f"Data path {data_path} not found.")

        ## 读取数据
        df_data = pd.read_csv(
            data_path,
            sep=" ",
            names=[
                "id", "xmin", "ymin", "xmax", "ymax",
                "frame", "lost", "occluded", "generated", "label",
            ],
            usecols=["id", "xmin", "ymin", "xmax", "ymax", "frame", "label"],
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
        df_data = df_data[['f', 'id', 'x', 'y', 'type']]  # 第一维向右，第二维向下

        ## 仿射变换
        H = cls.get_homography_mat(data_path=data_path)
        df_data[['x', 'y']] = affine_transformation(df_data[['x', 'y']].values, H)

        ## 清洗数据
        df_list = []
        for pid, group in df_data.groupby('id'):
            if group['type'].nunique() > 1:
                raise ValueError(f"ID {pid} has multiple types: {group['type'].unique()}")
            if group.iloc[0]['type'] != 'pedestrian':
                df_list.append(group.assign(id=len(df_list)))  # 重新编号
            else:
                traj = group[['f', 'x', 'y']].values
                traj[:, 0] /= args.fps  # convert to seconds
                cleaned_segs = cls.clean_trajectory(
                    traj,
                    median_k=5,
                    do_savgol=True, sg_win=9, sg_poly=2,
                    do_loess=False, loess_frac=0.08,
                    speed_window=11, speed_k=5.0,
                    angle_window=11, angle_k=5.0,
                    theta_thresh=np.deg2rad(90.0),
                    split_angle_thresh=np.deg2rad(120.0),
                    split_speed_delta=0.5, 
                    split_time_thresh=1.0
                )
                for seg in cleaned_segs:
                    seg[:, 0] *= args.fps  # convert back to frame index
                    df_list.append(pd.DataFrame(seg, columns=['f', 'x', 'y']).assign(id=len(df_list), type='pedestrian'))
        df_data = pd.concat(df_list, ignore_index=True)

        ## 数据重采样
        df_data = cls.resample_dataframe(df_data, raw_fps=cls.raw_fps, target_fps=args.fps)
        
        ## 读取地图
        map_path = data_path.parent / f"map.png"
        if map_path.exists():
            image = np.array(Image.open(map_path).convert('L')) / 255.0 # (H, W)  第一维向下，第二维向右
            image_ = image.T # 转置，使得第一维向右，第二维向下，与 df_data 中的坐标系对齐
            map, xmin, xmax, ymin, ymax = image_to_world(image_, H, dot_per_meter=args.dot_per_meter)  # 第一维向右，第二维向上，即 xy 坐标
        else:
            dot_per_meter = args.dot_per_meter
            xmin, xmax = df_data['x'].min(), df_data['x'].max()
            ymin, ymax = df_data['y'].min(), df_data['y'].max()
            len_x = len(np.arange(xmin, xmax+1/dot_per_meter, 1/dot_per_meter)[:-1])
            len_y = len(np.arange(ymin, ymax+1/dot_per_meter, 1/dot_per_meter)[:-1])
            map = np.full((len_x, len_y), np.nan)
        map_data = RasterizedMap(map=map, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

        ## 标准化坐标
        df_data, map_data = cls.normalize_xy(df_data, map_data)

        ## 处理数据集
        dataset = cls(name=name, args=args, df_data=df_data, map_data=map_data)

        ## 保存缓存
        cache_path = cls._make_cache_path(args, str(data_path), name)
        _logger.info(f"Caching dataset to {cache_path}")
        cls.save_cache(dataset, cache_path)

        return dataset

    @classmethod
    def load_data_batch(cls, args: Namespace, data_path: str, show_tqdm=True) -> List["SDDDataset"]:
        """批量加载 SDD 数据集。"""
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
                files = list(sorted(data_path.glob("**/annotations.txt")))
            elif "*" in str(data_path):
                if data_path.is_absolute():
                    data_path = data_path.relative_to(".")
                files = list(sorted(Path(".").glob(data_path)))
            else:
                files = [data_path]
            _logger.info(f"Caching dataset-list to {cache_path}")
            cls.save_cache(files, cache_path)

        datasets = []
        pbar = tqdm(files, disable=not show_tqdm, desc="Loading SDD datasets")
        for file in pbar:
            pbar.set_postfix_str(file.parent.parent.name + "/" + file.parent.name)
            datasets.append(cls.load_data(args, file))
            datasets[-1].path = str(file)
        return datasets

    @staticmethod
    def get_homography_mat(data_path):
        """
        根据数据集目录结构获取对应的单应性矩阵（像素/米 比例）。
        
        SDD 的不同场景（如 'bookstore', 'deathCircle'）有不同的缩放比例。

        Args:
            data_path (Path): 数据文件路径，用于推断场景名称。

        Returns:
            np.ndarray: 3x3 缩放矩阵 (实际上是对角矩阵)。
        """
        image0 = np.array(Image.open(data_path.parent / 'reference.jpg'))
        meter_per_pixel_dict = {
            'bookstore': {
                'video0': 36 / 1080, 'video1': 36 / 1080, 'video2': 39 / 1080,
                'video3': 53 / 1080, 'video4': 54 / 1080, 
                'video5': 34 / 1080, 'video6': 49 / 1080,
            },
            'coupa': {
                'video0': 30 / 1080, 'video1': 31 / 1080,
                'video2': 33 / 1080, 'video3': 33 / 1080,
            },
            'deathCircle': {
                'video0': 60 / 904, 'video1': 77 / 1080, 'video2': 82 / 1080,
                'video3': 55 / 1080, 'video4': 77 / 1080,
            },
            'gates': {
                'video0': 32 / 728, 'video1': 35 / 780, 'video2': 43 / 728,
                'video3': 31 / 772, 'video4': 40 / 784, 'video5': 51 / 1080,
                'video6': 32 / 712, 'video7': 37 / 728, 'video8': 32 / 728,
            },
            'hyang': {
                'video0': 45 / 816, 'video1': 60 / 780, 'video2': 43 / 842,
                'video3': 70 / 1434, 'video4': 41 / 836, 'video5': 40 / 788,
                'video6': 71 / 1416, 'video7': 46 / 808, 'video8': 41 / 752,
                'video9': 62 / 1080, 'video10': 36 / 748, 'video11': 36 / 748,
                'video12': 44 / 848, 'video13': 39 / 748, 'video14': 39 / 748,
            },
            'little': {
                'video0': 42/1080, 'video1': 40/1080, 
                'video2': 40/1080, 'video3': 40/1080,
            },
            'nexus': {
                'video0': 60 / 740, 'video1': 45 / 796, 'video2': 43 / 716,
                'video3': 37 / 716, 'video4': 39 / 740, 'video5': 38 / 728,
                'video6': 42 / 788, 'video7': 38 / 728, 'video8': 36 / 732,
                'video9': 37 / 788, 'video10': 33 / 732, 'video11': 42 / 772,
            },
            'quad': {
                'video0': 56 / 1968, 'video1': 58 / 1968, 
                'video2': 58 / 1968, 'video3': 60 / 1968,
            }
        }
        ratio = meter_per_pixel_dict[data_path.parent.parent.name][data_path.parent.name]
        h, w, _ = image0.shape
        H = calc_homography_mat(
            np.array([[0, h], [w, h], [0, 0], [w, 0]]),
            np.array([[0, 0], [w, 0], [0, h], [w, h]]) * ratio,
        )
        return H

    @staticmethod
    def clean_trajectory(traj,
                        median_k=5,
                        do_savgol=True, sg_win=9, sg_poly=2,
                        do_loess=False, loess_frac=0.08,
                        speed_window=11, speed_k=5.0, # for MAD threshold
                        angle_window=11, angle_k=5.0, # for MAD threshold
                        theta_thresh=np.deg2rad(90.0),
                        split_angle_thresh=np.deg2rad(120.0),
                        split_speed_delta=0.5, 
                        split_time_thresh=1.0):
        """
        [静态工具方法] 轨迹清洗与平滑。

        对原始轨迹进行去噪、平滑（Savitzky-Golay 或 LOESS）、
        速度和角度异常检测，并将轨迹分割成合理的片段。

        Args:
            traj (np.ndarray): 原始轨迹 (N, 3)，列为 [t, x, y]。
            median_k (int): 中值滤波窗口大小。
            do_savgol (bool): 是否使用 Savitzky-Golay 滤波。
            ... (其他平滑与分割阈值参数)

        Returns:
            List[np.ndarray]: 清洗后的轨迹片段列表。
        """
        # 1. sort & unique
        traj = np.array(traj)
        order = np.argsort(traj[:,0])
        traj = traj[order]
        # unique by time
        _, idx = np.unique(traj[:,0], return_index=True)
        traj = traj[idx]

        t = traj[:,0]
        x = traj[:,1].copy()
        y = traj[:,2].copy()

        N = len(t)
        if N < 4:
            return []

        # 2. median filter to remove spikes (operate on x,y separately)
        if median_k is not None and median_k >= 3:
            # medfilt requires odd kernel and pads; use same length
            k = median_k if median_k%2==1 else median_k+1
            x = medfilt(x, kernel_size=k)
            y = medfilt(y, kernel_size=k)

        # 3. regularize if needed: SG requires at least window length points and evenly sampled
        if do_savgol:
            if sg_win >= N:
                sg_win = N-1 if (N-1)%2==1 else N-2
                if sg_win < 3:
                    do_savgol = False
            if do_savgol:
                try:
                    x = savgol_filter(x, window_length=sg_win, polyorder=sg_poly, mode='interp')
                    y = savgol_filter(y, window_length=sg_win, polyorder=sg_poly, mode='interp')
                except Exception:
                    do_savgol = False

        # 4. LOESS as alternative (overrides SG if requested and available)
        if do_loess:
            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess
                x = lowess(x, t, frac=loess_frac, return_sorted=False)
                y = lowess(y, t, frac=loess_frac, return_sorted=False)
            except Exception:
                pass

        # 5. compute velocities and directions
        dt = np.diff(t)
        vxs = np.diff(x) / dt
        vys = np.diff(y) / dt
        speeds = np.sqrt(vxs**2 + vys**2)  # length N-1
        # pad to length N for per-point measures (associate speed[i] as speed from i->i+1)
        speeds_p = np.concatenate(([speeds[0]], speeds))
        angles = np.arctan2(np.concatenate(([vys[0]], vys)), np.concatenate(([vxs[0]], vxs)))


        # 6. local median & MAD on speed and on angle differences
        def rolling_median_mad(arr, window):
            # returns center-aligned median and MAD arrays (nan at edges)
            n = len(arr)
            med = np.full(n, np.nan)
            mad = np.full(n, np.nan)
            hw = window // 2
            for i in range(n):
                L = max(0, i - hw)
                R = min(n, i + hw + 1)
                w = arr[L:R]
                med[i] = np.median(w)
                mad[i] = np.median(np.abs(w - med[i])) if len(w)>0 else np.nan
            return med, mad

        med_speed, mad_speed = rolling_median_mad(speeds_p, speed_window)
        # robust threshold
        mad_eps = 1.4826 * mad_speed
        # indicator of speed anomaly (use k * MAD)
        speed_diff = np.abs(speeds_p - med_speed)
        is_speed_anom = (speed_diff > speed_k * mad_eps) | np.isnan(speed_diff)

        # angle change anomaly: look at difference between local median angles
        def circular_diff(a, b):
            d = a - b
            d = (d + np.pi) % (2*np.pi) - np.pi
            return np.abs(d)

        # compute local median angle
        med_angle, mad_angle = rolling_median_mad(angles, angle_window)
        # angle difference to median
        angle_diff = circular_diff(angles, med_angle)
        # indicator of speed anomaly (use k * MAD)
        mad_eps = 1.4826 * mad_angle
        # 将转角过大的点标记为异常
        delta_theta = circular_diff(angles[:-2], angles[2:])  # length N-1
        delta_theta = np.concatenate(([0.0], delta_theta, [0.0]))  # pad to length N
        is_angle_anom = (angle_diff > angle_k * mad_eps) | np.isnan(angle_diff) | (delta_theta > theta_thresh)

        # combine anomalies
        is_point_anom = is_speed_anom | is_angle_anom

        # 7. group consecutive anomalies into segments
        segments = []
        i = 0
        while i < N:
            if not is_point_anom[i]:
                i += 1
                continue
            # start a candidate anomaly region
            j = i
            while j+1 < N and is_point_anom[j+1]:
                j += 1
            # region [i..j]
            L = max(0, i-3)  # context window
            R = min(N-1, j+3)
            # compute med speed/angle left vs right
            left_idx = range(L, i)
            right_idx = range(j+1, R+1)
            if len(left_idx) == 0 or len(right_idx) == 0:
                # edge case near start/end -> prefer split or trim
                # if small region, remove by interpolation
                do_split = False if (j-i)<=1 else True
            else:
                v_left = np.median(speeds_p[list(left_idx)])
                v_right = np.median(speeds_p[list(right_idx)])
                theta_left = np.median(angles[list(left_idx)])
                theta_right = np.median(angles[list(right_idx)])
                # decide
                do_split = (abs(v_left - v_right) > split_speed_delta) or (circular_diff(theta_left, theta_right) > split_angle_thresh) or ((t[j] - t[i]) > split_time_thresh)

            if do_split:
                # cut trajectory: flush segment before i (if any) and start new segment after j
                # we add the clean chunk from previous pointer up to i-1 as a segment
                segments.append(('split', i, j))
            else:
                segments.append(('interp', i, j))
            i = j+1

        # 8. apply interpolation / splitting to produce final segments
        # Strategy: accumulate indices to include in each output segment
        cut_points = set()
        to_remove_intervals = []
        for typ, a, b in segments:
            if typ == 'split':
                # indicate a split between a-1 and a, and between b and b+1
                cut_points.add(a)   # split before index a
                cut_points.add(b+1) # split before b+1 (i.e., after b)
            else:
                # interpolation: mark indices a..b for removal and interpolation
                to_remove_intervals.append((a,b))

        # build segments by cutting at cut_points (sorted)
        cuts = sorted(list(cut_points))
        # always include start 0 and end N
        boundaries = [0] + cuts + [N]
        final_segments = []
        for sidx in range(len(boundaries)-1):
            st = boundaries[sidx]
            ed = boundaries[sidx+1]
            # segment indices [st, ed)
            if ed - st < 2:
                continue
            seg_t = t[st:ed].copy()
            seg_x = x[st:ed].copy()
            seg_y = y[st:ed].copy()
            final_segments.append(np.stack((seg_t, seg_x, seg_y), axis=1))

        # now apply interpolation removals within each final segment
        def apply_interps(segment):
            seg_t = segment[:,0]
            seg_x = segment[:,1]
            seg_y = segment[:,2]
            # find intervals fully contained in this segment
            base_start = seg_t[0]
            base_end = seg_t[-1]
            keep_mask = np.ones(len(seg_t), dtype=bool)
            for a,b in to_remove_intervals:
                # check if [a..b] indices map into this segment
                # map by time: if any times in segment correspond to indices a..b in original
                # original times are t
                idxs = np.where((t >= t[a]) & (t <= t[b]))[0]
                if len(idxs)==0:
                    continue
                # mark these indices for removal
                rel = np.where((seg_t >= t[a]) & (seg_t <= t[b]))[0]
                keep_mask[rel] = False
            if np.all(keep_mask):
                return segment
            # else, build new t/x/y with interpolation across removed points
            new_t = seg_t[keep_mask]
            new_x = seg_x[keep_mask]
            new_y = seg_y[keep_mask]
            # But we must ensure we can interpolate across gaps: check if gap endpoints exist
            # Use linear interpolation on time
            f_x = interp1d(new_t, new_x, kind='linear', bounds_error=False, fill_value="extrapolate")
            f_y = interp1d(new_t, new_y, kind='linear', bounds_error=False, fill_value="extrapolate")
            full_x = f_x(seg_t)
            full_y = f_y(seg_t)
            return np.stack((seg_t, full_x, full_y), axis=1)

        cleaned = [apply_interps(seg) for seg in final_segments]
        # filter tiny segments (<2 points)
        cleaned = [c for c in cleaned if len(c) >= 3]
        cleaned = [c for c in cleaned if np.sum(np.sqrt(np.sum(np.square(np.diff(c[:, 1:], axis=0)), axis=1))) > 3.0]  # remove near-constant segments
        return cleaned
