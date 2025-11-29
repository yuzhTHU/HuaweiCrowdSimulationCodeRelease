import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from argparse import Namespace
from PIL import Image, ImageOps
from .base_dataset import BaseDataset, RasterizedMap
from ..utils.homography import calc_homography_mat, affine_transformation, image_to_world
from typing import List

_logger = logging.getLogger(__name__)

class ORCADataset(BaseDataset):
    raw_fps = None # 取决于数据文件

    @classmethod
    def load_data(cls, args: Namespace, data_path: str, with_shape=False) -> "ORCADataset":
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data path {data_path} not found.")
        name = data_path.parent.name

        ## 检查缓存
        cache_path = cls._make_cache_path(args, str(data_path), name)
        if args.cache_dataset and cache_path.exists():
            _logger.info(f"Loading cached dataset from {cache_path}")
            dataset = cls.load_cache(cache_path)
            if len(dataset) == 0:
                raise ValueError(f"Cached dataset {cache_path} is empty.")
            try:
                cls.collate_fn([dataset[0]]) # 测试能否正常使用
                return dataset
            except Exception as e:
                _logger.error(f"Failed to use cached dataset {cache_path}: {e}")

        ## 读取数据
        df_data = pd.read_csv(data_path).assign(type='pedestrian')

        ## 数据重采样
        raw_fps = float((data_path.parent / 'fps.txt').read_text().strip())
        df_data = cls.resample_dataframe(df_data, raw_fps=raw_fps, target_fps=args.fps)
        
        ## 读取地图
        if (image_path := data_path.parent / "map.png").exists():
            image = Image.open(image_path).convert('L')
            # 将黑白反转，变成白色障碍物，黑色道路
            image = ImageOps.invert(image)
            # 缩小图片，防止 image_to_world 内存爆炸
            h, w = image.size
            total_pixels = h * w
            max_pixels = 1e5
            if total_pixels > max_pixels:
                scale = (max_pixels / total_pixels) ** 0.5
                image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            image = np.array(image) / 255.0  # (H, W)  第一维向下，第二维向右
            h, w = image.shape
            xmin0, xmax0, ymin0, ymax0 = np.loadtxt(data_path.parent / 'map_range.txt')
            H = calc_homography_mat(
                np.array([[0, 0], [h, 0], [0, w], [h, w]]),
                np.array([[xmin0, ymax0], [xmin0, ymin0], [xmax0, ymax0], [xmax0, ymin0]]),
            )
            map, xmin, xmax, ymin, ymax = image_to_world(image, H, dot_per_meter=args.dot_per_meter)  # 第一维向右，第二维向上，即 xy 坐标
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
    def load_data_batch(cls, args: Namespace, data_path: str, show_tqdm=True) -> List["ORCADataset"]:
        ## 检查缓存
        name = '-'.join(Path(data_path).relative_to('./data').parts)
        cache_path = Path('./data/.cache') / Path(name).with_suffix(".pkl")
        if args.cache_dataset and cache_path.exists():
            _logger.info(f"Loading cached dataset-list from {cache_path}")
            files = cls.load_cache(cache_path)
        else:
            data_path = Path(data_path)
            if data_path.is_dir():
                files = list(sorted(data_path.glob("**/data.csv.gz")))
            elif "*" in str(data_path):
                if data_path.is_absolute():
                    data_path = data_path.relative_to(".")
                files = list(sorted(Path(".").glob(data_path)))
            else:
                files = [data_path]
            _logger.info(f"Caching dataset-list to {cache_path}")
            cls.save_cache(files, cache_path)
    
        datasets = []
        pbar = tqdm(files, disable=not show_tqdm, desc="Loading ORCA datasets")
        for file in pbar:
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
