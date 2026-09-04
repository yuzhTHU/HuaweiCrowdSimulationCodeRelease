"""Helpers for datasets uploaded through the web console."""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageOps
from src.dataset.base_dataset import BaseDataset, RasterizedMap
from src.utils.homography import calc_homography_mat, image_to_world

__all__ = ["safe_filename", "validate_custom_csv", "prepare_custom_upload", "load_custom_dataset"]

REQUIRED_COLUMNS = {"f", "x", "y", "id", "type"}


def safe_filename(filename: str) -> str:
    """Return a flat, filesystem-safe upload filename."""
    name = Path(filename or "upload").name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name) or "upload"


def _normalise_types(series: pd.Series) -> pd.Series:
    aliases = {
        "ped": "pedestrian", "person": "pedestrian", "walker": "pedestrian",
        "pedestrian": "pedestrian", "2": "pedestrian",
        "vehicle": "vehicle", "car": "vehicle", "cyclist": "vehicle",
        "unset": "vehicle", "other": "vehicle", "1": "vehicle", "3": "vehicle",
        "4": "vehicle", "0": "vehicle",
    }
    values = series.astype(str).str.strip().str.lower().map(aliases)
    if values.isna().any():
        invalid = sorted(series[values.isna()].astype(str).unique().tolist())
        raise ValueError(f"Unsupported values in 'type': {invalid!r}. Use pedestrian or vehicle.")
    return values


def validate_custom_csv(path: Path) -> pd.DataFrame:
    """Validate and canonicalise a user trajectory CSV."""
    df = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    df = df[["f", "x", "y", "id", "type"]].copy()
    for column in ("f", "x", "y", "id"):
        df[column] = pd.to_numeric(df[column], errors="raise")
    if df.empty or not np.isfinite(df[["f", "x", "y", "id"]].to_numpy()).all():
        raise ValueError("Trajectory data must be non-empty and contain only finite numeric values.")
    if df.duplicated(["f", "id"]).any():
        raise ValueError("The uploaded data contains duplicate (f, id) rows.")
    if not np.equal(df[["f", "id"]], np.floor(df[["f", "id"]])).all().all():
        raise ValueError("Columns 'f' and 'id' must contain integers.")
    df[["f", "id"]] = df[["f", "id"]].astype(int)
    df["type"] = _normalise_types(df["type"])
    return df.sort_values(["f", "id"]).reset_index(drop=True)


def prepare_custom_upload(upload_dir: Path, source: Path) -> Path:
    """Store canonical data.csv.gz while retaining every original upload."""
    df = validate_custom_csv(source)
    output = upload_dir / "data.csv.gz"
    df.to_csv(output, index=False, compression="gzip")
    return output


def _empty_map(df: pd.DataFrame, dot_per_meter: float) -> RasterizedMap:
    xmin, xmax = float(df.x.min()), float(df.x.max())
    ymin, ymax = float(df.y.min()), float(df.y.max())
    span = max(xmax - xmin, ymax - ymin, 10.0)
    margin = max(span * 0.05, 1.0)
    xmin, xmax, ymin, ymax = xmin - margin, xmax + margin, ymin - margin, ymax + margin
    width = max(2, int(np.ceil((xmax - xmin) * dot_per_meter)))
    height = max(2, int(np.ceil((ymax - ymin) * dot_per_meter)))
    return RasterizedMap(np.zeros((width, height), dtype=np.float32), xmin, ymin, xmax, ymax)


def _load_map(data_path: Path, df: pd.DataFrame, dot_per_meter: float) -> RasterizedMap:
    image_path = data_path.parent / "map.png"
    range_path = data_path.parent / "map_range.txt"
    if not image_path.exists() or not range_path.exists():
        return _empty_map(df, dot_per_meter)
    image = ImageOps.invert(Image.open(image_path).convert("L"))
    image_array = np.asarray(image, dtype=float) / 255.0
    height, width = image_array.shape
    xmin, xmax, ymin, ymax = np.loadtxt(range_path)
    transform = calc_homography_mat(
        np.array([[0, 0], [height, 0], [0, width], [height, width]]),
        np.array([[xmin, ymax], [xmin, ymin], [xmax, ymax], [xmax, ymin]]),
    )
    grid, xmin, xmax, ymin, ymax = image_to_world(
        image_array, transform, dot_per_meter=dot_per_meter
    )
    return RasterizedMap(grid, xmin, ymin, xmax, ymax)


def load_custom_dataset(args, data_path: str) -> BaseDataset:
    """Load a canonical five-column upload into the simulation data model."""
    path = Path(data_path)
    df = validate_custom_csv(path)
    map_data = _load_map(path, df, args.dot_per_meter)
    df, map_data = BaseDataset.normalize_xy(df, map_data)
    return BaseDataset(name=path.parent.name, args=args, df_data=df, map_data=map_data)
