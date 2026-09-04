"""Waymo TFRecord conversion extracted from data/WayMo/process_data.ipynb."""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

__all__ = ["process_waymo_files"]


def scenario_dataframe(scenario) -> pd.DataFrame:
    rows = []
    for track in scenario.tracks:
        for frame, state in enumerate(track.states):
            if state.valid:
                rows.append({
                    "center_x": state.center_x, "center_y": state.center_y,
                    "center_z": state.center_z, "length": state.length,
                    "width": state.width, "height": state.height,
                    "heading": state.heading, "velocity_x": state.velocity_x,
                    "velocity_y": state.velocity_y, "f": frame,
                    "id": track.id, "type": track.object_type,
                })
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Waymo scenario contains no valid track states.")
    df["type"] = df["type"].map({
        0: "UNSET", 1: "VEHICLE", 2: "PEDESTRIAN", 3: "CYCLIST", 4: "OTHER"
    })
    return df


def render_maps(scenario, df: pd.DataFrame, map_path: Path, demo_path: Path) -> tuple[float, ...]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    features = {key: [] for key in ("lane", "road_line", "road_edge", "crosswalk", "driveway")}
    for item in scenario.map_features:
        for key in features:
            if item.HasField(key):
                geometry = getattr(item, key)
                points = geometry.polygon if key in ("crosswalk", "driveway") else geometry.polyline
                coords = [(point.x, point.y) for point in points]
                if key in ("crosswalk", "driveway") and coords:
                    coords.append(coords[0])
                features[key].append(coords)
                break

    xmin, xmax = float(df.center_x.min()), float(df.center_x.max())
    ymin, ymax = float(df.center_y.min()), float(df.center_y.max())
    dx, dy = max(xmax - xmin, 1.0), max(ymax - ymin, 1.0)
    xmin, xmax, ymin, ymax = xmin - dx * .05, xmax + dx * .05, ymin - dy * .05, ymax + dy * .05

    def draw(include_tracks: bool, destination: Path) -> None:
        fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), aspect="equal")
        ax.axis("off")
        for key in ("lane", "road_line", "road_edge"):
            for points in features[key]:
                if points:
                    ax.plot(*np.asarray(points).T, color=(.8, .8, .8), linewidth=1)
        for key in ("driveway", "crosswalk"):
            for points in features[key]:
                if points:
                    ax.add_patch(Polygon(np.asarray(points), facecolor=(.9, .9, .9), edgecolor=None))
        if include_tracks:
            for _, group in df.groupby("id"):
                color = "red" if group.type.iloc[0] == "PEDESTRIAN" else "blue"
                ax.plot(group.center_x, group.center_y, color=color, linewidth=.6, zorder=10)
        fig.savefig(destination, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    draw(False, map_path)
    draw(True, demo_path)
    return xmin, xmax, ymin, ymax


def process_waymo_files(files: list[Path], output_root: Path) -> list[Path]:
    """Convert every scenario in uploaded Waymo TFRecords into notebook-compatible assets."""
    try:
        import tensorflow as tf
        from waymo_open_dataset.protos import scenario_pb2
    except ImportError as exc:
        raise RuntimeError(
            "Waymo processing requires tensorflow and waymo-open-dataset to be installed."
        ) from exc

    outputs = []
    for source in files:
        records = tf.data.TFRecordDataset(str(source), compression_type="")
        for index, record in enumerate(records):
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(record.numpy())
            safe_id = "".join(c for c in scenario.scenario_id if c.isalnum() or c in "_-")
            source_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in source.name)
            target = output_root / f"{source_name}_{index:05d}_{safe_id}"
            target.mkdir(parents=True, exist_ok=True)
            df = scenario_dataframe(scenario)
            bounds = render_maps(scenario, df, target / "map.png", target / "demo.png")
            df.to_csv(target / "data.csv.gz", index=False, compression="gzip")
            (target / "map_range.txt").write_text(" ".join(map(str, bounds)) + "\n")
            outputs.append(target / "data.csv.gz")
    return outputs
