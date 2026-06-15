"""Build a time-aligned GPS + IMU dataset for one SensorLog recording.

SensorLog writes each sensor to its own CSV but on a shared clock
(seconds_elapsed). GPS (Location.csv) is the slowest stream here (~2 Hz);
the IMU streams run far faster (~100 Hz). This script attaches every IMU
stream to the GPS timestamps with a nearest-time match, so each GPS sample
carries the IMU values recorded closest to it.

This is the foundation for fusing IMU heading/motion with GPS. It does not
interpret or convert any IMU units yet; that is decided later from the data.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


EARTH_RADIUS_M = 6371000.0


def load_location(recording_dir: Path) -> pd.DataFrame:
    """Load Location.csv and project lat/lon to local east/north metres.

    Uses the same equirectangular projection as build_gps_baseline.py, so x_m
    and y_m mean exactly what they mean in the GPS-only baseline.
    """
    path = recording_dir / "Location.csv"
    df = pd.read_csv(path).sort_values("seconds_elapsed").reset_index(drop=True)

    required = {"seconds_elapsed", "latitude", "longitude", "horizontalAccuracy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

    lat0 = np.deg2rad(df["latitude"].iloc[0])
    lon0 = np.deg2rad(df["longitude"].iloc[0])
    lat = np.deg2rad(df["latitude"])
    lon = np.deg2rad(df["longitude"])

    out = pd.DataFrame()
    out["t_raw_s"] = df["seconds_elapsed"].astype(float)
    out["x_m"] = EARTH_RADIUS_M * (lon - lon0) * np.cos(lat0)
    out["y_m"] = EARTH_RADIUS_M * (lat - lat0)
    out["horizontalAccuracy"] = df["horizontalAccuracy"].astype(float)
    out["gps_bearing"] = df["bearing"].astype(float) if "bearing" in df.columns else np.nan
    out["gps_speed_mps"] = df["speed"].astype(float) if "speed" in df.columns else np.nan
    return out


def load_stream(recording_dir: Path, filename: str, value_cols: dict) -> pd.DataFrame:
    """Load one IMU CSV, keep seconds_elapsed plus the renamed value columns."""
    path = recording_dir / filename
    df = pd.read_csv(path).sort_values("seconds_elapsed").reset_index(drop=True)

    needed = {"seconds_elapsed", *value_cols}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

    out = pd.DataFrame({"t_raw_s": df["seconds_elapsed"].astype(float)})
    for src, dst in value_cols.items():
        out[dst] = df[src].astype(float)
    return out


def merge_stream(base: pd.DataFrame, stream: pd.DataFrame, name: str, tolerance_s: float) -> pd.DataFrame:
    """Attach a stream to base by nearest GPS timestamp and report match quality.

    Any GPS row with no IMU sample within tolerance_s keeps NaN for that stream,
    which makes genuine gaps visible instead of silently filled.
    """
    key = f"_{name}_t_raw_s"
    stream = stream.rename(columns={"t_raw_s": key})

    merged = pd.merge_asof(
        base,
        stream,
        left_on="t_raw_s",
        right_on=key,
        direction="nearest",
        tolerance=tolerance_s,
    )

    offset = (merged["t_raw_s"] - merged[key]).abs()
    matched = int(offset.notna().sum())
    worst = float(offset.max()) if matched else float("nan")
    print(f"  {name:<12} matched {matched}/{len(merged)} GPS rows, max |offset| = {worst:.4f} s")

    return merged.drop(columns=[key])


def build_fusion_dataset(recording_dir: Path, tolerance_s: float) -> pd.DataFrame:
    gps = load_location(recording_dir)
    print(f"Loaded {len(gps)} GPS samples from {recording_dir}")
    print("Aligning IMU streams to GPS timestamps (nearest match):")

    streams = [
        ("Orientation.csv", {"yaw": "orientation_yaw", "roll": "orientation_roll", "pitch": "orientation_pitch"}, "orientation"),
        ("Compass.csv", {"magneticBearing": "compass_bearing"}, "compass"),
        ("Gyroscope.csv", {"z": "gyro_z", "y": "gyro_y", "x": "gyro_x"}, "gyro"),
        ("Accelerometer.csv", {"z": "acc_z", "y": "acc_y", "x": "acc_x"}, "accel"),
    ]

    fused = gps
    for filename, cols, name in streams:
        stream = load_stream(recording_dir, filename, cols)
        fused = merge_stream(fused, stream, name, tolerance_s)

    fused.insert(0, "t_s", fused["t_raw_s"] - fused["t_raw_s"].iloc[0])
    return fused


def save_summary(df: pd.DataFrame, output_path: Path) -> None:
    summary = {
        "rows": len(df),
        "duration_s": round(float(df["t_s"].iloc[-1] - df["t_s"].iloc[0]), 3),
        "median_horizontal_accuracy_m": round(float(df["horizontalAccuracy"].median()), 3),
        "rows_missing_orientation": int(df["orientation_yaw"].isna().sum()),
        "rows_missing_compass": int(df["compass_bearing"].isna().sum()),
        "rows_missing_gyro": int(df["gyro_z"].isna().sum()),
        "rows_missing_accel": int(df["acc_z"].isna().sum()),
    }
    pd.DataFrame([summary]).to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/private/gps_walk_02")
    parser.add_argument("--label", default="gps_walk_02")
    parser.add_argument("--tolerance-s", type=float, default=0.1)
    args = parser.parse_args()

    recording_dir = Path(args.input)
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    fused = build_fusion_dataset(recording_dir, tolerance_s=args.tolerance_s)

    output_csv = results_dir / f"{args.label}_fusion.csv"
    summary_csv = results_dir / f"{args.label}_fusion_summary.csv"
    fused.to_csv(output_csv, index=False)
    save_summary(fused, summary_csv)

    print(f"\nSaved fused dataset: {output_csv}  ({len(fused)} rows, {len(fused.columns)} columns)")
    print(f"Saved summary: {summary_csv}")
    print("Columns:", ", ".join(fused.columns))


if __name__ == "__main__":
    main()