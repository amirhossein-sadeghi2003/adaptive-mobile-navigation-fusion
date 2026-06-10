from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


EARTH_RADIUS_M = 6371000.0


def load_location_as_local_xy(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {
        "seconds_elapsed",
        "latitude",
        "longitude",
        "horizontalAccuracy",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in Location.csv: {sorted(missing)}")

    df = df.copy()
    df["t_s"] = df["seconds_elapsed"] - df["seconds_elapsed"].iloc[0]

    lat0 = np.deg2rad(df["latitude"].iloc[0])
    lon0 = np.deg2rad(df["longitude"].iloc[0])

    lat = np.deg2rad(df["latitude"])
    lon = np.deg2rad(df["longitude"])

    df["x_m"] = EARTH_RADIUS_M * (lon - lon0) * np.cos(lat0)
    df["y_m"] = EARTH_RADIUS_M * (lat - lat0)

    dx = df["x_m"].diff()
    dy = df["y_m"].diff()
    dt = df["t_s"].diff()

    df["step_distance_m"] = np.sqrt(dx**2 + dy**2).fillna(0.0)
    df["computed_speed_mps"] = (df["step_distance_m"] / dt).replace([np.inf, -np.inf], np.nan)
    df["computed_speed_mps"] = df["computed_speed_mps"].fillna(0.0)

    df["distance_from_start_m"] = np.sqrt(df["x_m"]**2 + df["y_m"]**2)
    df["cumulative_distance_m"] = df["step_distance_m"].cumsum()

    if "speed" in df.columns:
        df = df.rename(columns={"speed": "reported_speed_mps"})
    else:
        df["reported_speed_mps"] = np.nan

    keep_cols = [
        "t_s",
        "x_m",
        "y_m",
        "horizontalAccuracy",
        "reported_speed_mps",
        "computed_speed_mps",
        "step_distance_m",
        "distance_from_start_m",
        "cumulative_distance_m",
    ]

    return df[keep_cols]


def save_summary(df: pd.DataFrame, output_path: Path) -> None:
    summary = {
        "rows": len(df),
        "duration_s": round(float(df["t_s"].iloc[-1] - df["t_s"].iloc[0]), 3),
        "median_horizontal_accuracy_m": round(float(df["horizontalAccuracy"].median()), 3),
        "max_distance_from_start_m": round(float(df["distance_from_start_m"].max()), 3),
        "total_gps_path_length_m": round(float(df["cumulative_distance_m"].iloc[-1]), 3),
        "median_reported_speed_mps": round(float(df["reported_speed_mps"].median()), 3),
        "median_computed_speed_mps": round(float(df["computed_speed_mps"].median()), 3),
        "max_computed_speed_mps": round(float(df["computed_speed_mps"].max()), 3),
    }

    pd.DataFrame([summary]).to_csv(output_path, index=False)


def plot_speed_comparison(df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(df["t_s"], df["reported_speed_mps"], label="reported GPS speed")
    plt.plot(df["t_s"], df["computed_speed_mps"], label="computed from GPS positions", alpha=0.8)
    plt.title("GPS-only speed baseline")
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_cumulative_distance(df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(df["t_s"], df["cumulative_distance_m"], label="GPS cumulative distance")
    plt.title("GPS-only cumulative path length")
    plt.xlabel("Time (s)")
    plt.ylabel("Cumulative distance (m)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_raw_vs_smoothed_trajectory(df: pd.DataFrame, output_path: Path) -> None:
    smooth = df.copy()
    smooth["x_smooth_m"] = smooth["x_m"].rolling(window=7, center=True, min_periods=1).mean()
    smooth["y_smooth_m"] = smooth["y_m"].rolling(window=7, center=True, min_periods=1).mean()

    plt.figure(figsize=(7, 7))
    plt.plot(df["x_m"], df["y_m"], marker="o", markersize=2, linewidth=1, label="raw GPS local path")
    plt.plot(smooth["x_smooth_m"], smooth["y_smooth_m"], linewidth=2, label="rolling mean path")
    plt.scatter(df["x_m"].iloc[0], df["y_m"].iloc[0], s=80, label="start")
    plt.scatter(df["x_m"].iloc[-1], df["y_m"].iloc[-1], marker="x", s=80, label="end")
    plt.title("GPS-only trajectory baseline")
    plt.xlabel("East displacement from start (m)")
    plt.ylabel("North displacement from start (m)")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/private/gps_walk_02/Location.csv")
    parser.add_argument("--label", default="gps_walk_02")
    args = parser.parse_args()

    results_dir = Path("results")
    figures_dir = Path("figures")
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    df = load_location_as_local_xy(Path(args.input))

    baseline_path = results_dir / f"{args.label}_gps_baseline.csv"
    summary_path = results_dir / f"{args.label}_gps_baseline_summary.csv"

    df.to_csv(baseline_path, index=False)
    save_summary(df, summary_path)

    plot_speed_comparison(df, figures_dir / f"{args.label}_gps_baseline_speed_compare.png")
    plot_cumulative_distance(df, figures_dir / f"{args.label}_gps_baseline_cumulative_distance.png")
    plot_raw_vs_smoothed_trajectory(df, figures_dir / f"{args.label}_gps_baseline_trajectory.png")

    print(f"Saved baseline: {baseline_path}")
    print(f"Saved summary: {summary_path}")
    print("Saved figures:")
    for path in sorted(figures_dir.glob(f"{args.label}_gps_baseline_*.png")):
        print(f"  {path}")


if __name__ == "__main__":
    main()
