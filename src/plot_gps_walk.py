from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


EARTH_RADIUS_M = 6371000.0


def load_location(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"seconds_elapsed", "latitude", "longitude", "horizontalAccuracy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in Location.csv: {sorted(missing)}")

    df = df.copy()
    df["t"] = df["seconds_elapsed"] - df["seconds_elapsed"].iloc[0]

    lat0 = np.deg2rad(df["latitude"].iloc[0])
    lon0 = np.deg2rad(df["longitude"].iloc[0])

    lat = np.deg2rad(df["latitude"])
    lon = np.deg2rad(df["longitude"])

    df["x_m"] = EARTH_RADIUS_M * (lon - lon0) * np.cos(lat0)
    df["y_m"] = EARTH_RADIUS_M * (lat - lat0)

    dx = df["x_m"].diff()
    dy = df["y_m"].diff()
    df["step_distance_m"] = np.sqrt(dx**2 + dy**2).fillna(0.0)
    df["distance_from_start_m"] = np.sqrt(df["x_m"]**2 + df["y_m"]**2)

    return df


def save_summary(df: pd.DataFrame, output_path: Path) -> None:
    duration = float(df["t"].iloc[-1] - df["t"].iloc[0])
    dt = df["seconds_elapsed"].diff().dropna()
    median_dt = float(dt.median()) if len(dt) else np.nan
    sample_rate = float(1.0 / median_dt) if median_dt > 0 else np.nan

    summary = {
        "rows": [len(df)],
        "duration_s": [round(duration, 3)],
        "approx_sample_rate_hz": [round(sample_rate, 3)],
        "median_horizontal_accuracy_m": [round(float(df["horizontalAccuracy"].median()), 3)],
        "max_distance_from_start_m": [round(float(df["distance_from_start_m"].max()), 3)],
        "total_gps_path_length_m": [round(float(df["step_distance_m"].sum()), 3)],
    }

    if "speed" in df.columns:
        summary["median_reported_speed_mps"] = [round(float(df["speed"].median()), 3)]
        summary["max_reported_speed_mps"] = [round(float(df["speed"].max()), 3)]

    pd.DataFrame(summary).to_csv(output_path, index=False)


def plot_trajectory(df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(7, 7))
    plt.plot(df["x_m"], df["y_m"], marker="o", markersize=3, linewidth=1, label="GPS path")
    plt.scatter(df["x_m"].iloc[0], df["y_m"].iloc[0], marker="o", s=80, label="start")
    plt.scatter(df["x_m"].iloc[-1], df["y_m"].iloc[-1], marker="x", s=80, label="end")

    plt.title("GPS walk local trajectory")
    plt.xlabel("East displacement from start (m)")
    plt.ylabel("North displacement from start (m)")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_filtered_trajectory(df: pd.DataFrame, output_path: Path, max_accuracy_m: float = 8.0) -> None:
    filtered = df[df["horizontalAccuracy"] <= max_accuracy_m].copy()

    if len(filtered) < 2:
        return

    lat0 = np.deg2rad(filtered["latitude"].iloc[0])
    lon0 = np.deg2rad(filtered["longitude"].iloc[0])

    lat = np.deg2rad(filtered["latitude"])
    lon = np.deg2rad(filtered["longitude"])

    filtered["x_filtered_m"] = EARTH_RADIUS_M * (lon - lon0) * np.cos(lat0)
    filtered["y_filtered_m"] = EARTH_RADIUS_M * (lat - lat0)

    plt.figure(figsize=(7, 7))
    plt.plot(
        filtered["x_filtered_m"],
        filtered["y_filtered_m"],
        marker="o",
        markersize=3,
        linewidth=1,
        label=f"GPS path, accuracy <= {max_accuracy_m:g} m",
    )
    plt.scatter(filtered["x_filtered_m"].iloc[0], filtered["y_filtered_m"].iloc[0], marker="o", s=80, label="filtered start")
    plt.scatter(filtered["x_filtered_m"].iloc[-1], filtered["y_filtered_m"].iloc[-1], marker="x", s=80, label="filtered end")

    plt.title("GPS walk local trajectory after accuracy filtering")
    plt.xlabel("East displacement from filtered start (m)")
    plt.ylabel("North displacement from filtered start (m)")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_accuracy(df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(df["t"], df["horizontalAccuracy"], label="horizontal accuracy")
    plt.title("GPS horizontal accuracy during walk")
    plt.xlabel("Time (s)")
    plt.ylabel("Accuracy estimate (m)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_speed(df: pd.DataFrame, output_path: Path) -> None:
    if "speed" not in df.columns:
        return

    plt.figure(figsize=(10, 5))
    plt.plot(df["t"], df["speed"], label="reported GPS speed")
    plt.title("Reported GPS speed during walk")
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/private/gps_walk_01/Location.csv")
    parser.add_argument("--label", default="gps_walk_01")
    args = parser.parse_args()

    location_path = Path(args.input)
    figures_dir = Path("figures")
    results_dir = Path("results")
    figures_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    df = load_location(location_path)

    save_summary(df, results_dir / f"{args.label}_location_summary.csv")
    plot_trajectory(df, figures_dir / f"{args.label}_local_trajectory.png")
    plot_filtered_trajectory(df, figures_dir / f"{args.label}_filtered_trajectory.png")
    plot_accuracy(df, figures_dir / f"{args.label}_gps_accuracy.png")
    plot_speed(df, figures_dir / f"{args.label}_gps_speed.png")

    print("Saved:")
    print(f"  results/{args.label}_location_summary.csv")
    print(f"  figures/{args.label}_local_trajectory.png")
    print(f"  figures/{args.label}_gps_accuracy.png")
    print(f"  figures/{args.label}_gps_speed.png")


if __name__ == "__main__":
    main()
