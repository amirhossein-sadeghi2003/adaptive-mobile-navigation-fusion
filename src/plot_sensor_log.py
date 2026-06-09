from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_sensor_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "seconds_elapsed" not in df.columns:
        raise ValueError(f"{path.name} does not contain seconds_elapsed column")

    df = df.copy()
    df["t"] = df["seconds_elapsed"] - df["seconds_elapsed"].iloc[0]
    return df


def summarize_file(path: Path) -> dict:
    df = pd.read_csv(path)
    duration = float(df["seconds_elapsed"].iloc[-1] - df["seconds_elapsed"].iloc[0])
    dt = df["seconds_elapsed"].diff().dropna()
    median_dt = float(dt.median()) if len(dt) else np.nan
    sample_rate = float(1.0 / median_dt) if median_dt and median_dt > 0 else np.nan

    return {
        "file": path.name,
        "rows": len(df),
        "duration_s": round(duration, 3),
        "median_dt_s": round(median_dt, 5),
        "approx_sample_rate_hz": round(sample_rate, 2),
        "columns": ", ".join(df.columns),
    }


def plot_xyz(df: pd.DataFrame, title: str, ylabel: str, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))

    for axis in ["x", "y", "z"]:
        if axis in df.columns:
            plt.plot(df["t"], df[axis], label=axis)

    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_orientation(df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))

    for angle in ["roll", "pitch", "yaw"]:
        if angle in df.columns:
            plt.plot(df["t"], np.degrees(df[angle]), label=angle)

    plt.title("Phone orientation during room test")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (degrees)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_compass(df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))

    plt.plot(df["t"], df["magneticBearing"], label="magnetic bearing")
    plt.title("Compass bearing during room test")
    plt.xlabel("Time (s)")
    plt.ylabel("Bearing (degrees)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_acceleration_norm(df: pd.DataFrame, output_path: Path) -> None:
    required = {"x", "y", "z"}
    if not required.issubset(df.columns):
        return

    norm = np.sqrt(df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2)

    plt.figure(figsize=(10, 5))
    plt.plot(df["t"], norm, label="acceleration norm")
    plt.title("Total acceleration magnitude during room test")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration magnitude (m/s²)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/private/room_test_01")
    parser.add_argument("--label", default="room_test_01")
    args = parser.parse_args()

    input_dir = Path(args.input)
    figures_dir = Path("figures")
    results_dir = Path("results")
    figures_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    summary = [summarize_file(path) for path in csv_files]
    summary_df = pd.DataFrame(summary)
    summary_path = results_dir / f"{args.label}_sensor_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    accel_path = input_dir / "Accelerometer.csv"
    gyro_path = input_dir / "Gyroscope.csv"
    orientation_path = input_dir / "Orientation.csv"
    compass_path = input_dir / "Compass.csv"
    total_accel_path = input_dir / "TotalAcceleration.csv"

    if accel_path.exists():
        accel = read_sensor_csv(accel_path)
        plot_xyz(
            accel,
            "Phone accelerometer during room test",
            "Linear acceleration (m/s²)",
            figures_dir / f"{args.label}_accelerometer.png",
        )

    if gyro_path.exists():
        gyro = read_sensor_csv(gyro_path)
        plot_xyz(
            gyro,
            "Phone gyroscope during room test",
            "Angular velocity (rad/s)",
            figures_dir / f"{args.label}_gyroscope.png",
        )

    if orientation_path.exists():
        orientation = read_sensor_csv(orientation_path)
        plot_orientation(
            orientation,
            figures_dir / f"{args.label}_orientation.png",
        )

    if compass_path.exists():
        compass = read_sensor_csv(compass_path)
        plot_compass(
            compass,
            figures_dir / f"{args.label}_compass.png",
        )

    if total_accel_path.exists():
        total_accel = read_sensor_csv(total_accel_path)
        plot_acceleration_norm(
            total_accel,
            figures_dir / f"{args.label}_total_acceleration_norm.png",
        )

    print(f"Saved summary: {summary_path}")
    print("Saved figures:")
    for path in sorted(figures_dir.glob(f"{args.label}_*.png")):
        print(f"  {path}")


if __name__ == "__main__":
    main()
