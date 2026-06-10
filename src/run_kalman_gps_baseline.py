from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def make_process_noise(dt: float, accel_std: float) -> np.ndarray:
    q = accel_std ** 2

    return q * np.array([
        [dt**4 / 4, 0.0, dt**3 / 2, 0.0],
        [0.0, dt**4 / 4, 0.0, dt**3 / 2],
        [dt**3 / 2, 0.0, dt**2, 0.0],
        [0.0, dt**3 / 2, 0.0, dt**2],
    ])


def run_kalman_filter(df: pd.DataFrame, accel_std: float, min_gps_std: float) -> pd.DataFrame:
    if not {"t_s", "x_m", "y_m", "horizontalAccuracy"}.issubset(df.columns):
        raise ValueError("Input CSV must contain t_s, x_m, y_m, and horizontalAccuracy")

    result_rows = []

    x = np.array([df["x_m"].iloc[0], df["y_m"].iloc[0], 0.0, 0.0], dtype=float)
    P = np.diag([10.0, 10.0, 5.0, 5.0])

    H = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ])

    I = np.eye(4)

    prev_t = float(df["t_s"].iloc[0])

    for _, row in df.iterrows():
        t = float(row["t_s"])
        dt = max(t - prev_t, 1e-3)
        prev_t = t

        F = np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        Q = make_process_noise(dt, accel_std)

        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        z = np.array([float(row["x_m"]), float(row["y_m"])])
        gps_std = max(float(row["horizontalAccuracy"]), min_gps_std)
        R = np.diag([gps_std**2, gps_std**2])

        y = z - (H @ x_pred)
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        x = x_pred + K @ y
        P = (I - K @ H) @ P_pred

        residual = z - (H @ x)
        innovation_norm = float(np.linalg.norm(y))
        residual_norm = float(np.linalg.norm(residual))

        result_rows.append({
            "t_s": t,
            "gps_x_m": float(row["x_m"]),
            "gps_y_m": float(row["y_m"]),
            "kalman_x_m": float(x[0]),
            "kalman_y_m": float(x[1]),
            "kalman_vx_mps": float(x[2]),
            "kalman_vy_mps": float(x[3]),
            "kalman_speed_mps": float(np.sqrt(x[2] ** 2 + x[3] ** 2)),
            "gps_horizontal_accuracy_m": float(row["horizontalAccuracy"]),
            "innovation_norm_m": innovation_norm,
            "residual_norm_m": residual_norm,
            "kalman_position_std_x_m": float(np.sqrt(max(P[0, 0], 0.0))),
            "kalman_position_std_y_m": float(np.sqrt(max(P[1, 1], 0.0))),
        })

    out = pd.DataFrame(result_rows)

    dx = out["kalman_x_m"].diff()
    dy = out["kalman_y_m"].diff()
    out["kalman_step_distance_m"] = np.sqrt(dx**2 + dy**2).fillna(0.0)
    out["kalman_cumulative_distance_m"] = out["kalman_step_distance_m"].cumsum()

    return out


def path_length(x: pd.Series, y: pd.Series) -> float:
    dx = x.diff()
    dy = y.diff()
    return float(np.sqrt(dx**2 + dy**2).fillna(0.0).sum())


def save_summary(raw_df: pd.DataFrame, kf_df: pd.DataFrame, output_path: Path) -> None:
    summary = {
        "rows": len(kf_df),
        "duration_s": round(float(kf_df["t_s"].iloc[-1] - kf_df["t_s"].iloc[0]), 3),
        "median_gps_accuracy_m": round(float(kf_df["gps_horizontal_accuracy_m"].median()), 3),
        "raw_gps_path_length_m": round(path_length(raw_df["x_m"], raw_df["y_m"]), 3),
        "kalman_path_length_m": round(float(kf_df["kalman_cumulative_distance_m"].iloc[-1]), 3),
        "median_innovation_m": round(float(kf_df["innovation_norm_m"].median()), 3),
        "median_residual_m": round(float(kf_df["residual_norm_m"].median()), 3),
        "max_raw_computed_speed_mps": round(float(raw_df["computed_speed_mps"].max()), 3),
        "max_kalman_speed_mps": round(float(kf_df["kalman_speed_mps"].max()), 3),
        "median_kalman_speed_mps": round(float(kf_df["kalman_speed_mps"].median()), 3),
    }

    pd.DataFrame([summary]).to_csv(output_path, index=False)


def plot_trajectory(raw_df: pd.DataFrame, kf_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(7, 7))

    plt.plot(raw_df["x_m"], raw_df["y_m"], marker="o", markersize=2, linewidth=1, label="raw GPS")
    plt.plot(kf_df["kalman_x_m"], kf_df["kalman_y_m"], linewidth=2, label="Kalman estimate")
    plt.scatter(raw_df["x_m"].iloc[0], raw_df["y_m"].iloc[0], s=80, label="start")
    plt.scatter(raw_df["x_m"].iloc[-1], raw_df["y_m"].iloc[-1], marker="x", s=80, label="end")

    plt.title("GPS-only Kalman trajectory baseline")
    plt.xlabel("East displacement from start (m)")
    plt.ylabel("North displacement from start (m)")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_speed(raw_df: pd.DataFrame, kf_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))

    if "reported_speed_mps" in raw_df.columns:
        plt.plot(raw_df["t_s"], raw_df["reported_speed_mps"], label="reported GPS speed")

    plt.plot(raw_df["t_s"], raw_df["computed_speed_mps"], label="computed from GPS points", alpha=0.6)
    plt.plot(kf_df["t_s"], kf_df["kalman_speed_mps"], linewidth=2, label="Kalman speed")

    plt.title("Speed comparison with GPS-only Kalman baseline")
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_residuals(kf_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))

    plt.plot(kf_df["t_s"], kf_df["innovation_norm_m"], label="innovation before update")
    plt.plot(kf_df["t_s"], kf_df["residual_norm_m"], label="residual after update")
    plt.plot(kf_df["t_s"], kf_df["gps_horizontal_accuracy_m"], label="GPS accuracy estimate", alpha=0.8)

    plt.title("Kalman residuals and GPS accuracy")
    plt.xlabel("Time (s)")
    plt.ylabel("Meters")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/gps_walk_02_gps_baseline.csv")
    parser.add_argument("--label", default="gps_walk_02")
    parser.add_argument("--accel-std", type=float, default=0.8)
    parser.add_argument("--min-gps-std", type=float, default=3.0)
    args = parser.parse_args()

    input_path = Path(args.input)
    results_dir = Path("results")
    figures_dir = Path("figures")
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    raw_df = pd.read_csv(input_path)
    kf_df = run_kalman_filter(raw_df, accel_std=args.accel_std, min_gps_std=args.min_gps_std)

    output_csv = results_dir / f"{args.label}_kalman_baseline.csv"
    summary_csv = results_dir / f"{args.label}_kalman_baseline_summary.csv"

    kf_df.to_csv(output_csv, index=False)
    save_summary(raw_df, kf_df, summary_csv)

    plot_trajectory(raw_df, kf_df, figures_dir / f"{args.label}_kalman_trajectory.png")
    plot_speed(raw_df, kf_df, figures_dir / f"{args.label}_kalman_speed_compare.png")
    plot_residuals(kf_df, figures_dir / f"{args.label}_kalman_residuals.png")

    print(f"Saved Kalman baseline: {output_csv}")
    print(f"Saved summary: {summary_csv}")
    print("Saved figures:")
    for path in sorted(figures_dir.glob(f"{args.label}_kalman_*.png")):
        print(f"  {path}")


if __name__ == "__main__":
    main()
