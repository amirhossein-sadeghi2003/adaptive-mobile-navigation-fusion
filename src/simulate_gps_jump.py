from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from run_kalman_gps_baseline import make_process_noise, path_length


def run_kalman_with_gps_jump(
    df: pd.DataFrame,
    jump_start_s: float,
    jump_end_s: float,
    jump_x_m: float,
    jump_y_m: float,
    accel_std: float,
    min_gps_std: float,
) -> pd.DataFrame:
    required = {"t_s", "x_m", "y_m", "horizontalAccuracy"}
    if not required.issubset(df.columns):
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

        clean_z = np.array([float(row["x_m"]), float(row["y_m"])])
        jump_active = jump_start_s <= t <= jump_end_s

        if jump_active:
            observed_z = clean_z + np.array([jump_x_m, jump_y_m])
        else:
            observed_z = clean_z

        gps_std = max(float(row["horizontalAccuracy"]), min_gps_std)
        R = np.diag([gps_std**2, gps_std**2])

        innovation = observed_z - (H @ x_pred)
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        x = x_pred + K @ innovation
        P = (I - K @ H) @ P_pred

        residual = observed_z - (H @ x)
        position_error_to_clean_gps = float(np.linalg.norm(clean_z - (H @ x)))
        observed_gps_error_to_clean = float(np.linalg.norm(observed_z - clean_z))

        result_rows.append({
            "t_s": t,
            "clean_gps_x_m": float(clean_z[0]),
            "clean_gps_y_m": float(clean_z[1]),
            "observed_gps_x_m": float(observed_z[0]),
            "observed_gps_y_m": float(observed_z[1]),
            "gps_jump_active": bool(jump_active),
            "observed_gps_error_to_clean_m": observed_gps_error_to_clean,
            "jump_kalman_x_m": float(x[0]),
            "jump_kalman_y_m": float(x[1]),
            "jump_kalman_vx_mps": float(x[2]),
            "jump_kalman_vy_mps": float(x[3]),
            "jump_kalman_speed_mps": float(np.sqrt(x[2] ** 2 + x[3] ** 2)),
            "gps_horizontal_accuracy_m": float(row["horizontalAccuracy"]),
            "innovation_norm_m": float(np.linalg.norm(innovation)),
            "residual_norm_m": float(np.linalg.norm(residual)),
            "position_error_to_clean_gps_m": position_error_to_clean_gps,
            "kalman_position_std_x_m": float(np.sqrt(max(P[0, 0], 0.0))),
            "kalman_position_std_y_m": float(np.sqrt(max(P[1, 1], 0.0))),
        })

    out = pd.DataFrame(result_rows)

    dx = out["jump_kalman_x_m"].diff()
    dy = out["jump_kalman_y_m"].diff()
    out["jump_kalman_step_distance_m"] = np.sqrt(dx**2 + dy**2).fillna(0.0)
    out["jump_kalman_cumulative_distance_m"] = out["jump_kalman_step_distance_m"].cumsum()

    return out


def save_summary(
    raw_df: pd.DataFrame,
    jump_df: pd.DataFrame,
    output_path: Path,
    jump_start_s: float,
    jump_end_s: float,
    jump_x_m: float,
    jump_y_m: float,
) -> None:
    jump_rows = jump_df[jump_df["gps_jump_active"]]
    jump_offset_norm = float(np.sqrt(jump_x_m**2 + jump_y_m**2))

    if len(jump_rows) > 0:
        max_jump_error = round(float(jump_rows["position_error_to_clean_gps_m"].max()), 3)
        median_jump_error = round(float(jump_rows["position_error_to_clean_gps_m"].median()), 3)
        max_jump_innovation = round(float(jump_rows["innovation_norm_m"].max()), 3)
        max_jump_speed = round(float(jump_rows["jump_kalman_speed_mps"].max()), 3)
    else:
        max_jump_error = np.nan
        median_jump_error = np.nan
        max_jump_innovation = np.nan
        max_jump_speed = np.nan

    summary = {
        "rows": len(jump_df),
        "duration_s": round(float(jump_df["t_s"].iloc[-1] - jump_df["t_s"].iloc[0]), 3),
        "jump_start_s": jump_start_s,
        "jump_end_s": jump_end_s,
        "jump_rows": len(jump_rows),
        "jump_offset_x_m": jump_x_m,
        "jump_offset_y_m": jump_y_m,
        "jump_offset_norm_m": round(jump_offset_norm, 3),
        "median_gps_accuracy_m": round(float(jump_df["gps_horizontal_accuracy_m"].median()), 3),
        "raw_gps_path_length_m": round(path_length(raw_df["x_m"], raw_df["y_m"]), 3),
        "jump_kalman_path_length_m": round(float(jump_df["jump_kalman_cumulative_distance_m"].iloc[-1]), 3),
        "max_position_error_to_clean_gps_m": round(float(jump_df["position_error_to_clean_gps_m"].max()), 3),
        "max_jump_window_position_error_to_clean_gps_m": max_jump_error,
        "median_jump_window_position_error_to_clean_gps_m": median_jump_error,
        "max_jump_window_innovation_m": max_jump_innovation,
        "max_jump_window_kalman_speed_mps": max_jump_speed,
        "final_position_error_to_clean_gps_m": round(float(jump_df["position_error_to_clean_gps_m"].iloc[-1]), 3),
    }

    pd.DataFrame([summary]).to_csv(output_path, index=False)


def plot_trajectory(
    raw_df: pd.DataFrame,
    jump_df: pd.DataFrame,
    output_path: Path,
) -> None:
    jump_rows = jump_df[jump_df["gps_jump_active"]]

    plt.figure(figsize=(7, 7))

    plt.plot(raw_df["x_m"], raw_df["y_m"], marker="o", markersize=2, linewidth=1, label="clean GPS")
    plt.plot(
        jump_df["jump_kalman_x_m"],
        jump_df["jump_kalman_y_m"],
        linewidth=2,
        label="Kalman using jumped GPS",
    )

    if len(jump_rows) > 0:
        plt.scatter(
            jump_rows["observed_gps_x_m"],
            jump_rows["observed_gps_y_m"],
            s=35,
            label="jumped GPS samples",
        )

    plt.scatter(raw_df["x_m"].iloc[0], raw_df["y_m"].iloc[0], s=80, label="start")
    plt.scatter(raw_df["x_m"].iloc[-1], raw_df["y_m"].iloc[-1], marker="x", s=80, label="end")

    plt.title("Kalman trajectory with simulated GPS jump")
    plt.xlabel("East displacement from start (m)")
    plt.ylabel("North displacement from start (m)")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_error(
    jump_df: pd.DataFrame,
    output_path: Path,
    jump_start_s: float,
    jump_end_s: float,
) -> None:
    plt.figure(figsize=(10, 5))

    plt.plot(
        jump_df["t_s"],
        jump_df["position_error_to_clean_gps_m"],
        label="Kalman position difference from clean GPS",
    )
    plt.plot(
        jump_df["t_s"],
        jump_df["innovation_norm_m"],
        label="innovation before update",
        alpha=0.8,
    )
    plt.axvspan(jump_start_s, jump_end_s, alpha=0.2, label="GPS jump window")

    plt.title("Effect of a simulated GPS position jump")
    plt.xlabel("Time (s)")
    plt.ylabel("Meters")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_speed(
    raw_df: pd.DataFrame,
    jump_df: pd.DataFrame,
    output_path: Path,
    jump_start_s: float,
    jump_end_s: float,
) -> None:
    plt.figure(figsize=(10, 5))

    if "reported_speed_mps" in raw_df.columns:
        plt.plot(raw_df["t_s"], raw_df["reported_speed_mps"], label="reported GPS speed")

    plt.plot(raw_df["t_s"], raw_df["computed_speed_mps"], label="computed from clean GPS points", alpha=0.6)
    plt.plot(jump_df["t_s"], jump_df["jump_kalman_speed_mps"], linewidth=2, label="Kalman speed with GPS jump")
    plt.axvspan(jump_start_s, jump_end_s, alpha=0.2, label="GPS jump window")

    plt.title("Speed behavior during simulated GPS jump")
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/gps_walk_02_gps_baseline.csv")
    parser.add_argument("--label", default="gps_walk_02")
    parser.add_argument("--jump-start", type=float, default=80.0)
    parser.add_argument("--jump-end", type=float, default=90.0)
    parser.add_argument("--jump-x", type=float, default=20.0)
    parser.add_argument("--jump-y", type=float, default=-10.0)
    parser.add_argument("--accel-std", type=float, default=0.8)
    parser.add_argument("--min-gps-std", type=float, default=3.0)
    args = parser.parse_args()

    input_path = Path(args.input)
    results_dir = Path("results")
    figures_dir = Path("figures")
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    raw_df = pd.read_csv(input_path)

    jump_df = run_kalman_with_gps_jump(
        raw_df,
        jump_start_s=args.jump_start,
        jump_end_s=args.jump_end,
        jump_x_m=args.jump_x,
        jump_y_m=args.jump_y,
        accel_std=args.accel_std,
        min_gps_std=args.min_gps_std,
    )

    output_csv = results_dir / f"{args.label}_jump_kalman.csv"
    summary_csv = results_dir / f"{args.label}_jump_kalman_summary.csv"

    jump_df.to_csv(output_csv, index=False)
    save_summary(raw_df, jump_df, summary_csv, args.jump_start, args.jump_end, args.jump_x, args.jump_y)

    plot_trajectory(raw_df, jump_df, figures_dir / f"{args.label}_jump_trajectory.png")
    plot_error(jump_df, figures_dir / f"{args.label}_jump_error.png", args.jump_start, args.jump_end)
    plot_speed(raw_df, jump_df, figures_dir / f"{args.label}_jump_speed.png", args.jump_start, args.jump_end)

    print(f"Saved GPS jump Kalman results: {output_csv}")
    print(f"Saved summary: {summary_csv}")
    print("Saved figures:")
    for path in sorted(figures_dir.glob(f"{args.label}_jump_*.png")):
        print(f"  {path}")


if __name__ == "__main__":
    main()
