from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from run_kalman_gps_baseline import make_process_noise, path_length


def run_kalman_with_dropout(
    df: pd.DataFrame,
    dropout_start_s: float,
    dropout_end_s: float,
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

        z = np.array([float(row["x_m"]), float(row["y_m"])])
        gps_std = max(float(row["horizontalAccuracy"]), min_gps_std)
        R = np.diag([gps_std**2, gps_std**2])

        innovation = z - (H @ x_pred)
        innovation_norm = float(np.linalg.norm(innovation))

        dropout_active = dropout_start_s <= t <= dropout_end_s

        if dropout_active:
            x = x_pred
            P = P_pred
            gps_update_used = False
            residual_norm = np.nan
        else:
            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T @ np.linalg.inv(S)

            x = x_pred + K @ innovation
            P = (I - K @ H) @ P_pred

            residual = z - (H @ x)
            gps_update_used = True
            residual_norm = float(np.linalg.norm(residual))

        position_error_to_gps = float(np.linalg.norm(z - (H @ x)))

        result_rows.append({
            "t_s": t,
            "gps_x_m": float(row["x_m"]),
            "gps_y_m": float(row["y_m"]),
            "dropout_active": bool(dropout_active),
            "gps_update_used": bool(gps_update_used),
            "dropout_kalman_x_m": float(x[0]),
            "dropout_kalman_y_m": float(x[1]),
            "dropout_kalman_vx_mps": float(x[2]),
            "dropout_kalman_vy_mps": float(x[3]),
            "dropout_kalman_speed_mps": float(np.sqrt(x[2] ** 2 + x[3] ** 2)),
            "gps_horizontal_accuracy_m": float(row["horizontalAccuracy"]),
            "innovation_norm_m": innovation_norm,
            "residual_norm_m": residual_norm,
            "position_error_to_gps_m": position_error_to_gps,
            "kalman_position_std_x_m": float(np.sqrt(max(P[0, 0], 0.0))),
            "kalman_position_std_y_m": float(np.sqrt(max(P[1, 1], 0.0))),
        })

    out = pd.DataFrame(result_rows)

    dx = out["dropout_kalman_x_m"].diff()
    dy = out["dropout_kalman_y_m"].diff()
    out["dropout_kalman_step_distance_m"] = np.sqrt(dx**2 + dy**2).fillna(0.0)
    out["dropout_kalman_cumulative_distance_m"] = out["dropout_kalman_step_distance_m"].cumsum()

    return out


def save_summary(
    raw_df: pd.DataFrame,
    dropout_df: pd.DataFrame,
    output_path: Path,
    dropout_start_s: float,
    dropout_end_s: float,
) -> None:
    dropout_rows = dropout_df[dropout_df["dropout_active"]]
    position_std = np.sqrt(
        dropout_df["kalman_position_std_x_m"] ** 2
        + dropout_df["kalman_position_std_y_m"] ** 2
    )

    if len(dropout_rows) > 0:
        dropout_position_std = np.sqrt(
            dropout_rows["kalman_position_std_x_m"] ** 2
            + dropout_rows["kalman_position_std_y_m"] ** 2
        )

        max_dropout_error = round(float(dropout_rows["position_error_to_gps_m"].max()), 3)
        median_dropout_error = round(float(dropout_rows["position_error_to_gps_m"].median()), 3)
        max_dropout_speed = round(float(dropout_rows["dropout_kalman_speed_mps"].max()), 3)
        max_dropout_position_std = round(float(dropout_position_std.max()), 3)
    else:
        max_dropout_error = np.nan
        median_dropout_error = np.nan
        max_dropout_speed = np.nan
        max_dropout_position_std = np.nan

    summary = {
        "rows": len(dropout_df),
        "duration_s": round(float(dropout_df["t_s"].iloc[-1] - dropout_df["t_s"].iloc[0]), 3),
        "dropout_start_s": dropout_start_s,
        "dropout_end_s": dropout_end_s,
        "dropout_rows": len(dropout_rows),
        "median_gps_accuracy_m": round(float(dropout_df["gps_horizontal_accuracy_m"].median()), 3),
        "raw_gps_path_length_m": round(path_length(raw_df["x_m"], raw_df["y_m"]), 3),
        "dropout_kalman_path_length_m": round(
            float(dropout_df["dropout_kalman_cumulative_distance_m"].iloc[-1]), 3
        ),
        "max_position_error_to_gps_m": round(float(dropout_df["position_error_to_gps_m"].max()), 3),
        "max_dropout_position_error_to_gps_m": max_dropout_error,
        "median_dropout_position_error_to_gps_m": median_dropout_error,
        "max_dropout_kalman_speed_mps": max_dropout_speed,
        "max_position_std_m": round(float(position_std.max()), 3),
        "max_dropout_position_std_m": max_dropout_position_std,
        "final_position_error_to_gps_m": round(float(dropout_df["position_error_to_gps_m"].iloc[-1]), 3),
    }

    pd.DataFrame([summary]).to_csv(output_path, index=False)


def plot_trajectory(
    raw_df: pd.DataFrame,
    dropout_df: pd.DataFrame,
    output_path: Path,
    dropout_start_s: float,
    dropout_end_s: float,
) -> None:
    dropout_raw = raw_df[
        (raw_df["t_s"] >= dropout_start_s)
        & (raw_df["t_s"] <= dropout_end_s)
    ]

    plt.figure(figsize=(7, 7))

    plt.plot(raw_df["x_m"], raw_df["y_m"], marker="o", markersize=2, linewidth=1, label="raw GPS")
    plt.plot(
        dropout_df["dropout_kalman_x_m"],
        dropout_df["dropout_kalman_y_m"],
        linewidth=2,
        label="Kalman with GPS dropout",
    )

    if len(dropout_raw) > 0:
        plt.scatter(
            dropout_raw["x_m"],
            dropout_raw["y_m"],
            s=35,
            label="withheld GPS samples",
        )

    plt.scatter(raw_df["x_m"].iloc[0], raw_df["y_m"].iloc[0], s=80, label="start")
    plt.scatter(raw_df["x_m"].iloc[-1], raw_df["y_m"].iloc[-1], marker="x", s=80, label="end")

    plt.title("Kalman trajectory with simulated GPS dropout")
    plt.xlabel("East displacement from start (m)")
    plt.ylabel("North displacement from start (m)")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_speed(
    raw_df: pd.DataFrame,
    dropout_df: pd.DataFrame,
    output_path: Path,
    dropout_start_s: float,
    dropout_end_s: float,
) -> None:
    plt.figure(figsize=(10, 5))

    if "reported_speed_mps" in raw_df.columns:
        plt.plot(raw_df["t_s"], raw_df["reported_speed_mps"], label="reported GPS speed")

    plt.plot(raw_df["t_s"], raw_df["computed_speed_mps"], label="computed from GPS points", alpha=0.6)
    plt.plot(
        dropout_df["t_s"],
        dropout_df["dropout_kalman_speed_mps"],
        linewidth=2,
        label="Kalman speed with dropout",
    )

    plt.axvspan(dropout_start_s, dropout_end_s, alpha=0.2, label="GPS dropout window")

    plt.title("Speed behavior during simulated GPS dropout")
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_error(
    dropout_df: pd.DataFrame,
    output_path: Path,
    dropout_start_s: float,
    dropout_end_s: float,
) -> None:
    position_std = np.sqrt(
        dropout_df["kalman_position_std_x_m"] ** 2
        + dropout_df["kalman_position_std_y_m"] ** 2
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        dropout_df["t_s"],
        dropout_df["position_error_to_gps_m"],
        label="position difference from withheld/raw GPS",
    )
    plt.plot(dropout_df["t_s"], position_std, label="Kalman position uncertainty")
    plt.axvspan(dropout_start_s, dropout_end_s, alpha=0.2, label="GPS dropout window")

    plt.title("Position error and uncertainty during GPS dropout")
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
    parser.add_argument("--dropout-start", type=float, default=55.0)
    parser.add_argument("--dropout-end", type=float, default=70.0)
    parser.add_argument("--accel-std", type=float, default=0.8)
    parser.add_argument("--min-gps-std", type=float, default=3.0)
    args = parser.parse_args()

    input_path = Path(args.input)
    results_dir = Path("results")
    figures_dir = Path("figures")
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    raw_df = pd.read_csv(input_path)

    dropout_df = run_kalman_with_dropout(
        raw_df,
        dropout_start_s=args.dropout_start,
        dropout_end_s=args.dropout_end,
        accel_std=args.accel_std,
        min_gps_std=args.min_gps_std,
    )

    output_csv = results_dir / f"{args.label}_dropout_kalman.csv"
    summary_csv = results_dir / f"{args.label}_dropout_kalman_summary.csv"

    dropout_df.to_csv(output_csv, index=False)
    save_summary(raw_df, dropout_df, summary_csv, args.dropout_start, args.dropout_end)

    plot_trajectory(
        raw_df,
        dropout_df,
        figures_dir / f"{args.label}_dropout_trajectory.png",
        args.dropout_start,
        args.dropout_end,
    )
    plot_speed(
        raw_df,
        dropout_df,
        figures_dir / f"{args.label}_dropout_speed.png",
        args.dropout_start,
        args.dropout_end,
    )
    plot_error(
        dropout_df,
        figures_dir / f"{args.label}_dropout_error.png",
        args.dropout_start,
        args.dropout_end,
    )

    print(f"Saved dropout Kalman results: {output_csv}")
    print(f"Saved summary: {summary_csv}")
    print("Saved figures:")
    for path in sorted(figures_dir.glob(f"{args.label}_dropout_*.png")):
        print(f"  {path}")


if __name__ == "__main__":
    main()
