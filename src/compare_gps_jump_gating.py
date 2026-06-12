from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from run_kalman_gps_baseline import make_process_noise, path_length


def run_gated_kalman_with_gps_jump(
    df: pd.DataFrame,
    jump_start_s: float,
    jump_end_s: float,
    jump_x_m: float,
    jump_y_m: float,
    accel_std: float,
    min_gps_std: float,
    gate_sigma: float,
    min_gate_m: float,
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
        innovation_norm = float(np.linalg.norm(innovation))
        gate_threshold_m = max(min_gate_m, gate_sigma * gps_std)

        gps_update_used = innovation_norm <= gate_threshold_m

        if gps_update_used:
            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T @ np.linalg.inv(S)

            x = x_pred + K @ innovation
            P = (I - K @ H) @ P_pred

            residual = observed_z - (H @ x)
            residual_norm = float(np.linalg.norm(residual))
        else:
            x = x_pred
            P = P_pred
            residual_norm = np.nan

        position_error_to_clean_gps = float(np.linalg.norm(clean_z - (H @ x)))
        observed_gps_error_to_clean = float(np.linalg.norm(observed_z - clean_z))

        result_rows.append({
            "t_s": t,
            "clean_gps_x_m": float(clean_z[0]),
            "clean_gps_y_m": float(clean_z[1]),
            "observed_gps_x_m": float(observed_z[0]),
            "observed_gps_y_m": float(observed_z[1]),
            "gps_jump_active": bool(jump_active),
            "gps_update_used": bool(gps_update_used),
            "gps_update_rejected": bool(not gps_update_used),
            "observed_gps_error_to_clean_m": observed_gps_error_to_clean,
            "gated_kalman_x_m": float(x[0]),
            "gated_kalman_y_m": float(x[1]),
            "gated_kalman_vx_mps": float(x[2]),
            "gated_kalman_vy_mps": float(x[3]),
            "gated_kalman_speed_mps": float(np.sqrt(x[2] ** 2 + x[3] ** 2)),
            "gps_horizontal_accuracy_m": float(row["horizontalAccuracy"]),
            "innovation_norm_m": innovation_norm,
            "innovation_gate_m": float(gate_threshold_m),
            "residual_norm_m": residual_norm,
            "position_error_to_clean_gps_m": position_error_to_clean_gps,
            "kalman_position_std_x_m": float(np.sqrt(max(P[0, 0], 0.0))),
            "kalman_position_std_y_m": float(np.sqrt(max(P[1, 1], 0.0))),
        })

    out = pd.DataFrame(result_rows)

    dx = out["gated_kalman_x_m"].diff()
    dy = out["gated_kalman_y_m"].diff()
    out["gated_kalman_step_distance_m"] = np.sqrt(dx**2 + dy**2).fillna(0.0)
    out["gated_kalman_cumulative_distance_m"] = out["gated_kalman_step_distance_m"].cumsum()

    return out


def save_comparison_summary(
    raw_df: pd.DataFrame,
    ungated_df: pd.DataFrame,
    gated_df: pd.DataFrame,
    output_path: Path,
    jump_start_s: float,
    jump_end_s: float,
    jump_x_m: float,
    jump_y_m: float,
    gate_sigma: float,
    min_gate_m: float,
) -> None:
    ungated_jump = ungated_df[ungated_df["gps_jump_active"]]
    gated_jump = gated_df[gated_df["gps_jump_active"]]

    summary = {
        "rows": len(gated_df),
        "duration_s": round(float(gated_df["t_s"].iloc[-1] - gated_df["t_s"].iloc[0]), 3),
        "jump_start_s": jump_start_s,
        "jump_end_s": jump_end_s,
        "jump_rows": int(len(gated_jump)),
        "jump_offset_x_m": jump_x_m,
        "jump_offset_y_m": jump_y_m,
        "jump_offset_norm_m": round(float(np.sqrt(jump_x_m**2 + jump_y_m**2)), 3),
        "gate_sigma": gate_sigma,
        "min_gate_m": min_gate_m,
        "raw_gps_path_length_m": round(path_length(raw_df["x_m"], raw_df["y_m"]), 3),
        "ungated_jump_kalman_path_length_m": round(float(ungated_df["jump_kalman_cumulative_distance_m"].iloc[-1]), 3),
        "gated_jump_kalman_path_length_m": round(float(gated_df["gated_kalman_cumulative_distance_m"].iloc[-1]), 3),
        "ungated_max_jump_window_error_m": round(float(ungated_jump["position_error_to_clean_gps_m"].max()), 3),
        "gated_max_jump_window_error_m": round(float(gated_jump["position_error_to_clean_gps_m"].max()), 3),
        "ungated_median_jump_window_error_m": round(float(ungated_jump["position_error_to_clean_gps_m"].median()), 3),
        "gated_median_jump_window_error_m": round(float(gated_jump["position_error_to_clean_gps_m"].median()), 3),
        "ungated_max_jump_window_speed_mps": round(float(ungated_jump["jump_kalman_speed_mps"].max()), 3),
        "gated_max_jump_window_speed_mps": round(float(gated_jump["gated_kalman_speed_mps"].max()), 3),
        "gated_rejected_updates_total": int(gated_df["gps_update_rejected"].sum()),
        "gated_rejected_updates_in_jump_window": int(gated_jump["gps_update_rejected"].sum()),
        "ungated_final_error_m": round(float(ungated_df["position_error_to_clean_gps_m"].iloc[-1]), 3),
        "gated_final_error_m": round(float(gated_df["position_error_to_clean_gps_m"].iloc[-1]), 3),
    }

    pd.DataFrame([summary]).to_csv(output_path, index=False)


def plot_trajectory_comparison(
    raw_df: pd.DataFrame,
    ungated_df: pd.DataFrame,
    gated_df: pd.DataFrame,
    output_path: Path,
) -> None:
    jump_rows = gated_df[gated_df["gps_jump_active"]]

    plt.figure(figsize=(7, 7))

    plt.plot(raw_df["x_m"], raw_df["y_m"], marker="o", markersize=2, linewidth=1, label="clean GPS")
    plt.plot(
        ungated_df["jump_kalman_x_m"],
        ungated_df["jump_kalman_y_m"],
        linewidth=2,
        label="ungated Kalman",
    )
    plt.plot(
        gated_df["gated_kalman_x_m"],
        gated_df["gated_kalman_y_m"],
        linewidth=2,
        label="gated Kalman",
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

    plt.title("GPS jump: ungated vs gated Kalman update")
    plt.xlabel("East displacement from start (m)")
    plt.ylabel("North displacement from start (m)")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_error_comparison(
    ungated_df: pd.DataFrame,
    gated_df: pd.DataFrame,
    output_path: Path,
    jump_start_s: float,
    jump_end_s: float,
) -> None:
    plt.figure(figsize=(10, 5))

    plt.plot(
        ungated_df["t_s"],
        ungated_df["position_error_to_clean_gps_m"],
        label="ungated Kalman error",
    )
    plt.plot(
        gated_df["t_s"],
        gated_df["position_error_to_clean_gps_m"],
        label="gated Kalman error",
    )
    plt.axvspan(jump_start_s, jump_end_s, alpha=0.2, label="GPS jump window")

    plt.title("GPS jump position error with innovation gating")
    plt.xlabel("Time (s)")
    plt.ylabel("Position difference from clean GPS (m)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_gate_decisions(
    gated_df: pd.DataFrame,
    output_path: Path,
    jump_start_s: float,
    jump_end_s: float,
) -> None:
    rejected = gated_df[gated_df["gps_update_rejected"]]

    plt.figure(figsize=(10, 5))

    plt.plot(gated_df["t_s"], gated_df["innovation_norm_m"], label="innovation norm")
    plt.plot(gated_df["t_s"], gated_df["innovation_gate_m"], label="innovation gate")
    plt.scatter(
        rejected["t_s"],
        rejected["innovation_norm_m"],
        s=30,
        label="rejected update",
    )
    plt.axvspan(jump_start_s, jump_end_s, alpha=0.2, label="GPS jump window")

    plt.title("Innovation gating decisions during GPS jump")
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
    parser.add_argument("--ungated", default="results/gps_walk_02_jump_kalman.csv")
    parser.add_argument("--label", default="gps_walk_02")
    parser.add_argument("--jump-start", type=float, default=80.0)
    parser.add_argument("--jump-end", type=float, default=90.0)
    parser.add_argument("--jump-x", type=float, default=20.0)
    parser.add_argument("--jump-y", type=float, default=-10.0)
    parser.add_argument("--accel-std", type=float, default=0.8)
    parser.add_argument("--min-gps-std", type=float, default=3.0)
    parser.add_argument("--gate-sigma", type=float, default=4.0)
    parser.add_argument("--min-gate", type=float, default=8.0)
    args = parser.parse_args()

    input_path = Path(args.input)
    ungated_path = Path(args.ungated)

    results_dir = Path("results")
    figures_dir = Path("figures")
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    raw_df = pd.read_csv(input_path)
    ungated_df = pd.read_csv(ungated_path)

    gated_df = run_gated_kalman_with_gps_jump(
        raw_df,
        jump_start_s=args.jump_start,
        jump_end_s=args.jump_end,
        jump_x_m=args.jump_x,
        jump_y_m=args.jump_y,
        accel_std=args.accel_std,
        min_gps_std=args.min_gps_std,
        gate_sigma=args.gate_sigma,
        min_gate_m=args.min_gate,
    )

    output_csv = results_dir / f"{args.label}_jump_gated_kalman.csv"
    summary_csv = results_dir / f"{args.label}_jump_gating_comparison_summary.csv"

    gated_df.to_csv(output_csv, index=False)
    save_comparison_summary(
        raw_df,
        ungated_df,
        gated_df,
        summary_csv,
        args.jump_start,
        args.jump_end,
        args.jump_x,
        args.jump_y,
        args.gate_sigma,
        args.min_gate,
    )

    plot_trajectory_comparison(
        raw_df,
        ungated_df,
        gated_df,
        figures_dir / f"{args.label}_jump_gating_trajectory.png",
    )
    plot_error_comparison(
        ungated_df,
        gated_df,
        figures_dir / f"{args.label}_jump_gating_error.png",
        args.jump_start,
        args.jump_end,
    )
    plot_gate_decisions(
        gated_df,
        figures_dir / f"{args.label}_jump_gating_decisions.png",
        args.jump_start,
        args.jump_end,
    )

    print(f"Saved gated GPS jump results: {output_csv}")
    print(f"Saved comparison summary: {summary_csv}")
    print("Saved figures:")
    for path in sorted(figures_dir.glob(f"{args.label}_jump_gating_*.png")):
        print(f"  {path}")


if __name__ == "__main__":
    main()
