from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from run_kalman_gps_baseline import make_process_noise, path_length


def run_reliability_weighted_kalman_with_gps_jump(
    df: pd.DataFrame,
    jump_start_s: float,
    jump_end_s: float,
    jump_x_m: float,
    jump_y_m: float,
    accel_std: float,
    min_gps_std: float,
    gate_sigma: float,
    min_gate_m: float,
    reliability_floor: float,
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
        base_R = np.diag([gps_std**2, gps_std**2])

        innovation = observed_z - (H @ x_pred)
        innovation_norm = float(np.linalg.norm(innovation))
        gate_threshold_m = max(min_gate_m, gate_sigma * gps_std)

        if innovation_norm <= gate_threshold_m:
            reliability_score = 1.0
        else:
            reliability_score = (gate_threshold_m / innovation_norm) ** 4
            reliability_score = max(float(reliability_score), reliability_floor)

        effective_R = base_R / reliability_score

        S = H @ P_pred @ H.T + effective_R
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
            "reliability_score": float(reliability_score),
            "effective_gps_std_m": float(gps_std / np.sqrt(reliability_score)),
            "reliability_kalman_x_m": float(x[0]),
            "reliability_kalman_y_m": float(x[1]),
            "reliability_kalman_vx_mps": float(x[2]),
            "reliability_kalman_vy_mps": float(x[3]),
            "reliability_kalman_speed_mps": float(np.sqrt(x[2] ** 2 + x[3] ** 2)),
            "gps_horizontal_accuracy_m": float(row["horizontalAccuracy"]),
            "innovation_norm_m": innovation_norm,
            "innovation_gate_m": float(gate_threshold_m),
            "residual_norm_m": float(np.linalg.norm(residual)),
            "position_error_to_clean_gps_m": position_error_to_clean_gps,
            "kalman_position_std_x_m": float(np.sqrt(max(P[0, 0], 0.0))),
            "kalman_position_std_y_m": float(np.sqrt(max(P[1, 1], 0.0))),
        })

    out = pd.DataFrame(result_rows)

    dx = out["reliability_kalman_x_m"].diff()
    dy = out["reliability_kalman_y_m"].diff()
    out["reliability_kalman_step_distance_m"] = np.sqrt(dx**2 + dy**2).fillna(0.0)
    out["reliability_kalman_cumulative_distance_m"] = out["reliability_kalman_step_distance_m"].cumsum()

    return out


def save_summary(
    raw_df: pd.DataFrame,
    ungated_df: pd.DataFrame,
    gated_df: pd.DataFrame,
    reliability_df: pd.DataFrame,
    output_path: Path,
    jump_start_s: float,
    jump_end_s: float,
    jump_x_m: float,
    jump_y_m: float,
    gate_sigma: float,
    min_gate_m: float,
    reliability_floor: float,
) -> None:
    ungated_jump = ungated_df[ungated_df["gps_jump_active"]]
    gated_jump = gated_df[gated_df["gps_jump_active"]]
    reliability_jump = reliability_df[reliability_df["gps_jump_active"]]

    summary = {
        "rows": len(reliability_df),
        "duration_s": round(float(reliability_df["t_s"].iloc[-1] - reliability_df["t_s"].iloc[0]), 3),
        "jump_start_s": jump_start_s,
        "jump_end_s": jump_end_s,
        "jump_rows": int(len(reliability_jump)),
        "jump_offset_x_m": jump_x_m,
        "jump_offset_y_m": jump_y_m,
        "jump_offset_norm_m": round(float(np.sqrt(jump_x_m**2 + jump_y_m**2)), 3),
        "gate_sigma": gate_sigma,
        "min_gate_m": min_gate_m,
        "reliability_floor": reliability_floor,
        "raw_gps_path_length_m": round(path_length(raw_df["x_m"], raw_df["y_m"]), 3),
        "ungated_path_length_m": round(float(ungated_df["jump_kalman_cumulative_distance_m"].iloc[-1]), 3),
        "gated_path_length_m": round(float(gated_df["gated_kalman_cumulative_distance_m"].iloc[-1]), 3),
        "reliability_weighted_path_length_m": round(float(reliability_df["reliability_kalman_cumulative_distance_m"].iloc[-1]), 3),
        "ungated_max_jump_error_m": round(float(ungated_jump["position_error_to_clean_gps_m"].max()), 3),
        "gated_max_jump_error_m": round(float(gated_jump["position_error_to_clean_gps_m"].max()), 3),
        "reliability_weighted_max_jump_error_m": round(float(reliability_jump["position_error_to_clean_gps_m"].max()), 3),
        "ungated_median_jump_error_m": round(float(ungated_jump["position_error_to_clean_gps_m"].median()), 3),
        "gated_median_jump_error_m": round(float(gated_jump["position_error_to_clean_gps_m"].median()), 3),
        "reliability_weighted_median_jump_error_m": round(float(reliability_jump["position_error_to_clean_gps_m"].median()), 3),
        "ungated_max_jump_speed_mps": round(float(ungated_jump["jump_kalman_speed_mps"].max()), 3),
        "gated_max_jump_speed_mps": round(float(gated_jump["gated_kalman_speed_mps"].max()), 3),
        "reliability_weighted_max_jump_speed_mps": round(float(reliability_jump["reliability_kalman_speed_mps"].max()), 3),
        "gated_rejected_updates_in_jump_window": int(gated_jump["gps_update_rejected"].sum()),
        "reliability_min_score_in_jump_window": round(float(reliability_jump["reliability_score"].min()), 3),
        "reliability_median_score_in_jump_window": round(float(reliability_jump["reliability_score"].median()), 3),
        "ungated_final_error_m": round(float(ungated_df["position_error_to_clean_gps_m"].iloc[-1]), 3),
        "gated_final_error_m": round(float(gated_df["position_error_to_clean_gps_m"].iloc[-1]), 3),
        "reliability_weighted_final_error_m": round(float(reliability_df["position_error_to_clean_gps_m"].iloc[-1]), 3),
    }

    pd.DataFrame([summary]).to_csv(output_path, index=False)


def plot_error_comparison(
    ungated_df: pd.DataFrame,
    gated_df: pd.DataFrame,
    reliability_df: pd.DataFrame,
    output_path: Path,
    jump_start_s: float,
    jump_end_s: float,
) -> None:
    plt.figure(figsize=(10, 5))

    plt.plot(ungated_df["t_s"], ungated_df["position_error_to_clean_gps_m"], label="ungated")
    plt.plot(gated_df["t_s"], gated_df["position_error_to_clean_gps_m"], label="hard gate")
    plt.plot(
        reliability_df["t_s"],
        reliability_df["position_error_to_clean_gps_m"],
        label="soft reliability weighting",
    )
    plt.axvspan(jump_start_s, jump_end_s, alpha=0.2, label="GPS jump window")

    plt.title("GPS jump error: ungated vs hard gate vs soft reliability")
    plt.xlabel("Time (s)")
    plt.ylabel("Position difference from clean GPS (m)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_reliability_scores(
    reliability_df: pd.DataFrame,
    output_path: Path,
    jump_start_s: float,
    jump_end_s: float,
) -> None:
    plt.figure(figsize=(10, 5))

    plt.plot(reliability_df["t_s"], reliability_df["reliability_score"], label="GPS reliability score")
    plt.axvspan(jump_start_s, jump_end_s, alpha=0.2, label="GPS jump window")

    plt.title("Soft GPS reliability score during simulated GPS jump")
    plt.xlabel("Time (s)")
    plt.ylabel("Reliability score")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_trajectory_comparison(
    raw_df: pd.DataFrame,
    ungated_df: pd.DataFrame,
    gated_df: pd.DataFrame,
    reliability_df: pd.DataFrame,
    output_path: Path,
) -> None:
    jump_rows = reliability_df[reliability_df["gps_jump_active"]]

    plt.figure(figsize=(7, 7))

    plt.plot(raw_df["x_m"], raw_df["y_m"], marker="o", markersize=2, linewidth=1, label="clean GPS")
    plt.plot(ungated_df["jump_kalman_x_m"], ungated_df["jump_kalman_y_m"], linewidth=2, label="ungated")
    plt.plot(gated_df["gated_kalman_x_m"], gated_df["gated_kalman_y_m"], linewidth=2, label="hard gate")
    plt.plot(
        reliability_df["reliability_kalman_x_m"],
        reliability_df["reliability_kalman_y_m"],
        linewidth=2,
        label="soft reliability",
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

    plt.title("GPS jump handling comparison")
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
    parser.add_argument("--input", default="results/gps_walk_02_gps_baseline.csv")
    parser.add_argument("--ungated", default="results/gps_walk_02_jump_kalman.csv")
    parser.add_argument("--gated", default="results/gps_walk_02_jump_gated_kalman.csv")
    parser.add_argument("--label", default="gps_walk_02")
    parser.add_argument("--jump-start", type=float, default=80.0)
    parser.add_argument("--jump-end", type=float, default=90.0)
    parser.add_argument("--jump-x", type=float, default=20.0)
    parser.add_argument("--jump-y", type=float, default=-10.0)
    parser.add_argument("--accel-std", type=float, default=0.8)
    parser.add_argument("--min-gps-std", type=float, default=3.0)
    parser.add_argument("--gate-sigma", type=float, default=4.0)
    parser.add_argument("--min-gate", type=float, default=8.0)
    parser.add_argument("--reliability-floor", type=float, default=0.05)
    args = parser.parse_args()

    raw_df = pd.read_csv(args.input)
    ungated_df = pd.read_csv(args.ungated)
    gated_df = pd.read_csv(args.gated)

    results_dir = Path("results")
    figures_dir = Path("figures")
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    reliability_df = run_reliability_weighted_kalman_with_gps_jump(
        raw_df,
        jump_start_s=args.jump_start,
        jump_end_s=args.jump_end,
        jump_x_m=args.jump_x,
        jump_y_m=args.jump_y,
        accel_std=args.accel_std,
        min_gps_std=args.min_gps_std,
        gate_sigma=args.gate_sigma,
        min_gate_m=args.min_gate,
        reliability_floor=args.reliability_floor,
    )

    output_csv = results_dir / f"{args.label}_jump_reliability_weighted_kalman.csv"
    summary_csv = results_dir / f"{args.label}_jump_reliability_comparison_summary.csv"

    reliability_df.to_csv(output_csv, index=False)
    save_summary(
        raw_df,
        ungated_df,
        gated_df,
        reliability_df,
        summary_csv,
        args.jump_start,
        args.jump_end,
        args.jump_x,
        args.jump_y,
        args.gate_sigma,
        args.min_gate,
        args.reliability_floor,
    )

    plot_error_comparison(
        ungated_df,
        gated_df,
        reliability_df,
        figures_dir / f"{args.label}_jump_reliability_error.png",
        args.jump_start,
        args.jump_end,
    )
    plot_reliability_scores(
        reliability_df,
        figures_dir / f"{args.label}_jump_reliability_score.png",
        args.jump_start,
        args.jump_end,
    )
    plot_trajectory_comparison(
        raw_df,
        ungated_df,
        gated_df,
        reliability_df,
        figures_dir / f"{args.label}_jump_reliability_trajectory.png",
    )

    print(f"Saved reliability-weighted GPS jump results: {output_csv}")
    print(f"Saved comparison summary: {summary_csv}")
    print("Saved figures:")
    for path in sorted(figures_dir.glob(f"{args.label}_jump_reliability_*.png")):
        print(f"  {path}")


if __name__ == "__main__":
    main()
