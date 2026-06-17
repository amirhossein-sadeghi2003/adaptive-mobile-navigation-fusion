from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def propagate(df, entry, drop_idx, s0, theta0, sign, use_imu):
    """Dead-reckon position across the dropout window.
    use_imu=False -> constant velocity, no turning (the prediction-only failure mode).
    use_imu=True  -> turn heading by the IMU yaw change (orientation_yaw delta) each step.
    Speed is held constant; only heading is aided by the IMU.
    """
    t = df["t_s"].values
    x = df["x_m"].values
    y = df["y_m"].values
    yaw = df["orientation_yaw"].values

    px, py, th, prev = x[entry], y[entry], theta0, entry
    xs, ys, errs = [], [], []
    for i in drop_idx:
        dt = t[i] - t[prev]
        if use_imu:
            th = th + sign * wrap(yaw[i] - yaw[prev])
        px += s0 * dt * np.cos(th)
        py += s0 * dt * np.sin(th)
        xs.append(px)
        ys.append(py)
        errs.append(float(np.hypot(px - x[i], py - y[i])))
        prev = i
    return np.array(xs), np.array(ys), np.array(errs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/gps_walk_02_fusion.csv")
    parser.add_argument("--label", default="gps_walk_02")
    parser.add_argument("--dropout-start", type=float, default=55.0)
    parser.add_argument("--dropout-end", type=float, default=70.0)
    parser.add_argument("--speed-window", type=float, default=5.0)
    # fixed frame relationship between device yaw and the East/North math angle,
    # confirmed once against GPS course; NOT tuned per walk
    parser.add_argument("--heading-sign", type=float, default=-1.0)
    args = parser.parse_args()

    df = pd.read_csv(args.input).sort_values("t_s").reset_index(drop=True)
    required = {"t_s", "x_m", "y_m", "gps_bearing", "gps_speed_mps", "orientation_yaw"}
    if not required.issubset(df.columns):
        raise ValueError(f"Input must contain {sorted(required)}")

    t = df["t_s"].values
    x = df["x_m"].values
    y = df["y_m"].values
    brg = df["gps_bearing"].values
    spd = df["gps_speed_mps"].values
    D0, D1 = args.dropout_start, args.dropout_end

    moving_before = np.where((t < D0) & (spd > 0.5))[0]
    if len(moving_before) == 0:
        raise ValueError("No moving GPS sample before dropout to anchor heading")
    entry = int(moving_before[-1])

    theta0 = np.deg2rad(90.0 - brg[entry])  # GPS course-over-ground -> math angle (CCW from East)
    pre = (t >= D0 - args.speed_window) & (t < D0)
    s0 = float(np.mean(spd[pre])) if pre.any() else float(spd[entry])

    drop_idx = np.where((t >= D0) & (t <= D1))[0]
    if len(drop_idx) == 0:
        raise ValueError("No samples inside the dropout window")

    cv_x, cv_y, cv_err = propagate(df, entry, drop_idx, s0, theta0, args.heading_sign, False)
    imu_x, imu_y, imu_err = propagate(df, entry, drop_idx, s0, theta0, args.heading_sign, True)

    out = pd.DataFrame({
        "t_s": t[drop_idx],
        "gps_x_m": x[drop_idx],
        "gps_y_m": y[drop_idx],
        "cv_x_m": cv_x,
        "cv_y_m": cv_y,
        "imu_x_m": imu_x,
        "imu_y_m": imu_y,
        "cv_error_m": cv_err,
        "imu_error_m": imu_err,
    })

    cv_max, cv_final = float(cv_err.max()), float(cv_err[-1])
    imu_max, imu_final = float(imu_err.max()), float(imu_err[-1])
    reduction = round(100.0 * (cv_max - imu_max) / cv_max, 1) if cv_max > 0 else float("nan")

    summary = {
        "dropout_start_s": D0,
        "dropout_end_s": D1,
        "dropout_samples": int(len(drop_idx)),
        "entry_t_s": round(float(t[entry]), 3),
        "entry_heading_deg": round(float(np.degrees(theta0)), 2),
        "assumed_speed_mps": round(s0, 3),
        "heading_sign": args.heading_sign,
        "cv_max_error_m": round(cv_max, 3),
        "cv_final_error_m": round(cv_final, 3),
        "imu_max_error_m": round(imu_max, 3),
        "imu_final_error_m": round(imu_final, 3),
        "max_error_reduction_pct": reduction,
    }

    results_dir = Path("results")
    figures_dir = Path("figures")
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    out_csv = results_dir / f"{args.label}_dropout_imu.csv"
    summary_csv = results_dir / f"{args.label}_dropout_imu_summary.csv"
    out.to_csv(out_csv, index=False)
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)

    drop_gps = (t >= D0) & (t <= D1)
    plt.figure(figsize=(7, 7))
    plt.plot(x, y, marker="o", markersize=2, linewidth=1, label="raw GPS")
    plt.plot(cv_x, cv_y, linewidth=2, label="dead reckon: no turning")
    plt.plot(imu_x, imu_y, linewidth=2, label="dead reckon: IMU heading")
    plt.scatter(x[drop_gps], y[drop_gps], s=35, label="withheld GPS")
    plt.scatter(x[entry], y[entry], s=80, label="dropout start")
    plt.title("Dead reckoning during GPS dropout: no-turn vs IMU heading")
    plt.xlabel("East displacement from start (m)")
    plt.ylabel("North displacement from start (m)")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / f"{args.label}_dropout_imu_trajectory.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(t[drop_idx], cv_err, marker="o", markersize=3, label="no-turn error")
    plt.plot(t[drop_idx], imu_err, marker="o", markersize=3, label="IMU-heading error")
    plt.title("Position error during GPS dropout")
    plt.xlabel("Time (s)")
    plt.ylabel("Error vs withheld GPS (m)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / f"{args.label}_dropout_imu_error.png", dpi=160)
    plt.close()

    print(f"entry t={t[entry]:.2f}s  heading={np.degrees(theta0):.1f} deg  speed={s0:.3f} m/s  samples={len(drop_idx)}")
    print(f"no-turn     : max {cv_max:6.2f} m | final {cv_final:6.2f} m")
    print(f"IMU heading : max {imu_max:6.2f} m | final {imu_final:6.2f} m  ({reduction}% lower max)")
    print("(project Kalman dropout baseline for reference: 25.929 m)")
    print(f"saved: {out_csv}")
    print(f"saved: {summary_csv}")
    print(f"saved: figures/{args.label}_dropout_imu_trajectory.png")
    print(f"saved: figures/{args.label}_dropout_imu_error.png")


if __name__ == "__main__":
    main()
