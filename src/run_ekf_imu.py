from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def wrap(a):
    """Wrap an angle (rad) to [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def run_ekf(df, D0, D1, heading_sign, min_gps_std, pos_std, speed_std, yaw_rate_std):
    """EKF with heading in the state, IMU yaw change as the turn input.

    State s = [x, y, v, theta]; theta is the math angle (CCW from East).
    Motion model (turn first with the measured yaw change, then step):
        theta_new = theta + sign * wrap(yaw[k] - yaw[prev])
        x_new = x + v * dt * cos(theta_new)
        y_new = y + v * dt * sin(theta_new)
        v_new = v
    The cos/sin of theta make the prediction nonlinear, so we linearise with
    the Jacobian F (that is the only reason this is an EKF and not the linear
    KF). GPS measures position only, so the update stays linear.
    During [D0, D1] the GPS update is skipped: the filter coasts on v and is
    steered only by the IMU turn, the same outage the dead-reckoning test used.
    """
    t = df["t_s"].values
    gx = df["x_m"].values
    gy = df["y_m"].values
    yaw = df["orientation_yaw"].values
    bearing = df["gps_bearing"].values
    gspeed = df["gps_speed_mps"].values
    hacc = df["horizontalAccuracy"].values

    # initial heading/speed from the first clearly moving GPS sample
    moving = np.where(gspeed > 0.5)[0]
    init_i = int(moving[0]) if len(moving) else 0
    theta0 = wrap(np.deg2rad(90.0 - bearing[init_i]))  # course-over-ground -> math angle
    v0 = float(gspeed[init_i])

    s = np.array([gx[0], gy[0], v0, theta0], dtype=float)
    P = np.diag([5.0**2, 5.0**2, 1.0**2, np.deg2rad(45.0) ** 2])

    H = np.array([[1.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0]])
    I4 = np.eye(4)

    rows = []
    prev_t = float(t[0])
    prev_yaw = float(yaw[0])

    for k in range(len(df)):
        tk = float(t[k])
        dt = max(tk - prev_t, 1e-3)
        dtheta = heading_sign * wrap(float(yaw[k]) - prev_yaw)
        prev_t = tk
        prev_yaw = float(yaw[k])

        x, y, v, th = s
        th_n = th + dtheta

        # --- predict ---
        s_pred = np.array([
            x + v * dt * np.cos(th_n),
            y + v * dt * np.sin(th_n),
            v,
            wrap(th_n),
        ])

        F = np.array([
            [1.0, 0.0, dt * np.cos(th_n), -v * dt * np.sin(th_n)],
            [0.0, 1.0, dt * np.sin(th_n),  v * dt * np.cos(th_n)],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        Q = np.diag([
            pos_std**2,
            pos_std**2,
            (speed_std * dt) ** 2,
            (yaw_rate_std * dt) ** 2,
        ])

        P_pred = F @ P @ F.T + Q

        in_dropout = (tk >= D0) and (tk <= D1)
        if in_dropout:
            # GPS withheld: prediction only
            s, P = s_pred, P_pred
        else:
            z = np.array([float(gx[k]), float(gy[k])])
            gps_std = max(float(hacc[k]), min_gps_std)
            R = np.diag([gps_std**2, gps_std**2])

            innov = z - (H @ s_pred)
            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T @ np.linalg.inv(S)
            s = s_pred + K @ innov
            s[3] = wrap(s[3])
            P = (I4 - K @ H) @ P_pred

        rows.append({
            "t_s": tk,
            "gps_x_m": float(gx[k]),
            "gps_y_m": float(gy[k]),
            "ekf_x_m": float(s[0]),
            "ekf_y_m": float(s[1]),
            "ekf_speed_mps": float(s[2]),
            "ekf_heading_deg": float(np.degrees(s[3])),
            "in_dropout": bool(in_dropout),
            "error_m": float(np.hypot(s[0] - gx[k], s[1] - gy[k])),
            "std_x_m": float(np.sqrt(max(P[0, 0], 0.0))),
            "std_y_m": float(np.sqrt(max(P[1, 1], 0.0))),
            "std_heading_deg": float(np.degrees(np.sqrt(max(P[3, 3], 0.0)))),
        })

    return pd.DataFrame(rows), float(np.degrees(theta0)), v0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="results/gps_walk_02_fusion.csv")
    p.add_argument("--label", default="gps_walk_02")
    p.add_argument("--dropout-start", type=float, default=55.0)
    p.add_argument("--dropout-end", type=float, default=70.0)
    # fixed device-yaw -> math-angle sign, confirmed once against GPS course; not tuned per walk
    p.add_argument("--heading-sign", type=float, default=-1.0)
    p.add_argument("--min-gps-std", type=float, default=3.0)
    p.add_argument("--pos-std", type=float, default=0.2)
    p.add_argument("--speed-std", type=float, default=0.5)        # m/s^2 on speed
    p.add_argument("--yaw-rate-std", type=float, default=0.1)     # rad/s on heading
    args = p.parse_args()

    df = pd.read_csv(args.input).sort_values("t_s").reset_index(drop=True)
    required = {"t_s", "x_m", "y_m", "horizontalAccuracy",
                "gps_bearing", "gps_speed_mps", "orientation_yaw"}
    if not required.issubset(df.columns):
        raise ValueError(f"Input must contain {sorted(required)}")

    D0, D1 = args.dropout_start, args.dropout_end
    out, theta0_deg, v0 = run_ekf(
        df, D0, D1, args.heading_sign, args.min_gps_std,
        args.pos_std, args.speed_std, args.yaw_rate_std,
    )

    drop = out[out["in_dropout"]]
    if drop.empty:
        raise ValueError("No samples inside the dropout window")
    ekf_max = float(drop["error_m"].max())
    ekf_final = float(drop["error_m"].iloc[-1])

    summary = {
        "dropout_start_s": D0,
        "dropout_end_s": D1,
        "dropout_samples": int(len(drop)),
        "init_heading_deg": round(theta0_deg, 2),
        "init_speed_mps": round(v0, 3),
        "heading_sign": args.heading_sign,
        "ekf_max_error_m": round(ekf_max, 3),
        "ekf_final_error_m": round(ekf_final, 3),
        "ekf_heading_std_end_deg": round(float(drop["std_heading_deg"].iloc[-1]), 2),
    }

    results_dir = Path("results")
    figures_dir = Path("figures")
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    out_csv = results_dir / f"{args.label}_dropout_ekf.csv"
    summary_csv = results_dir / f"{args.label}_dropout_ekf_summary.csv"
    out.to_csv(out_csv, index=False)
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)

    drop_gps = (out["t_s"] >= D0) & (out["t_s"] <= D1)
    plt.figure(figsize=(7, 7))
    plt.plot(out["gps_x_m"], out["gps_y_m"], marker="o", markersize=2, linewidth=1, label="raw GPS")
    plt.plot(out["ekf_x_m"], out["ekf_y_m"], linewidth=2, label="EKF estimate")
    plt.scatter(out.loc[drop_gps, "gps_x_m"], out.loc[drop_gps, "gps_y_m"], s=35, label="withheld GPS")
    plt.title("EKF with heading in state: trajectory (GPS withheld 55-70 s)")
    plt.xlabel("East displacement from start (m)")
    plt.ylabel("North displacement from start (m)")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / f"{args.label}_dropout_ekf_trajectory.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(drop["t_s"], drop["error_m"], marker="o", markersize=3, label="EKF error vs withheld GPS")
    plt.plot(drop["t_s"], drop["std_heading_deg"], alpha=0.7, label="heading std (deg)")
    plt.title("EKF position error and heading uncertainty during dropout")
    plt.xlabel("Time (s)")
    plt.ylabel("meters / degrees")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / f"{args.label}_dropout_ekf_error.png", dpi=160)
    plt.close()

    print(f"init heading={theta0_deg:.1f} deg  init speed={v0:.3f} m/s  dropout samples={len(drop)}")
    print(f"EKF        : max {ekf_max:6.2f} m | final {ekf_final:6.2f} m | heading std end {summary['ekf_heading_std_end_deg']:.1f} deg")
    print("(for reference, separate dead-reckoning test: no-turn 23.19 m, IMU-heading 17.23 m)")
    print(f"saved: {out_csv}")
    print(f"saved: {summary_csv}")
    print(f"saved: figures/{args.label}_dropout_ekf_trajectory.png")
    print(f"saved: figures/{args.label}_dropout_ekf_error.png")


if __name__ == "__main__":
    main()