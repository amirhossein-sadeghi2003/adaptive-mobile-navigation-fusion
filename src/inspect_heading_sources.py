"""Inspect and compare the three heading sources in the fused dataset.

The fused table carries three independent heading-like signals:
- gps_bearing      : GPS course over ground (only meaningful while moving)
- orientation_yaw  : device orientation yaw from the IMU
- compass_bearing  : magnetometer-based magnetic bearing

They live in different references (true vs magnetic north, device frame) and may
use different units (radians vs degrees), so they will NOT match exactly. This
script reports each signal's raw range to expose its unit, converts everything to
degrees in [0, 360), and measures the offsets between sources on moving samples
only. Those offsets are the real thing to understand before using any heading to
aid the GPS filter.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def to_degrees_auto(series: pd.Series, name: str) -> pd.Series:
    """Return the signal in degrees, detecting radians vs degrees from its range.

    A radian heading spans roughly [-2*pi, 2*pi] (|value| <= ~6.3); a degree
    heading reaches ~360. The detection is printed so the assumption is visible
    and checkable, not hidden.
    """
    max_abs = float(series.abs().max())
    if max_abs <= 7.0:
        print(f"  {name:<16} looks like RADIANS (max |value| = {max_abs:.3f}) -> degrees")
        deg = np.rad2deg(series)
    else:
        print(f"  {name:<16} looks like DEGREES (max |value| = {max_abs:.3f})")
        deg = series
    return deg % 360.0


def angle_diff_deg(a: pd.Series, b: pd.Series) -> pd.Series:
    """Smallest signed difference a - b, wrapped to [-180, 180) degrees."""
    return (a - b + 180.0) % 360.0 - 180.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/gps_walk_02_fusion.csv")
    parser.add_argument("--label", default="gps_walk_02")
    parser.add_argument("--moving-speed-mps", type=float, default=0.5,
                        help="GPS bearing is only trusted above this speed")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    figures_dir = Path("figures")
    results_dir = Path("results")
    figures_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    print("Raw heading-signal ranges (before any conversion):")
    for col in ["gps_bearing", "orientation_yaw", "compass_bearing"]:
        s = df[col]
        print(f"  {col:<16} min={s.min():.3f}  max={s.max():.3f}  mean={s.mean():.3f}")

    print("Unit detection:")
    gps_deg = to_degrees_auto(df["gps_bearing"], "gps_bearing")
    yaw_deg = to_degrees_auto(df["orientation_yaw"], "orientation_yaw")
    comp_deg = to_degrees_auto(df["compass_bearing"], "compass_bearing")

    # GPS course over ground is noise when nearly stationary; compare moving samples only
    moving = df["gps_speed_mps"] > args.moving_speed_mps
    n_moving = int(moving.sum())
    print(f"Moving samples (gps_speed > {args.moving_speed_mps} m/s): {n_moving}/{len(df)}")

    d_comp_gps = angle_diff_deg(comp_deg[moving], gps_deg[moving])
    d_yaw_gps = angle_diff_deg(yaw_deg[moving], gps_deg[moving])
    d_comp_yaw = angle_diff_deg(comp_deg[moving], yaw_deg[moving])

    offsets = {
        "moving_samples": n_moving,
        "median_compass_minus_gps_deg": round(float(d_comp_gps.median()), 2),
        "median_yaw_minus_gps_deg": round(float(d_yaw_gps.median()), 2),
        "median_compass_minus_yaw_deg": round(float(d_comp_yaw.median()), 2),
        "std_compass_minus_gps_deg": round(float(d_comp_gps.std()), 2),
        "std_yaw_minus_gps_deg": round(float(d_yaw_gps.std()), 2),
    }
    print("Heading offsets on moving samples (degrees):")
    for k, v in offsets.items():
        print(f"  {k} = {v}")

    offsets_path = results_dir / f"{args.label}_heading_offsets.csv"
    pd.DataFrame([offsets]).to_csv(offsets_path, index=False)

    plt.figure(figsize=(11, 5))
    plt.plot(df["t_s"], gps_deg, ".", markersize=3, label="GPS bearing")
    plt.plot(df["t_s"], yaw_deg, linewidth=1, label="orientation yaw")
    plt.plot(df["t_s"], comp_deg, linewidth=1, alpha=0.8, label="compass bearing")
    plt.title("Heading sources over time (degrees, 0-360)")
    plt.xlabel("Time (s)")
    plt.ylabel("Heading (deg)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / f"{args.label}_heading_sources.png", dpi=160)
    plt.close()

    plt.figure(figsize=(11, 5))
    plt.plot(df["t_s"][moving], d_comp_gps, ".", markersize=3, label="compass - GPS")
    plt.plot(df["t_s"][moving], d_yaw_gps, ".", markersize=3, label="yaw - GPS")
    plt.axhline(0.0, color="k", linewidth=0.8)
    plt.title("Heading difference from GPS bearing (moving samples only)")
    plt.xlabel("Time (s)")
    plt.ylabel("Difference (deg)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / f"{args.label}_heading_differences.png", dpi=160)
    plt.close()

    print(f"\nSaved offsets: {offsets_path}")
    print("Saved figures:")
    for p in sorted(figures_dir.glob(f"{args.label}_heading_*.png")):
        print(f"  {p}")


if __name__ == "__main__":
    main()