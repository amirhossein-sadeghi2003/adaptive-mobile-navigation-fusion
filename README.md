# Adaptive Mobile Navigation Fusion

Phone GPS is noisy enough that a straight walk does not always look straight on a plot.

This project uses real phone-collected GPS/IMU logs to build a small navigation-estimation pipeline. The first milestone is a GPS-only baseline, followed by a simple Kalman filter for 2D motion.

## Current results

| Experiment | Duration | Samples | Main result |
|---|---:|---:|---|
| Room IMU sanity check | 35.6 s | ~3.5k per IMU stream | Phone accelerometer, gyroscope, orientation, compass, and total acceleration logs were readable at about 100 Hz. |
| Short GPS walk | 34.2 s | 68 GPS samples | Short paths are strongly affected by early GPS error. |
| Longer GPS walk | 122.8 s | 283 GPS samples | The out-and-back walking path is visible and usable for a first baseline. |
| GPS-only baseline | 122.8 s | 283 GPS samples | Point-to-point GPS speed has unrealistic spikes up to about 8.3 m/s. |
| Kalman baseline | 122.8 s | 283 GPS samples | Kalman speed stays more realistic, with max speed around 2.25 m/s and median speed around 1.09 m/s. |

## Longer GPS walk

The second outdoor walk is the first useful navigation log in this repository. The phone GPS had a median horizontal accuracy of about 3.0 m.

![Longer GPS walk trajectory](figures/gps_walk_02_local_trajectory.png)

The filtered view removes the worst low-accuracy GPS fixes.

![Filtered GPS trajectory](figures/gps_walk_02_filtered_trajectory.png)

## GPS-only baseline

The public baseline file does not store raw latitude or longitude. It keeps local east/north coordinates relative to the starting point, plus distance, speed, and GPS accuracy fields.

The GPS-only baseline shows the main problem clearly: computing speed directly from consecutive GPS points creates sharp spikes.

![GPS-only speed baseline](figures/gps_walk_02_gps_baseline_speed_compare.png)

## Kalman baseline

A simple constant-velocity Kalman filter was added with state:

`[x, y, vx, vy]`

The filter does not solve the full navigation problem yet, but it reduces the worst GPS-derived speed spikes. In this run, the maximum raw computed GPS speed was about 8.3 m/s, while the maximum Kalman speed was about 2.25 m/s.

![Kalman speed comparison](figures/gps_walk_02_kalman_speed_compare.png)

The estimated trajectory still follows the same general out-and-back path.

![Kalman trajectory](figures/gps_walk_02_kalman_trajectory.png)

## Generated outputs

Main result files:

- `results/gps_walk_02_gps_baseline.csv`
- `results/gps_walk_02_gps_baseline_summary.csv`
- `results/gps_walk_02_kalman_baseline.csv`
- `results/gps_walk_02_kalman_baseline_summary.csv`

Main scripts:

- `src/plot_sensor_log.py`
- `src/plot_gps_walk.py`
- `src/build_gps_baseline.py`
- `src/run_kalman_gps_baseline.py`

## Planned direction

Next steps:

- compare GPS-only smoothing against Kalman filtering
- simulate GPS dropout on the longer walk
- test artificial GPS jumps
- use IMU-derived features for sensor reliability checks
- build an adaptive fusion experiment

## Limitations

- phone GPS is not ground truth
- phone IMU orientation is not fixed to a robot body frame
- the current Kalman filter uses GPS positions only
- the current test path is simple and mostly straight
- future tests should include turns, stops, and controlled GPS dropout
