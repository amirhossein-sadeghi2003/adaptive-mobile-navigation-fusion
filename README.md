# Adaptive Mobile Navigation Fusion

This project explores mobile navigation using real phone-collected GPS/IMU data.

The goal is to combine classical state estimation with learning-based sensor reliability estimation. The first version will start with a simple real trajectory, GPS-only analysis, and Kalman-based filtering. Later versions will test GPS dropout, noisy segments, and an ML-assisted adaptive fusion pipeline.

## Planned direction

- collect GPS/IMU data from a phone
- clean and convert trajectory data
- build a GPS-only baseline
- implement Kalman-based state estimation
- simulate GPS dropout and noisy GPS jumps
- train a lightweight ML model to estimate GPS reliability
- compare standard filtering with adaptive fusion
- document results with plots, metrics, and limitations

## Why this project

This repository is intended to connect real-world sensing, navigation, estimation, and AI-assisted robustness. It is not a pure simulation project; the main dataset will come from real phone movement logs.

## Current status

The project now has an initial real phone-IMU sanity check.

## First sensor sanity check

The first controlled phone-IMU recording has been collected and inspected.

The log is a short indoor room test, about 35.6 seconds long, with accelerometer, gyroscope, orientation, magnetometer, compass, and total acceleration streams sampled at about 100 Hz. The raw phone CSV files are kept out of the repository, but the generated summary and plots are included.

Generated outputs:

- `results/room_test_01_sensor_summary.csv`
- `figures/room_test_01_accelerometer.png`
- `figures/room_test_01_gyroscope.png`
- `figures/room_test_01_orientation.png`
- `figures/room_test_01_compass.png`
- `figures/room_test_01_total_acceleration_norm.png`

This first test does not estimate a trajectory yet. It only checks that the phone sensor logs are readable, time-stamped consistently, and active enough to support the next navigation experiments.
