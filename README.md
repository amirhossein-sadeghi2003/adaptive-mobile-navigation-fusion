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

Initial project structure and plan.
