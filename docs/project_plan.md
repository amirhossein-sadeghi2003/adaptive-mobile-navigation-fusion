# Project Plan: Adaptive Mobile Navigation Fusion

## Core idea

Use real GPS/IMU data collected from a phone to study mobile navigation under noisy sensing conditions.

The project combines:

- real movement data
- GPS/IMU preprocessing
- state estimation
- Kalman filtering
- GPS dropout and jump experiments
- machine-learning-based GPS reliability estimation
- adaptive sensor fusion

## Motivation

The project is aligned with autonomous and networked cyber-physical systems. The focus is not generic AI or generic mobile data analysis. The focus is navigation: estimating the state of a moving platform using imperfect sensors.

A phone is used as a low-cost mobile sensing platform. It is not a perfect rover, but it provides real GPS and inertial data that can be logged, inspected, filtered, and evaluated.

## Phase 1 — Data collection and inspection

Goal:

Collect one or more real trajectories using a phone sensor logger.

Expected files:

- raw GPS/IMU CSV logs
- a short note describing where/how the data was collected
- basic trajectory plot

Questions to answer:

- Is GPS frequency stable?
- Are there missing samples?
- Are there visible GPS jumps?
- Does the IMU data look usable?

## Phase 2 — GPS-only baseline

Goal:

Build a baseline from GPS positions only.

Expected outputs:

- converted local x/y coordinates
- trajectory plot
- speed estimate from GPS
- basic data quality summary

## Phase 3 — Kalman filter baseline

Goal:

Implement a simple state estimator for 2D motion.

Possible state:

- x position
- y position
- x velocity
- y velocity

Expected outputs:

- estimated trajectory
- comparison against GPS
- behavior during short GPS dropout periods

## Phase 4 — Learning-based GPS reliability

Goal:

Train a lightweight ML model to detect unreliable GPS segments.

Possible features:

- GPS speed
- GPS jump distance
- acceleration magnitude
- gyroscope magnitude
- time gap
- heading change
- recent residual/error

Possible labels:

- reliable GPS
- unreliable GPS / jump / dropout

Labels may start from rule-based heuristics, then be improved manually if needed.

## Phase 5 — Adaptive fusion

Goal:

Use GPS reliability estimates to adjust how much the filter trusts GPS.

Expected comparison:

- GPS-only baseline
- standard Kalman filter
- adaptive Kalman filter with ML-assisted GPS reliability

## Evaluation

Possible metrics:

- error on masked GPS points
- trajectory smoothness
- recovery after GPS dropout
- effect of artificial GPS jumps
- qualitative trajectory comparison

## Limitations

- phone GPS is not high-precision ground truth
- phone IMU can drift and may not be aligned with a rover body frame
- artificial dropout/jumps are controlled experiments, not full real-world failure coverage
- this is a navigation-estimation study, not a complete autonomous robot stack

## First milestone

Create the project structure, collect the first real phone trajectory, and produce the first trajectory plot.
