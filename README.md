# Adaptive Mobile Navigation Fusion

Phone GPS is noisy enough that a straight walk does not always look straight on a plot.

This project uses real phone-collected GPS/IMU logs to build a small navigation-estimation pipeline. It starts from a GPS-only baseline and a simple 2D Kalman filter, then adds GPS fault tests and a first GPS/IMU step that uses the phone IMU to hold heading through a GPS outage.

## Current results

| Experiment | Duration | Samples | Main result |
|---|---:|---:|---|
| Room IMU sanity check | 35.6 s | ~3.5k per IMU stream | Phone accelerometer, gyroscope, orientation, compass, and total acceleration logs were readable at about 100 Hz. |
| Short GPS walk | 34.2 s | 68 GPS samples | Short paths are strongly affected by early GPS error. |
| Longer GPS walk | 122.8 s | 283 GPS samples | The out-and-back walking path is visible and usable for a first baseline. |
| GPS-only baseline | 122.8 s | 283 GPS samples | Point-to-point GPS speed has unrealistic spikes up to about 8.3 m/s. |
| Kalman baseline | 122.8 s | 283 GPS samples | Kalman speed stays more realistic, with max speed around 2.25 m/s and median speed around 1.09 m/s. |
| GPS dropout simulation | 122.8 s | 283 GPS samples | During a simulated 55-70 s GPS outage, the prediction-only Kalman estimate drifts up to about 25.9 m from GPS before recovering after GPS updates return. |
| GPS jump simulation | 122.8 s | 283 GPS samples | A simulated 22.4 m GPS position jump pulls the simple Kalman estimate away from the clean path, with position error reaching about 26.5 m. |
| GPS jump with innovation gating | 122.8 s | 283 GPS samples | A simple innovation gate rejects all 25 jumped GPS updates and reduces max jump-window error from about 26.5 m to about 3.8 m. |
| IMU-aided dropout dead reckoning | 122.8 s | 283 GPS samples | Over a simulated 55-70 s outage (28 withheld samples), adding IMU heading change to a constant-speed dead reckoning cut the max drift from about 23.2 m (no turning) to about 17.2 m, roughly 25.7% lower. |
| EKF with heading in state | 122.8 s | 283 GPS samples | Putting heading in the EKF state and driving it with the IMU turn rate, on the same 55-70 s outage, brought max dropout error down to about 10.9 m, below the 17.2 m dead-reckoning result. |

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

## GPS dropout simulation

To test navigation robustness, I simulated a GPS outage from 55 s to 70 s on the longer walk. During that window, the Kalman filter keeps predicting motion but does not use GPS position updates.

The result is expected: uncertainty grows during the outage, and the estimate drifts away from the withheld GPS samples. In this run, the largest position difference during dropout was about 25.9 m. Once GPS updates return, the filter quickly moves back toward the measured path.

![GPS dropout trajectory](figures/gps_walk_02_dropout_trajectory.png)

The uncertainty plot makes the failure mode easier to see than the trajectory plot alone.

![GPS dropout error](figures/gps_walk_02_dropout_error.png)

## GPS jump simulation

Dropout is one failure mode. A different problem is bad GPS that still looks like a valid measurement. To test that case, I injected a 20 m east and -10 m north offset from 80 s to 90 s.

The simple Kalman filter follows the corrupted GPS instead of rejecting it. In this run, a 22.4 m injected GPS jump caused the Kalman position error to reach about 26.5 m relative to the clean GPS path. When the GPS returns to the clean path, the filter recovers, but the speed estimate briefly spikes.

![GPS jump trajectory](figures/gps_walk_02_jump_trajectory.png)

The error plot shows why a plain Kalman filter is not enough for GPS fault handling. A next step would be innovation gating or a GPS reliability score before accepting position updates.

![GPS jump error](figures/gps_walk_02_jump_error.png)

## GPS jump with innovation gating

The previous jump test showed a weakness: a plain Kalman filter trusts the corrupted GPS measurements. I added a simple innovation gate before the update step. If the GPS innovation is too large, the filter skips that measurement and keeps predicting.

With a 4-sigma gate and a minimum gate of 8 m, the filter rejected all 25 GPS updates during the injected jump window. The maximum jump-window position error dropped from about 26.5 m without gating to about 3.8 m with gating.

![GPS jump gating error](figures/gps_walk_02_jump_gating_error.png)

The gate decision plot shows the rejected updates during the artificial GPS jump.

![GPS jump gating decisions](figures/gps_walk_02_jump_gating_decisions.png)

## IMU-heading dead reckoning during GPS dropout

The earlier dropout test let the Kalman filter coast on its last velocity, and the estimate drifted about 25.9 m before GPS returned. That version has no way to know the person turned during the outage. This experiment asks a narrower question: if I hold the speed fixed but let the phone IMU tell me how the heading changed, how much of that drift goes away?

I rebuilt the outage as a dead-reckoning problem on the synced GPS+IMU table from milestone 1. At the last moving GPS sample before 55 s, I take the GPS course-over-ground as the starting heading and the mean speed over the previous 5 s (0.856 m/s) as a fixed speed. Across the 55-70 s window (28 withheld samples) I propagate position two ways, using the same speed and entry heading so the only difference is the turning:

- no turning: heading frozen at the entry value
- IMU heading: heading turned at each step by the change in `orientation_yaw`, not its absolute value

Using the change and not the absolute yaw comes straight from the milestone 2 heading check. GPS bearing, orientation yaw, and compass bearing each had a standard deviation near 30 deg on this walk, so no single one is a trustworthy absolute reference. The turn between two nearby samples is far more reliable than any one heading reading.

![Dead reckoning during dropout: no-turn vs IMU heading](figures/gps_walk_02_dropout_imu_trajectory.png)

The no-turn baseline reached a max error of 23.19 m over the window. Adding the IMU heading brought it down to 17.23 m, about 25.7% lower. For both methods the error is largest at the very end of the window, which is what dead reckoning should do: drift keeps accumulating the longer GPS is gone.

![Position error during dropout: no-turn vs IMU heading](figures/gps_walk_02_dropout_imu_error.png)

One sign detail is worth recording. `heading_sign` is fixed at -1 because the device yaw increases clockwise while the East/North math angle increases counter-clockwise. I checked this once against the GPS course direction and then left it fixed. It is a frame relationship, not a value I tuned to make this walk look good.

This is still far from solved. 17 m is a large error to accept, and holding speed constant ignores that real walking speed changes across those 15 seconds. The next step is to feed the IMU turn rate into the filter as a control input instead of correcting heading after the fact, and to update speed from the accelerometer instead of holding it fixed.

## EKF with heading in the state

The dead-reckoning test corrected heading after the fact: I picked one entry bearing, turned it with the IMU, and only then propagated position. This experiment folds heading into the filter itself. The state becomes `[x, y, v, theta]`, the IMU yaw change drives `theta` as a turn input at each step, and the position update stays the same GPS-only update as before. Because the prediction now multiplies speed by the sine and cosine of `theta`, the model is nonlinear, so the filter linearises it with a Jacobian. That is the only reason this is an EKF and not the earlier linear one.

The part worth pausing on is how heading gets corrected at all. GPS measures position, never heading, and the measurement matrix only touches `x` and `y`. But the Jacobian couples speed and heading into the predicted position, so the covariance builds up a correlation between position and heading. When a fix arrives and the position residual is nonzero, the Kalman gain pushes part of that correction into `theta` and `v` too. A position-only sensor ends up cleaning the heading it never directly sees, which is exactly what the dead-reckoning version had no way to do.

Run on the same 55-70 s outage (28 withheld samples), the EKF reached a maximum dropout error of about 10.9 m, against 17.2 m for the IMU dead reckoning and 23.2 m for the no-turn baseline.

![EKF trajectory with GPS withheld 55-70 s](figures/gps_walk_02_dropout_ekf_trajectory.png)

The heading uncertainty tells the honest version of the story. Outside the outage the filter holds its heading standard deviation near 13 deg. Across the 15 s with no GPS it inflates to about 22.6 deg, then relaxes back toward 13 deg once fixes return. The filter is aware it is coasting blind, and its own covariance shows it.

![EKF error and heading uncertainty during dropout](figures/gps_walk_02_dropout_ekf_error.png)

I want to be careful about the 10.9 vs 17.2 comparison, because it is not a clean single-variable test. The EKF enters the outage carrying a speed and heading that GPS has been correcting right up to that moment, while the dead-reckoning run froze a windowed-mean speed and a single entry bearing. So part of the gap comes from the better entry state, not only from the filter structure. The mechanism above is real, but the headline number flatters it. One smaller detail from the code: the initial heading and speed are seeded from the first GPS sample moving faster than 0.5 m/s, not from sample zero, because a near-stationary first fix has a meaningless course-over-ground.

Even taken at face value, 11 m of peak error over a 15 s outage is still large, and all of this rests on a single walk. The next improvement is to stop holding speed fixed through the outage and correct it from the accelerometer, the same way heading is now carried in the filter instead of frozen.

## Generated outputs

Main result files:

- `results/gps_walk_02_gps_baseline.csv`
- `results/gps_walk_02_gps_baseline_summary.csv`
- `results/gps_walk_02_kalman_baseline.csv`
- `results/gps_walk_02_kalman_baseline_summary.csv`
- `results/gps_walk_02_dropout_kalman.csv`
- `results/gps_walk_02_dropout_kalman_summary.csv`
- `results/gps_walk_02_jump_kalman.csv`
- `results/gps_walk_02_jump_kalman_summary.csv`
- `results/gps_walk_02_jump_gated_kalman.csv`
- `results/gps_walk_02_jump_gating_comparison_summary.csv`
- `results/gps_walk_02_dropout_imu.csv`
- `results/gps_walk_02_dropout_imu_summary.csv`
- `results/gps_walk_02_dropout_ekf.csv`
- `results/gps_walk_02_dropout_ekf_summary.csv`

Main scripts:

- `src/plot_sensor_log.py`
- `src/plot_gps_walk.py`
- `src/build_gps_baseline.py`
- `src/run_kalman_gps_baseline.py`
- `src/simulate_gps_dropout.py`
- `src/simulate_gps_jump.py`
- `src/compare_gps_jump_gating.py`
- `src/build_gps_imu_dataset.py`
- `src/inspect_heading_sources.py`
- `src/simulate_gps_dropout_imu.py`
- `src/run_ekf_imu.py`

## Planned direction

Next steps:

- update speed from the accelerometer during dropout instead of holding it constant
- collect several synced walks so an adaptive or learned reliability model has enough data to be more than a fit to one walk
- test a softer GPS reliability score instead of a hard innovation gate

## Limitations

- phone GPS is not ground truth
- phone IMU orientation is not fixed to a robot body frame
- the current Kalman filter uses GPS positions only
- the current test path is simple and mostly straight
- the dropout experiment is simulated from logged GPS data, not a live sensor failure
- the GPS jump experiment uses an injected offset, not a real spoofing device
- the current innovation gate is a simple fixed-threshold rule, not an adaptive reliability model
- the IMU dead-reckoning experiment holds speed constant and only corrects heading, so it cannot follow real speed changes during the outage
- a 15 s outage is long for dead reckoning, and 17 m is still a large absolute error
- the EKF carries heading in its state but uses the IMU only as a turn input and the accelerometer not at all, so speed is corrected only when GPS returns
- the EKF peak error is about 11 m over the same 15 s outage, still large, and rests on the same single walk
- future tests should include turns, stops, and live controlled GPS dropout