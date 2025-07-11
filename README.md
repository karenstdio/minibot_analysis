
# MiniBot Data Analysis

This project analyzes MiniBot log data collected over time. It includes various metrics such as RPM values, battery voltage, linear speed, and distance. The dataset allows for robot efficiency evaluation, anomaly detection, and correlation analysis.

## Dataset

- **File**: `minibot_logs.csv`
- **Columns**:
  - `timestamp`: Timestamp of the log entry
  - `robot_id`: Unique identifier of the robot
  - `left_rpm`, `right_rpm`: Wheel rotation speed in RPM
  - `battery_v`: Battery voltage in volts
  - `distance_cm`: Total distance traveled in centimeters
  - `imu_heading_deg`: IMU heading in degrees

## Analysis Performed

- **Linear Speed Calculation** using average RPM and wheel radius
- **Anomaly Detection** based on battery voltage deviations
- **Efficiency Evaluation** in terms of distance traveled per voltage drop
- **Histogram Visualization** of linear speed
- **Correlation Matrix** between various metrics

## Results

- Most efficient robot: **MiniBot_2**
  - Distance Traveled: 28.8 cm
  - Battery Used: 0.47 V
  - Efficiency: 61.28 cm/V

## Requirements

```bash
pip install pandas matplotlib numpy
```

## Example Plots

- `linear_speed_histogram.png`
- `correlation_matrix.png`
- `battery_voltage_over_time.png`

