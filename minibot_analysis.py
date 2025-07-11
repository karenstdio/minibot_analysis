# Import Dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load and Preprocess Data
df = pd.read_csv('minibot_logs.csv', parse_dates=['timestamp'])
df.set_index('timestamp', inplace=True)
df.sort_index(inplace=True)

# Basic Statistics
print(df.describe())
print(df.groupby('robot_id')['left_rpm'].mean())

# Resample Data to 1-second intervals
df_numeric = df.select_dtypes(include='number')
df_1s = df_numeric.resample('1S').mean()

expected = 10  # expected packets per second
missing = df.groupby('robot_id').resample('1S')['left_rpm'].count().reset_index()
missing['lost_packets'] = expected - missing['left_rpm']
print(missing.head())

# Calculate Linear Speed (cm/s)
wheel_radius_cm = 3.3
rpm_to_cm_s = (2 * np.pi * wheel_radius_cm) / 60
df['linear_speed_cm_s'] = ((df['left_rpm'] + df['right_rpm']) / 2) * rpm_to_cm_s

# Battery Voltage Over Time
plt.figure(figsize=(10, 4))
for rid, sub in df.groupby('robot_id'):
    sub['battery_v'].plot(label=rid)
plt.legend()
plt.ylabel('Battery Voltage (V)')
plt.title('Battery Voltage Over Time')
plt.grid(True)
plt.tight_layout()
plt.show()

# Turning Detection
df['rpm_diff'] = df['left_rpm'] - df['right_rpm']
turns = df[np.abs(df['rpm_diff']) > 30]
print(turns.head())

# Summary Statistics Per Robot
summary = df.groupby('robot_id').agg({
    'linear_speed_cm_s': ['mean', 'max'],
    'battery_v': ['mean', 'min'],
    'distance_cm': 'mean'
})
print(summary)

# Distance per Minute
minute_dist = (df.reset_index()
                  .assign(minute=lambda x: x['timestamp'].dt.floor('T'))
                  .groupby(['minute','robot_id'])['distance_cm'].mean()
                  .unstack())
print(minute_dist.head())

# Correlation Matrix
corr = df[['left_rpm','right_rpm','linear_speed_cm_s','battery_v','distance_cm']].corr()
plt.figure(figsize=(6,4))
plt.imshow(corr, cmap='viridis')
plt.xticks(range(len(corr)), corr.columns, rotation=45)
plt.yticks(range(len(corr)), corr.columns)
plt.colorbar()
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# Anomaly Detection (Global)
threshold = df['battery_v'].mean() - 3 * df['battery_v'].std()
anomalies = df[df['battery_v'] < threshold]
print(anomalies.head())

# Anomaly Detection (Per Robot)
def detect_anomalies(group):
    threshold = group['battery_v'].mean() - 3 * group['battery_v'].std()
    group = group.copy()
    group['anomaly'] = group['battery_v'] < threshold
    return group

df = df.groupby('robot_id', group_keys=False).apply(detect_anomalies)
anomalies = df[df['anomaly']].copy()

# Plot Battery + Anomalies
plt.figure(figsize=(10, 4))
for rid, sub in df.groupby('robot_id'):
    sub['battery_v'].plot(label=rid)
plt.scatter(anomalies.index.to_pydatetime(), anomalies['battery_v'], color='red', label='Anomaly', zorder=5)
plt.legend()
plt.ylabel('Battery Voltage (V)')
plt.title('Battery Voltage and Anomalies')
plt.grid(True)
plt.tight_layout()
plt.show()

# Packet Loss Detection
df = pd.read_csv('minibot_logs.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

results = []

for robot_id, group in df.groupby('robot_id'):
    group = group.sort_values('timestamp')
    time_diff = (group['timestamp'].max() - group['timestamp'].min()).total_seconds()
    expected_packets = int(time_diff / 0.1) + 1
    actual_packets = len(group)
    missing_packets = expected_packets - actual_packets
    missing_ratio = (missing_packets / expected_packets) * 100

    results.append({
        'robot_id': robot_id,
        'expected_packets': expected_packets,
        'actual_packets': actual_packets,
        'missing_packets': missing_packets,
        'missing_ratio_percent': round(missing_ratio, 2)
    })

missing_df = pd.DataFrame(results)
print(missing_df)

# Efficiency Calculation (cm/volt)
df['timestamp'] = pd.to_datetime(df['timestamp'])

efficiency_results = []

for robot_id, group in df.groupby('robot_id'):
    group = group.sort_values('timestamp')
    start_distance = group['distance_cm'].iloc[0]
    end_distance = group['distance_cm'].iloc[-1]
    start_battery = group['battery_v'].iloc[0]
    end_battery = group['battery_v'].iloc[-1]
    distance_travelled = end_distance - start_distance
    battery_used = start_battery - end_battery
    efficiency = distance_travelled / battery_used if battery_used != 0 else float('inf')

    efficiency_results.append({
        'robot_id': robot_id,
        'distance_cm': round(distance_travelled, 2),
        'battery_used_v': round(battery_used, 4),
        'efficiency_cm_per_v': round(efficiency, 2)
    })

efficiency_df = pd.DataFrame(efficiency_results)
most_efficient = efficiency_df.sort_values(by='efficiency_cm_per_v', ascending=False).iloc[0]

print(efficiency_df)
print(f"\n🔋 Most Efficient Robot: {most_efficient['robot_id']}")
print(f"→ Distance Travelled: {most_efficient['distance_cm']} cm")
print(f"→ Battery Used: {most_efficient['battery_used_v']} V")
print(f"→ Efficiency: {most_efficient['efficiency_cm_per_v']} cm/V")