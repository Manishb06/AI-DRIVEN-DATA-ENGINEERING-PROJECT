import pandas as pd
import os
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

base_path = os.path.dirname(os.path.dirname(__file__))

input_path = os.path.join(base_path, "data", "processed_logs.csv")
output_path = os.path.join(base_path, "output", "anomaly_output.csv")

os.makedirs(os.path.join(base_path, "output"), exist_ok=True)

df = pd.read_csv(input_path)

features = df[['response_time', 'status']]

model = IsolationForest(contamination=0.05, random_state=42)
df['anomaly'] = model.fit_predict(features)

total = len(df)
anomalies = len(df[df['anomaly'] == -1])

print("\n📊 Summary:")
print("Total logs:", total)
print("Anomalies:", anomalies)
print("Anomaly %:", (anomalies / total) * 100)

df.to_csv(output_path, index=False)

print("\n✅ Output saved at:", output_path)

import matplotlib.pyplot as plt

normal = df[df['anomaly'] == 1]
anomaly = df[df['anomaly'] == -1]

plt.figure(figsize=(10, 6))

plt.scatter(normal.index, normal['response_time'],
            label="Normal Logs",
            alpha=0.6)

plt.scatter(anomaly.index, anomaly['response_time'],
            label="Anomalies",
            marker='x')

plt.title("Log Anomaly Detection", fontsize=14)
plt.xlabel("Log Index", fontsize=12)
plt.ylabel("Response Time", fontsize=12)

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

counts = df['anomaly'].value_counts()

plt.figure()
plt.bar(["Normal", "Anomaly"], [counts[1], counts[-1]])
plt.title("Anomaly vs Normal Logs")
plt.ylabel("Count")
plt.show()
