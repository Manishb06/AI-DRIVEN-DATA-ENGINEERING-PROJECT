import pandas as pd
import os

base_path = os.path.dirname(os.path.dirname(__file__))

input_path = os.path.join(base_path, "data", "log_dataset_1000.csv")
output_path = os.path.join(base_path, "data", "processed_logs.csv")

df = pd.read_csv(input_path)

print("Raw Data:")
print(df.head())

df = df.dropna()

df['response_time'] = df['response_time'].astype(float)
df['status'] = df['status'].astype(int)

df.to_csv(output_path, index=False)

print("\n✅ ETL Completed. Clean data saved at:", output_path)