README:
# AI-Driven Log Anomaly Detection

## Overview
This project implements a machine learning-based system to detect anomalies in system log data. It uses a simple ETL pipeline for preprocessing and the Isolation Forest algorithm for identifying abnormal behavior.

## Dataset
The dataset contains system logs with the following fields:
- timestamp
- level (INFO, WARN, ERROR)
- response_time
- status (200, 404, 500)

## Methodology
1. Load and clean log data using pandas (ETL)
2. Select features: response_time and status
3. Apply Isolation Forest for anomaly detection
4. Label logs as normal (1) or anomaly (-1)
5. Generate summary and visualizations

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Matplotlib

## Output
- Processed dataset with anomaly labels
- Anomaly count and percentage
- Graphical analysis of system behavior

## How to Run
```bash
python etl/spark_etl.py
python ml/anomaly.py
