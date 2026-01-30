import sqlite3
import pandas as pd

conn = sqlite3.connect('plant_disease.db')

query = "SELECT * FROM inference_logs ORDER BY id DESC"
df = pd.read_sql_query(query, conn)

if not df.empty:
    print("\n--- Inference Logs (Most Recent First) ---")
    print(df.to_string(index=False))
    
    print("\n--- Analytics ---")
    print(f"Total Scans: {len(df)}")
    print(f"Average Confidence: {df['confidence'].mean() * 100:.2f}%")
else:
    print("Database is empty. Run inference.py first!")

conn.close()