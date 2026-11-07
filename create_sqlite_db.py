"""
Script to create SQLite database and run SQL analysis queries.
"""

import os
import sqlite3
import pandas as pd
from src.preprocessing import clean_data, add_engineered_features, load_raw_data

project_dir = os.path.dirname(os.path.abspath(__file__))
raw_data_path = os.path.join(project_dir, 'dataset', 'customer_churn.csv')
db_path = os.path.join(project_dir, 'dataset', 'telco_churn.db')

# Load and clean
df_raw = load_raw_data(raw_data_path)
df_clean = clean_data(df_raw)
df_feat = add_engineered_features(df_clean)

# Re-insert customerID if needed or create synthetic customer ID
df_feat['CustomerID'] = df_raw['customerID'] if 'customerID' in df_raw.columns else [f"CUST-{i:05d}" for i in range(1, len(df_feat)+1)]

# Save to SQLite
conn = sqlite3.connect(db_path)
df_feat.to_sql('customers', conn, if_exists='replace', index=False)
conn.close()

print(f"Database created successfully at {db_path} with {len(df_feat)} records.")

