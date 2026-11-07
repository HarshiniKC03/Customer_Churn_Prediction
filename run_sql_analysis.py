"""
Execute SQL analysis queries against SQLite and print summary results.
"""

import os
import sqlite3
import pandas as pd

project_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(project_dir, 'dataset', 'telco_churn.db')
sql_file_path = os.path.join(project_dir, 'sql', 'customer_analysis.sql')

conn = sqlite3.connect(db_path)

with open(sql_file_path, 'r') as f:
    sql_script = f.read()

# Split individual queries by semicolon
queries = [q.strip() for q in sql_script.split(';') if q.strip()]

titles = [
    "1. Overall Churn Rate & Revenue Loss",
    "2. Churn Rate & Revenue Loss by Contract Type",
    "3. Tenure Cohort Analysis",
    "4. Internet Service Type & Tech Support Impact",
    "5. Payment Method & Friction Analysis",
    "6. High-Value Customer Churn Sample",
    "7. Multi-Service Bundling Effect",
    "8. Senior Citizen & Family Demographics Analysis"
]

print("=======================================================")
print("EXECUTING SQL QUERIES FROM sql/customer_analysis.sql")
print("=======================================================\n")

for i, query in enumerate(queries):
    title = titles[i] if i < len(titles) else f"Query {i+1}"
    print(f"--- {title} ---")
    try:
        df_res = pd.read_sql_query(query, conn)
        print(df_res.to_string(index=False))
        print("\n")
    except Exception as e:
        print(f"Error running query: {e}\n")

conn.close()
print("SQL Analysis Completed Successfully!")

