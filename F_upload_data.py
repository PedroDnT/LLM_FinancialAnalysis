import pandas as pd
import psycopg2
from psycopg2 import sql
from sqlalchemy import create_engine
import os

# Database connection string
user = 

db_connection_string = "postgresql://cvmdb_owner:n3YuMA6raJxh@ep-proud-pine-a4ahmncp.us-east-1.aws.neon.tech/cvmdb?sslmode=require"

# Create SQLAlchemy engine
engine = create_engine(db_connection_string)

# CSV files and corresponding table names
csv_files = {
    'income_statments_checked.csv': 'income_statements',
    'balance_sheets_checked.csv': 'balance_sheets',
    'cash_flows_checked.csv': 'cash_flows'
}

def create_table_and_upload_data(csv_file, table_name):
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Create table
    df.head(0).to_sql(table_name, engine, if_exists='replace', index=False)
    
    # Connect to the database
    conn = psycopg2.connect(db_connection_string)
    cur = conn.cursor()
    
    # Open the CSV file
    with open(csv_file, 'r') as f:
        # Skip the header row
        next(f)
        
        # Copy data from CSV to the table
        cur.copy_expert(f"COPY {table_name} FROM STDIN WITH CSV", f)
    
    # Commit the transaction
    conn.commit()
    
    # Close the cursor and connection
    cur.close()
    conn.close()
    
    print(f"Data from {csv_file} uploaded to {table_name} table.")

# Process each CSV file
for csv_file, table_name in csv_files.items():
    if os.path.exists(csv_file):
        create_table_and_upload_data(csv_file, table_name)
    else:
        print(f"File {csv_file} not found.")

print("Data upload complete.")
