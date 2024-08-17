from call import get_financial_prediction_list
from call_groq import get_financial_prediction_list_groq
from sqlalchemy import create_engine, Table, Column, String, Integer, Float, BigInteger, MetaData, and_
from sqlalchemy.dialects.postgresql import insert
import pandas as pd
import os
from dotenv import load_dotenv
from typing import List, Optional
from tqdm import tqdm
import numpy as np
import psycopg2
from psycopg2 import sql
from utils import get_distinct_cd_cvm, get_connection

# Load environment variables
load_dotenv()

# Get database connection string from environment variable
db_connection_string = os.getenv('DB_CONNECTION_STRING')

def process_predictions(predictions_df: pd.DataFrame) -> pd.DataFrame:
    # Post added data
    processed_df = predictions_df
    
    print(f"Processed data. Shape: {processed_df.shape}")
    print(f"Columns: {processed_df.columns}")
    
    # Ensure Year_CD_CVM is a string
    processed_df['Year_CD_CVM'] = processed_df['Year'].astype(str) + '_' + processed_df['CD_CVM'].astype(str)

    # Convert Year and CD_CVM to appropriate integer types
    processed_df['Year'] = processed_df['Year'].astype(int)
    processed_df['CD_CVM'] = processed_df['CD_CVM'].astype(int)

    # Convert numeric columns to appropriate types
    numeric_columns = ['Confidence', 'Prompt Tokens', 'Completion Tokens', 'Total Tokens']
    for col in numeric_columns:
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].astype(float)
    
    # Reset the index to avoid the 'Unconsumed column names: index' error
    processed_df = processed_df.reset_index(drop=True)

    # Replace NaN values in actual_earnings_direction with 0
    processed_df['actual_earnings_direction'] = processed_df['actual_earnings_direction'].fillna(0)

    return processed_df

def upload_to_database(processed_df: pd.DataFrame, table_name: str) -> int:
    # Create database connection
    engine = create_engine(db_connection_string)
    
    # Define table structure
    metadata = MetaData()
    
    # Create a dictionary to map DataFrame dtypes to SQLAlchemy column types
    dtype_map = {
        'int64': BigInteger,
        'float64': Float,
        'object': String
    }
    
    # Create table columns based on DataFrame structure
    columns = [
        Column('Year_CD_CVM', String, primary_key=True),
        Column('Model Name', String, primary_key=True)
    ]
    for column, dtype in processed_df.dtypes.items():
        if column not in ['Year_CD_CVM', 'Model Name']:
            sql_type = dtype_map.get(str(dtype), String)
            if column in ['Prompt Tokens', 'Completion Tokens', 'Total Tokens']:
                sql_type = Integer
            elif column == 'Year':
                sql_type = Integer
            elif column == 'CD_CVM':
                sql_type = BigInteger
            elif column == 'Confidence':
                sql_type = Float
            columns.append(Column(column, sql_type))
    
    # Create the table object
    table = Table(table_name, metadata, *columns)
    
    # Create table if it doesn't exist
    metadata.create_all(engine)
    
    # Convert DataFrame to list of dictionaries
    data = processed_df.to_dict(orient='records')
    
    # Perform insert-update operation
    with engine.connect() as conn:
        stmt = insert(table).values(data)
        update_dict = {c.name: c for c in stmt.excluded if c.name not in ['Year_CD_CVM', 'Model Name']}
        stmt = stmt.on_conflict_do_update(
            index_elements=['Year_CD_CVM', 'Model Name'],
            set_=update_dict,
            where=and_(
                table.c.Year_CD_CVM == stmt.excluded.Year_CD_CVM,
                table.c['Model Name'] == stmt.excluded['Model Name']
            )
        )
        result = conn.execute(stmt)
        conn.commit()
    
    return result.rowcount

def chunk_list(lst: List[int], chunk_size: int) -> List[List[int]]:
    """Helper function to split a list into chunks of specified size."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def add_unique_constraint(table_name, constraint_name, columns):
    with get_connection() as conn:
        cursor = conn.cursor()
        try:
            # Check if the constraint already exists
            cursor.execute(f"""
                SELECT constraint_name 
                FROM information_schema.table_constraints 
                WHERE table_name = '{table_name}' 
                AND constraint_name = '{constraint_name}'
            """)
            if cursor.fetchone():
                print(f"Constraint {constraint_name} already exists on table {table_name}.")
                return

            # If the constraint doesn't exist, add it
            column_names = ', '.join(f'"{col}"' for col in columns)
            cursor.execute(f"""
                ALTER TABLE {table_name}
                ADD CONSTRAINT {constraint_name} UNIQUE ({column_names});
            """)
            conn.commit()
            print(f"Unique constraint added to table {table_name}.")
        except Exception as e:
            print(f"Error adding unique constraint: {e}")

def get_existing_values(table_name):
    db_connection_string = os.getenv('DB_CONNECTION_STRING')
    try:
        conn = psycopg2.connect(db_connection_string)
        cursor = conn.cursor()
        query = sql.SQL("""
            SELECT DISTINCT "Year_CD_CVM", "Model Name"
            FROM {};
        """).format(sql.Identifier(table_name))
        
        cursor.execute(query)
        result = cursor.fetchall()
        existing_values = [(row[0], row[1]) for row in result]
        return existing_values
    except psycopg2.Error as error:
        print(f"Error executing query: {error}")
        if conn:
            conn.rollback()
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def create_table_if_not_exists(table_name):
    with get_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    "Year_CD_CVM" VARCHAR(255),
                    "Model Name" VARCHAR(255),
                    "Year" INTEGER,
                    "CD_CVM" INTEGER,
                    "Panel A" TEXT,
                    "Panel B" TEXT,
                    "Panel C" TEXT,
                    "Prediction Direction" VARCHAR(10),
                    "Magnitude" VARCHAR(50),
                    "Confidence" FLOAT,
                    "Prompt Tokens" INTEGER,
                    "Completion Tokens" INTEGER,
                    "Total Tokens" INTEGER,
                    actual_earnings_direction INTEGER,
                    "NAME" VARCHAR(255),
                    PRIMARY KEY ("Year_CD_CVM", "Model Name")
                )
            """)
            conn.commit()
            print(f"Table {table_name} created successfully or already exists.")
        except Exception as e:
            print(f"Error creating table: {e}")



def insert_predictions(predictions_df: pd.DataFrame, table_name: str):
    with get_connection() as conn:
        cursor = conn.cursor()
        try:
            # Ensure all required columns are present
            required_columns = ["Year_CD_CVM", "Model Name", "Year", "CD_CVM", "Panel A", "Panel B", "Panel C", 
                                "Prediction Direction", "Magnitude", "Confidence", "Prompt Tokens", 
                                "Completion Tokens", "Total Tokens", "actual_earnings_direction", "NAME"]
            
            for col in required_columns:
                if col not in predictions_df.columns:
                    predictions_df[col] = None  # Add missing columns with None values
            
            # Select only the required columns in the correct order
            predictions_df = predictions_df[required_columns]
            
            # Convert DataFrame to list of tuples
            data = [tuple(x) for x in predictions_df.to_numpy()]
            
            # Prepare the INSERT statement
            columns = ', '.join(f'"{col}"' for col in required_columns)
            placeholders = ', '.join(['%s'] * len(required_columns))
            insert_query = f"""
                INSERT INTO {table_name} ({columns})
                VALUES ({placeholders})
                ON CONFLICT ("Year_CD_CVM", "Model Name") DO UPDATE
                SET 
                    "Year" = EXCLUDED."Year",
                    "CD_CVM" = EXCLUDED."CD_CVM",
                    "Panel A" = EXCLUDED."Panel A",
                    "Panel B" = EXCLUDED."Panel B",
                    "Panel C" = EXCLUDED."Panel C",
                    "Prediction Direction" = EXCLUDED."Prediction Direction",
                    "Magnitude" = EXCLUDED."Magnitude",
                    "Confidence" = EXCLUDED."Confidence",
                    "Prompt Tokens" = EXCLUDED."Prompt Tokens",
                    "Completion Tokens" = EXCLUDED."Completion Tokens",
                    "Total Tokens" = EXCLUDED."Total Tokens",
                    actual_earnings_direction = EXCLUDED.actual_earnings_direction,
                    "NAME" = EXCLUDED."NAME"
            """
            
            # Execute the INSERT statement
            cursor.executemany(insert_query, data)
            conn.commit()
            print(f"Inserted {len(data)} rows into {table_name}")
        except Exception as e:
            conn.rollback()
            print(f"Error inserting predictions: {e}")
            print(f"DataFrame columns: {predictions_df.columns}")
            print(f"DataFrame shape: {predictions_df.shape}")
            print(f"First row of data: {predictions_df.iloc[0] if not predictions_df.empty else 'Empty DataFrame'}")
            raise

def get_existing_values(table_name):
    db_connection_string = os.getenv('DB_CONNECTION_STRING')
    try:
        conn = psycopg2.connect(db_connection_string)
        cursor = conn.cursor()
        query = sql.SQL("""
            SELECT DISTINCT "CD_CVM", "Model Name"
            FROM {};
        """).format(sql.Identifier(table_name))
        
        cursor.execute(query)
        result = cursor.fetchall()
        existing_values = set((row[0], row[1]) for row in result)
        return existing_values
    except psycopg2.Error as error:
        print(f"Error executing query: {error}")
        if conn:
            conn.rollback()
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def upload_predictions(cd_cvm_list, table_name='ibov', n_years=None, provider='openai'):
    create_table_if_not_exists(table_name)
    add_unique_constraint(table_name, f"{table_name}_year_cd_cvm_model_name_key", ["Year_CD_CVM", "Model Name"])

    # Get existing values
    existing_values = get_existing_values(table_name)
    
    # Filter cd_cvm_list based on existing values and provider
    filtered_cd_cvm_list = [cd_cvm for cd_cvm in cd_cvm_list if (cd_cvm, provider) not in existing_values]
    
    print(f"Removed {len(cd_cvm_list) - len(filtered_cd_cvm_list)} values already present in the database for the current provider.")

    print(f"Starting upload_predictions for CD_CVM list: {filtered_cd_cvm_list}, n_years: {n_years}")

    for cd_cvm in tqdm(filtered_cd_cvm_list, desc="Processing CD_CVM"):
        print(f"Processing CD_CVM: {cd_cvm}")
        try:
            if provider == 'groq':
                predictions_df = get_financial_prediction_list_groq([cd_cvm], n_years)
            else:  # default to OpenAI
                predictions_df = get_financial_prediction_list([cd_cvm], n_years)

            if predictions_df.empty:
                print(f"No predictions generated for CD_CVM: {cd_cvm}")
                continue

            # Ensure the DataFrame has the correct structure
            required_columns = ["Year_CD_CVM", "Model Name", "Year", "CD_CVM", "Panel A", "Panel B", "Panel C", 
                                "Prediction Direction", "Magnitude", "Confidence", "Prompt Tokens", 
                                "Completion Tokens", "Total Tokens", "actual_earnings_direction", "NAME"]
            
            for col in required_columns:
                if col not in predictions_df.columns:
                    predictions_df[col] = None

            predictions_df = predictions_df[required_columns]

            insert_predictions(predictions_df, table_name)
            print(f"Predictions for CD_CVM {cd_cvm} inserted successfully.")
        except Exception as e:
            print(f"Error processing CD_CVM: {cd_cvm}. Error: {str(e)}")
if __name__ == "__main__":
    distinct = get_distinct_cd_cvm()
    upload_predictions(distinct, "financial_predictions_new")