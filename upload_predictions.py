from call import get_financial_prediction_list, post_added_data
from sqlalchemy import create_engine, Table, Column, String, Integer, Float, BigInteger, MetaData
from sqlalchemy.dialects.postgresql import insert
import pandas as pd
import os
from dotenv import load_dotenv
from typing import List, Optional
from tqdm import tqdm
import numpy as np
import psycopg2
from psycopg2 import sql

# Load environment variables
load_dotenv()

# Get database connection string from environment variable
db_connection_string = os.getenv('DB_CONNECTION_STRING')

def process_predictions(predictions_df: pd.DataFrame) -> pd.DataFrame:
    # Post added data
    processed_df = post_added_data(predictions_df)
    
    print(f"Processed data. Shape: {processed_df.shape}")
    print(f"Columns: {processed_df.columns}")
    
    # Ensure Year_CD_CVM is a string (if it exists)
    if 'Year_CD_CVM' in processed_df.columns:
        processed_df['Year_CD_CVM'] = processed_df['Year_CD_CVM'].astype(str)
    else:
        processed_df['Year_CD_CVM'] = processed_df['Year'].astype(str) + '_' + processed_df['CD_CVM'].astype(str)

    # Convert Year and CD_CVM to appropriate integer types
    processed_df['Year'] = processed_df['Year'].astype(int)
    processed_df['CD_CVM'] = processed_df['CD_CVM'].astype(int)

    # Convert numeric columns to appropriate types
    numeric_columns = ['Confidence', 'Completion Tokens', 'Prompt Tokens', 
                    'Average Logprob', 'Median Logprob', 'Std Logprob']
    for col in numeric_columns:
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].astype(float)
    
    # Reset the index to avoid the 'Unconsumed column names: index' error
    processed_df = processed_df.reset_index(drop=True)

    # Convert columns to appropriate data types
    processed_df['Year_CD_CVM'] = processed_df['Year_CD_CVM'].astype(str)
    processed_df['Year'] = processed_df['Year'].astype(int)
    processed_df['CD_CVM'] = processed_df['CD_CVM'].astype(int)
    processed_df['Completion Tokens'] = processed_df['Completion Tokens'].astype(int)
    processed_df['Prompt Tokens'] = processed_df['Prompt Tokens'].astype(int)

    # Handle potential out-of-range values
    processed_df['Average Logprob'] = processed_df['Average Logprob'].clip(-1e6, 1e6)
    processed_df['Median Logprob'] = processed_df['Median Logprob'].clip(-1e6, 1e6)
    processed_df['Std Logprob'] = processed_df['Std Logprob'].clip(-1e6, 1e6)

    # Replace NaN values in actual_earnings_direction with 0
    processed_df['actual_earnings_direction'] = processed_df['actual_earnings_direction'].fillna(0)

    return processed_df

def upload_to_database(processed_df: pd.DataFrame) -> int:
    # Create database connection
    engine = create_engine(db_connection_string)
    
    # Define table structure
    metadata = MetaData()
    table_name = 'financial_predictions'
    
    # Create a dictionary to map DataFrame dtypes to SQLAlchemy column types
    dtype_map = {
        'int64': BigInteger,
        'float64': Float,
        'object': String
    }
    
    # Create table columns based on DataFrame structure
    columns = [Column('Year_CD_CVM', String, primary_key=True)]
    for column, dtype in processed_df.dtypes.items():
        if column != 'Year_CD_CVM':
            sql_type = dtype_map.get(str(dtype), String)
            # Handle specific columns that need to be Float
            if column in ['Completion Tokens', 'Prompt Tokens', 'Average Logprob', 'Median Logprob', 'Std Logprob']:
                sql_type = Float
            elif column == 'Year':
                sql_type = Integer
            elif column == 'CD_CVM':
                sql_type = BigInteger
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
        update_dict = {c.name: c for c in stmt.excluded if c.name != 'Year_CD_CVM'}
        stmt = stmt.on_conflict_do_update(
            index_elements=['Year_CD_CVM'],
            set_=update_dict
        )
        result = conn.execute(stmt)
        conn.commit()
    
    return result.rowcount

def chunk_list(lst: List[int], chunk_size: int) -> List[List[int]]:
    """Helper function to split a list into chunks of specified size."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def upload_predictions(cd_cvm_list: List[int], n_years: Optional[int] = None) -> int:
    try:
        # Get the list of existing CD_CVM values
        existing_cd_cvm_list = get_unique_cd_cvm_from_predictions()
        
        # Filter out values present in the existing list
        original_length = len(cd_cvm_list)
        cd_cvm_list = [cd_cvm for cd_cvm in cd_cvm_list if cd_cvm not in existing_cd_cvm_list]
        filtered_length = len(cd_cvm_list)
        removed_count = original_length - filtered_length
        
        print(f"Removed {removed_count} values already present in the database.")
        
        # Remove duplicates from cd_cvm_list
        cd_cvm_list = list(set(cd_cvm_list))
        print(f"Starting upload_predictions for CD_CVM list: {cd_cvm_list}, n_years: {n_years}")
        
        total_uploaded = 0
        
        # Create a progress bar
        progress_bar = tqdm(total=len(cd_cvm_list), desc="Processing CD_CVM")
        
        # Process and upload in chunks
        for chunk in chunk_list(cd_cvm_list, 6):
            retry_count = 0
            while retry_count < 3:
                try:
                    # Get predictions
                    predictions_df = get_financial_prediction_list(chunk, n_years)
                    
                    if predictions_df.empty:
                        print(f"No predictions generated for chunk: {chunk}")
                        progress_bar.update(len(chunk))
                        break
                    
                    print(f"Predictions generated for chunk: {chunk}. Shape: {predictions_df.shape}")
                    #print(f"Columns: {predictions_df.columns}")
                    
                    # Process predictions
                    processed_df = process_predictions(predictions_df)
                    
                    # Upload to database
                    num_uploaded = upload_to_database(processed_df)
                    total_uploaded += num_uploaded
                    
                    print(f"\nInserted/updated {num_uploaded} rows for chunk: {chunk}.")
                    
                    # Update progress bar
                    progress_bar.update(len(chunk))
                    break
                
                except Exception as e:
                    retry_count += 1
                    print(f"Error occurred during processing chunk {chunk}: {str(e)}")
                    if retry_count < 3:
                        print(f"Retrying... (Attempt {retry_count + 1}/3)")
                    else:
                        print(f"Failed to process chunk {chunk} after 3 attempts. Removing problematic CD_CVM and retrying.")
                        # Identify and remove the problematic CD_CVM
                        for cd_cvm in chunk:
                            try:
                                # Try to get predictions for individual CD_CVM to identify the problematic one
                                get_financial_prediction_list([cd_cvm], n_years)
                            except Exception as inner_e:
                                print(f"Problematic CD_CVM identified: {cd_cvm}")
                                chunk.remove(cd_cvm)
                                break
                        retry_count = 0  # Reset retry count after removing problematic CD_CVM
        
        # Close the progress bar
        progress_bar.close()
        
        return total_uploaded
    
    except Exception as e:
        print(f"\nAn error occurred during prediction upload: {str(e)}")
        print("Error details:")
        print(e.__class__.__name__)
        print(e.__dict__)
        import traceback
        traceback.print_exc()
        return 0

def get_unique_cd_cvm_from_predictions():
    db_connection_string = "postgresql://cvmdb_owner:n3YuMA6raJxh@ep-proud-pine-a4ahmncp.us-east-1.aws.neon.tech/cvmdb?sslmode=require"
    try:
        conn = psycopg2.connect(db_connection_string)
        cursor = conn.cursor()
        query = sql.SQL("""
            SELECT DISTINCT "CD_CVM"
            FROM financial_predictions
            ORDER BY "CD_CVM";
        """)
        
        cursor.execute(query)
        result = cursor.fetchall()
        cd_cvm_list = [row[0] for row in result]
        return cd_cvm_list
    except psycopg2.Error as error:
        print(f"Error executing query: {error}")
        if conn:
            conn.rollback()
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()