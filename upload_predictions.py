from call import get_financial_prediction_list, post_added_data
from sqlalchemy import create_engine, Table, Column, String, Integer, Float, BigInteger, MetaData
from sqlalchemy.dialects.postgresql import insert
import pandas as pd
import os
from dotenv import load_dotenv
from typing import List, Optional
from tqdm import tqdm
import numpy as np

# Load environment variables
load_dotenv()

# Get database connection string from environment variable
db_connection_string = os.getenv('DB_CONNECTION_STRING')

def upload_predictions(cd_cvm_list: List[int], n_years: Optional[int] = None):
    try:
        print(f"Starting upload_predictions for CD_CVM list: {cd_cvm_list}, n_years: {n_years}")
        
        # Create a progress bar
        progress_bar = tqdm(total=len(cd_cvm_list), desc="Processing CD_CVM")
        
        # Get predictions
        predictions_df = get_financial_prediction_list(cd_cvm_list, n_years)
        
        # Update progress bar
        progress_bar.update(len(cd_cvm_list))
        
        if predictions_df.empty:
            print("No predictions generated.")
            progress_bar.close()
            return 0
        
        print(f"Predictions generated. Shape: {predictions_df.shape}")
        print(f"Columns: {predictions_df.columns}")
        
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

        # Debug prints
        print("\n--- Debug Information ---")
        print("\nDataFrame info:")
        processed_df.info()

        print("\nDataFrame dtypes:")
        print(processed_df.dtypes)

        print("\nSample data:")
        print(processed_df.head())

        print("\nColumn names:")
        print(processed_df.columns)

        print("\nDescribe numeric columns:")
        print(processed_df.describe())

        print("\nNon-numeric columns info:")
        for col in processed_df.select_dtypes(exclude=['number']).columns:
            print(f"\n{col}:")
            print(processed_df[col].value_counts())
            print(f"Unique values: {processed_df[col].nunique()}")
            print(f"Sample values: {processed_df[col].sample(5).tolist()}")

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
        
        num_uploaded = result.rowcount
        print(f"\nInserted/updated {num_uploaded} rows.")
        
        # Close the progress bar
        progress_bar.close()
        
        return num_uploaded
    
    except Exception as e:
        print(f"\nAn error occurred during prediction upload: {str(e)}")
        print("Error details:")
        print(e.__class__.__name__)
        print(e.__dict__)
        import traceback
        traceback.print_exc()
        return 0