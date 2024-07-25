from call import get_financial_prediction_list, post_added_data
from sqlalchemy import create_engine, Table, Column, String, Integer, Float, MetaData
from sqlalchemy.dialects.postgresql import insert
import pandas as pd
import os
from dotenv import load_dotenv
from typing import List, Optional

# Load environment variables
load_dotenv()

# Get database connection string from environment variable
db_connection_string = os.getenv('DB_CONNECTION_STRING')

def upload_predictions(cd_cvm_list: List[int], n_years: Optional[int] = None):
    try:
        print(f"Starting upload_predictions for CD_CVM list: {cd_cvm_list}, n_years: {n_years}")
        
        # Get predictions
        predictions_df = get_financial_prediction_list(cd_cvm_list, n_years)
        
        if predictions_df.empty:
            print("No predictions generated.")
            return 0
        
        print(f"Predictions generated. Shape: {predictions_df.shape}")
        print(f"Columns: {predictions_df.columns}")
        
        # Post added data
        processed_df = post_added_data(predictions_df)
        
        print(f"Processed data. Shape: {processed_df.shape}")
        print(f"Columns: {processed_df.columns}")
        
        # Set Year_CD_CVM as index
        processed_df['Year_CD_CVM'] = processed_df['Year'].astype(str) + '_' + processed_df['CD_CVM'].astype(str)
        processed_df.set_index('Year_CD_CVM', inplace=True)
        
        # Create database connection
        engine = create_engine(db_connection_string)
        
        # Define table structure
        metadata = MetaData()
        table_name = 'financial_predictions'
        
        # Create a dictionary to map DataFrame dtypes to SQLAlchemy column types
        dtype_map = {
            'int64': Integer,
            'float64': Float,
            'object': String
        }
        
        # Create table columns based on DataFrame structure
        columns = [Column('Year_CD_CVM', String, primary_key=True)]
        for column, dtype in processed_df.dtypes.items():
            if column != 'Year_CD_CVM':
                sql_type = dtype_map.get(str(dtype), String)
                # Handle the new Logprob Cumsum column
                if column == 'Logprob Cumsum':
                    sql_type = Float
                columns.append(Column(column, sql_type))
        
        # Create the table object
        table = Table(table_name, metadata, *columns)
        
        # Create table if it doesn't exist
        metadata.create_all(engine)
        
        # Convert DataFrame to list of dictionaries
        data = processed_df.reset_index().to_dict(orient='records')
        
        # Perform insert-ignore operation
        with engine.connect() as conn:
            stmt = insert(table).values(data)
            stmt = stmt.on_conflict_do_nothing(index_elements=['Year_CD_CVM'])
            result = conn.execute(stmt)
            conn.commit()
        
        num_uploaded = result.rowcount
        print(f"Inserted {num_uploaded} new predictions to {table_name}")
        return num_uploaded
    
    except Exception as e:
        print(f"An error occurred during prediction upload: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0