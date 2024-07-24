from call import get_financial_prediction_list, post_added_data
from sqlalchemy import create_engine
import pandas as pd

db_connection_string = "postgresql://cvmdb_owner:n3YuMA6raJxh@ep-proud-pine-a4ahmncp.us-east-1.aws.neon.tech/cvmdb?sslmode=require"

def upload_predictions(cd_cvm_list, n_years):
    # Get predictions
    predictions_df = get_financial_prediction_list(cd_cvm_list, n_years)
    
    # Post added data
    processed_df = post_added_data(predictions_df)
    
    # Create a primary key by concatenating CD_CVM and Year
    processed_df['primary_key'] = range(1, len(processed_df) + 1)
    
    # Set the primary key as the index
    processed_df.set_index('primary_key', inplace=True)
    
    # Create database connection
    engine = create_engine(db_connection_string)
    
    # Upload to database
    table_name = 'financial_predictions'  # You may want to adjust this name
    processed_df.to_sql(table_name, engine, if_exists='append', index=True)  # Set index=True to include the primary key
    
    print(f"Uploaded {len(processed_df)} predictions to {table_name}")