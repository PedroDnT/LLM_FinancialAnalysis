from call import get_financial_prediction_list, post_added_data
from sqlalchemy import create_engine
import pandas as pd
import os
from dotenv import load_dotenv
from typing import List, Optional

# Load environment variables
load_dotenv()

# Get database connection string from environment variable
db_connection_string = os.getenv('DB_CONNECTION_STRING')

def upload_predictions(cd_cvm_list: List[int], n_years: Optional[int] = None):
    """
    Generate financial predictions and upload them to the database.
    
    Args:
    cd_cvm_list (list): List of CVM codes.
    n_years (int, optional): Number of years to predict. If None, predicts for all available years.
    
    Returns:
    int: Number of predictions uploaded.
    """
    try:
        # Get predictions
        predictions_df = get_financial_prediction_list(cd_cvm_list, n_years)
        
        if predictions_df.empty:
            print("No predictions generated.")
            return 0
        
        # Post added data
        processed_df = post_added_data(predictions_df)
        
        # Create database connection
        engine = create_engine(db_connection_string)
        
        # Upload to database
        table_name = 'financial_predictions'
        processed_df.to_sql(table_name, engine, if_exists='append', index=False)
        
        num_uploaded = len(processed_df)
        print(f"Uploaded {num_uploaded} predictions to {table_name}")
        return num_uploaded
    
    except Exception as e:
        print(f"An error occurred during prediction upload: {str(e)}")
        return 0

if __name__ == "__main__":
    # Example usage
    cd_cvm_list = [1234, 5678]  # Replace with actual CVM codes
    uploaded_count = upload_predictions(cd_cvm_list)
    print(f"Total predictions uploaded: {uploaded_count}")