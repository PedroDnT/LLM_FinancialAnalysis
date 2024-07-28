# --- Functions that can be moved to other files ---
from typing import List, Tuple, Dict
import os
import pandas as pd
import psycopg2
from psycopg2 import sql
import pandas as pd
from sqlalchemy import create_engine
from psycopg2.pool import SimpleConnectionPool
from contextlib import contextmanager
import psycopg2
from psycopg2 import sql
from contextlib import contextmanager
from dotenv import load_dotenv
from sklearn.metrics import precision_score, f1_score, accuracy_score
import numpy as np
import pandas as pd
load_dotenv()
db_connection_string ="postgresql://cvmdb_owner:n3YuMA6raJxh@ep-proud-pine-a4ahmncp.us-east-1.aws.neon.tech/cvmdb?sslmode=require"


def calculate_actual_results(income_statement: pd.DataFrame) -> Tuple[List[Tuple[str, int]], List[str]]:
    """Calculates actual earnings direction based on income statement."""
    earnings_column = 'Resultado Líquido das Operações Continuadas'
    results = []
    if 'DS_CONTA' not in income_statement.columns:
        raise ValueError("Expected 'DS_CONTA' column in income statement")
    earnings_rows = income_statement[income_statement['DS_CONTA'] == earnings_column]
    date_columns = [col for col in earnings_rows.columns if col.startswith('20') and col.endswith('-12-31')]
    sorted_dates = sorted(date_columns)
    for i in range(5, len(sorted_dates)):
        current_earnings = earnings_rows[sorted_dates[i]].values[0]
        previous_earnings = earnings_rows[sorted_dates[i-1]].values[0]
        if pd.notnull(current_earnings) and pd.notnull(previous_earnings):
            result = 1 if current_earnings > previous_earnings else -1
            period = sorted_dates[i]
            results.append((period, result))
    return results, sorted_dates

def get_financial_statements_batch(cd_cvm_list: List[str]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Fetches financial statements for a batch of CD_CVM codes."""
    income_statements = execute_query(cd_cvm_list, 'ist')
    balance_sheets = execute_query(cd_cvm_list, 'bs')
    cash_flows = execute_query(cd_cvm_list, 'cf')
    # beforw returning remove doct entries with all 0 or 0.0 or Nan on all columns
    income_statements = {cd_cvm: df.drop(columns=[col for col in df.columns if all(df[col] == 0) or all(df[col] == 0.0) or all(df[col].isna())]) for cd_cvm, df in income_statements.items()}
    balance_sheets = {cd_cvm: df.drop(columns=[col for col in df.columns if all(df[col] == 0) or all(df[col] == 0.0) or all(df[col].isna())]) for cd_cvm, df in balance_sheets.items()}
    cash_flows = {cd_cvm: df.drop(columns=[col for col in df.columns if all(df[col] == 0) or all(df[col] == 0.0) or all(df[col].isna())]) for cd_cvm, df in cash_flows.items()}
    return income_statements, balance_sheets, cash_flows

# Create a connection pool
pool = SimpleConnectionPool(1, 20, db_connection_string)

@contextmanager
def get_connection():
    connection = pool.getconn()
    try:
        yield connection
    finally:
        pool.putconn(connection)

def execute_query(CD_CVM_list, table_name):
    with get_connection() as conn:
        cursor = conn.cursor()
        query = sql.SQL("""
            SELECT "CD_CVM", "CD_CONTA", "DS_CONTA", 
                MAX(CASE WHEN "DT_FIM_EXERC" = '2010-12-31' THEN "VL_CONTA" END) AS "2010-12-31", 
                MAX(CASE WHEN "DT_FIM_EXERC" = '2011-12-31' THEN "VL_CONTA" END) AS "2011-12-31", 
                MAX(CASE WHEN "DT_FIM_EXERC" = '2012-12-31' THEN "VL_CONTA" END) AS "2012-12-31", 
                MAX(CASE WHEN "DT_FIM_EXERC" = '2013-12-31' THEN "VL_CONTA" END) AS "2013-12-31", 
                MAX(CASE WHEN "DT_FIM_EXERC" = '2014-12-31' THEN "VL_CONTA" END) AS "2014-12-31", 
                MAX(CASE WHEN "DT_FIM_EXERC" = '2015-12-31' THEN "VL_CONTA" END) AS "2015-12-31", 
                MAX(CASE WHEN "DT_FIM_EXERC" = '2016-12-31' THEN "VL_CONTA" END) AS "2016-12-31", 
                MAX(CASE WHEN "DT_FIM_EXERC" = '2017-12-31' THEN "VL_CONTA" END) AS "2017-12-31", 
                MAX(CASE WHEN "DT_FIM_EXERC" = '2018-12-31' THEN "VL_CONTA" END) AS "2018-12-31", 
                MAX(CASE WHEN "DT_FIM_EXERC" = '2019-12-31' THEN "VL_CONTA" END) AS "2019-12-31", 
                MAX(CASE WHEN "DT_FIM_EXERC" = '2020-12-31' THEN "VL_CONTA" END) AS "2020-12-31", 
                MAX(CASE WHEN "DT_FIM_EXERC" = '2021-12-31' THEN "VL_CONTA" END) AS "2021-12-31", 
                MAX(CASE WHEN "DT_FIM_EXERC" = '2022-12-31' THEN "VL_CONTA" END) AS "2022-12-31", 
                MAX(CASE WHEN "DT_FIM_EXERC" = '2023-12-31' THEN "VL_CONTA" END) AS "2023-12-31" 
            FROM 
                (
                    SELECT 
                        *
                    FROM 
                        {}
                    WHERE 
                        "CD_CVM" = ANY(%s) AND 
                        "ST_CONTA_FIXA" = 'S'
                ) AS filtered_data
            GROUP BY "CD_CVM", "CD_CONTA", "DS_CONTA"
            ORDER BY "CD_CONTA"
        """).format(sql.Identifier(table_name))
        
        try:
            cursor.execute(query, (CD_CVM_list,))
            columns = [desc[0] for desc in cursor.description]
            result = cursor.fetchall()
            print(f"Successfully executed the SQL query on the table '{table_name}' for the following CVM codes: {CD_CVM_list}")
            df = pd.DataFrame(result, columns=columns)
            # Drop columns where all rows are None
            df = df.dropna(axis=1, how='all')
            # Drop columns where all rowas are 0 or 0.0
            df = df.drop(columns=[col for col in df.columns if all(df[col] == 0) or all(df[col] == 0.0)])
            # Group by CD_CVM
            return {cd_cvm: group.drop(['CD_CVM', 'CD_CONTA'], axis=1) for cd_cvm, group in df.groupby('CD_CVM')}
        except psycopg2.Error as error:
            print(f"Error executing query: {error}")
            conn.rollback()
            print("Transaction rolled back.")
            return None
    
def get_distinct_cd_cvm():
    with get_connection() as conn:
        cursor = conn.cursor()
        query = sql.SQL("""
            SELECT DISTINCT "CD_CVM"
            FROM bs
            ORDER BY "CD_CVM";
        """)
        
        try:
            cursor.execute(query)
            result = cursor.fetchall()
            print(f"Query executed successfully. Retrieved {len(result)} distinct CD_CVM values.")
            # Convert the result to a list of CD_CVM values
            cd_cvm_list = [row[0] for row in result]
            return cd_cvm_list
        except psycopg2.Error as error:
            print(f"Error executing query: {error}")
            conn.rollback()
            
def get_company_name_by_cd_cvm(cd_cvm):
    with get_connection() as conn:
        cursor = conn.cursor()
        query = sql.SQL("""
            SELECT "DENOM_CIA"
            FROM bs
            WHERE "CD_CVM" = %s
            LIMIT 1;
        """)

        try:
            cursor.execute(query, (cd_cvm,))
            result = cursor.fetchone()
            if result:
                print(f"Query executed successfully. Retrieved company name: {result[0]}")
                return result[0]
            else:
                print("No company found for CD_CVM:", cd_cvm)
                return None
        except psycopg2.Error as error:
            print(f"Error executing query: {error}")
            conn.rollback()
            print("Transaction rolled back.")
            return None
  
def get_financial_predictions_table() -> pd.DataFrame:
    """Queries the financial_predictions table and returns all entries as a pandas DataFrame."""
    with get_connection() as conn:
        cursor = conn.cursor()
        query = sql.SQL("""
            SELECT *
            FROM financial_predictions;
        """)

        try:
            cursor.execute(query)
            result = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]  # Get column names
            print(f"Query executed successfully. Retrieved {len(result)} entries from financial_predictions table.")
            return pd.DataFrame(result, columns=columns)  # Convert to pandas DataFrame
        except psycopg2.Error as error:
            print(f"Error executing query: {error}")
            conn.rollback()
            print("Transaction rolled back.")
            return None
            
def calculate_metrics(df):
    metrics = []
    unique_cd_cvm = df['CD_CVM'].unique()
    
    for cd_cvm in unique_cd_cvm:
        subset = df[df['CD_CVM'] == cd_cvm]
        if not subset.empty:
            # Filter out NaN and infinite values
            subset = subset.replace([np.inf, -np.inf], np.nan).dropna(subset=['actual_earnings_direction', 'Prediction Direction'])
            
            y_true = subset['actual_earnings_direction'].astype(int)  # Ensure y_true is int
            y_pred = subset['Prediction Direction'].astype(int)  # Ensure y_pred is int
            
            # Filter out invalid predictions
            valid_indices = y_pred.isin([1, -1])
            y_true = y_true[valid_indices]
            y_pred = y_pred[valid_indices]
            
            if not y_true.empty and not y_pred.empty:
                name = subset['NAME'].iloc[0]  # Get the corresponding NAME
                
                f1 = round(f1_score(y_true, y_pred, average='weighted'), 2)
                accuracy = round(accuracy_score(y_true, y_pred), 2)
                precision = round(precision_score(y_true, y_pred, average='weighted', zero_division=0), 2)
                
                # Calculate weighted average of 'Average Logprob' weighted by 'Completion Tokens'
                weighted_avg_logprob = round((subset['Average Logprob'] * subset['Completion Tokens']).sum() / subset['Completion Tokens'].sum(), 2)
                
                # Calculate linear probability for each row and aggregate using log-sum-exp trick
                log_linear_probabilities = -subset['Average Logprob']
                aggregated_log_linear_probability = np.sum(log_linear_probabilities)
                aggregated_linear_probability = round(np.exp(aggregated_log_linear_probability) * 100, 0)
                
                # Count the number of valid predictions
                num_valid_predictions = valid_indices.sum()
                
                # Count the number of predictions for each direction
                num_predictions_1 = (y_pred == 1).sum()
                num_predictions_minus_1 = (y_pred == -1).sum()
                
                metrics.append({
                    'CD_CVM': cd_cvm,
                    'NAME': name,
                    'F1 Score': f1,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Weighted Avg Logprob': weighted_avg_logprob,
                    'Linear Probability': aggregated_linear_probability,
                    'Valid Predictions': num_valid_predictions,  # New column
                    'Predictions 1': num_predictions_1,  # New column
                    'Predictions -1': num_predictions_minus_1  # New column
                })
    
    return pd.DataFrame(metrics)

def calculate_agg_metrics(metrics_df):
    agg_metrics = {
        'Metric': ['F1 Score', 'Accuracy', 'Precision'],
        'Average': [
            metrics_df['F1 Score'].mean(),
            metrics_df['Accuracy'].mean(),
            metrics_df['Precision'].mean()
        ],
        'Standard Deviation': [
            metrics_df['F1 Score'].std(),
            metrics_df['Accuracy'].std(),
            metrics_df['Precision'].std()
        ]
    }
    
    return pd.DataFrame(agg_metrics)