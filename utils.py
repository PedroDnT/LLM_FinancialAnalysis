# --- Functions that can be moved to other files ---
from typing import List, Tuple, Dict
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
from sklearn.metrics import precision_score, f1_score
import pandas as pd
load_dotenv()


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

# These functions can be moved to a separate file, e.g., "utils.py"


# --- Functions related to financial data fetching ---

def get_financial_statements_batch(cd_cvm_list: List[str]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Fetches financial statements for a batch of CD_CVM codes."""
    income_statements = execute_query(cd_cvm_list, 'ist')
    balance_sheets = execute_query(cd_cvm_list, 'bs')
    cash_flows = execute_query(cd_cvm_list, 'cf')
    return income_statements, balance_sheets, cash_flows

db_connection_string = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}?sslmode=require"

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
            SELECT "CD_CVM", "DS_CONTA", 
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
                        "CD_CVM",
                        "DS_CONTA", 
                        "DT_FIM_EXERC", 
                        "VL_CONTA"
                    FROM 
                        {}
                    WHERE 
                        "CD_CVM" = ANY(%s) AND 
                        "ST_CONTA_FIXA" = 'S'
                ) AS filtered_data
            GROUP BY "CD_CVM", "DS_CONTA"
        """).format(sql.Identifier(table_name))
        
        try:
            cursor.execute(query, (CD_CVM_list,))
            columns = [desc[0] for desc in cursor.description]
            result = cursor.fetchall()
            print(f"Query executed successfully for table '{table_name}' and CD_CVM list: {CD_CVM_list}")
            df = pd.DataFrame(result, columns=columns)
            # Drop columns where all rows are None
            df = df.dropna(axis=1, how='all')
            # Group by CD_CVM
            return {cd_cvm: group.drop('CD_CVM', axis=1) for cd_cvm, group in df.groupby('CD_CVM')}
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
  
def analyze_model_performance(df):
    grouped = df.groupby(['Model', 'Company'])

    # Initialize an empty DataFrame to store the results
    results = pd.DataFrame(columns=['Model', 'Company', 'Precision', 'F1 Score', 'Average Confidence Level', 'Count Magnitude'])

    for name, group in grouped:
        # Calculate precision and F1 score
        precision = precision_score(group['ACTUAL DIRECTION'], group['DIRECTION'], average='binary', zero_division=0)
        f1 = f1_score(group['ACTUAL DIRECTION'], group['DIRECTION'], average='binary', zero_division=0)
        
        # Calculate average confidence level
        avg_confidence = group['CONFIDENCE LEVEL'].mean()
        
        # Count values in 'MAGNITUDE'
        count_magnitude = group['MAGNITUDE'].value_counts().to_dict()
        
        # Create a DataFrame for the current results and concatenate it with the main results DataFrame
        current_results = pd.DataFrame([{
            'Model': name[0],
            'Company': name[1],
            'Precision': precision,
            'F1 Score': f1,
            'Average Confidence Level': avg_confidence,
            'Count Magnitude': count_magnitude
        }])
        results = pd.concat([results, current_results], ignore_index=True)

    return results
