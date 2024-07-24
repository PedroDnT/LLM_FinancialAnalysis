import os
import psycopg2
from psycopg2 import sql
import pandas as pd
from sqlalchemy import create_engine
from psycopg2.pool import SimpleConnectionPool
from contextlib import contextmanager
import psycopg2
from psycopg2 import sql
from contextlib import contextmanager

db_connection_string ="postgresql://cvmdb_owner:n3YuMA6raJxh@ep-proud-pine-a4ahmncp.us-east-1.aws.neon.tech/cvmdb?sslmode=require"

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
            print("Query executed successfully.")
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
            print("Query executed successfully.")
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
                print("Query executed successfully.")
                return result[0]
            else:
                print("No company found for CD_CVM:", cd_cvm)
                return None
        except psycopg2.Error as error:
            print(f"Error executing query: {error}")
            conn.rollback()
            print("Transaction rolled back.")
            return None
  


        
