
import pandas as pd
import numpy as np
from handler import read_files, read_files_ref, aggregate_df, remove_accents
from tqdm import tqdm
import time
import sys
import multiprocessing as mp

def process_statement(statement_type):
    df = read_files(statement_type)
    df = aggregate_df(df)
    df.columns = [remove_accents(col) for col in df.columns]
    return df

def create_csv_files(n=None):
    start_time = time.time()

    ref_df = read_files_ref()
    unique_cd_cvms = ref_df['CD_CVM'].unique()

    if n is not None:
        unique_cd_cvms = unique_cd_cvms[:n]

    print(f"Number of CD_CVMs being processed: {len(unique_cd_cvms)}")

    read_time = time.time()
    print(f"Time to read reference data: {read_time - start_time:.2f} seconds")

    # Use multiprocessing to process statements in parallel
    with mp.Pool(processes=3) as pool:
        results = pool.map(process_statement, ['CF', 'BS', 'IS'])

    process_time = time.time()
    print(f"Time to process statements: {process_time - read_time:.2f} seconds")

    cf_data, bs_data, is_data = results

    # Save to CSV files, overwriting existing files
    cf_data.to_csv('cash_flows.csv', index=False, mode='w', encoding='utf-8', decimal='.')
    bs_data.to_csv('balance_sheets.csv', index=False, mode='w', encoding='utf-8', decimal='.')
    is_data.to_csv('income_statments.csv', index=False, mode='w', encoding='utf-8', decimal='.')

    save_time = time.time()
    print(f"Time to save CSV files: {save_time - process_time:.2f} seconds")

    total_time = time.time() - start_time
    print(f"\nTotal time elapsed: {total_time:.2f} seconds")

    print("CSV files creates succesfully!")

if __name__ == "__main__":
    n = None
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except ValueError:
            if sys.argv[1].lower() != 'none':
                print("Argumento invalido. Processando todos os codigos.")
    create_csv_files(n)
