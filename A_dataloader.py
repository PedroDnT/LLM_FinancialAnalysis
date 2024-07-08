import requests
from bs4 import BeautifulSoup
import os
import zipfile
import shutil
from tqdm import tqdm
from collections import Counter

def download_cvm_zip_files():
    url = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/DFP/DADOS/"
    
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    zip_links = [link.get('href') for link in soup.find_all('a') if link.get('href', '').endswith('.zip')]
    
    if not os.path.exists('cvm_zip_files'):
        os.makedirs('cvm_zip_files')
    
    for link in tqdm(zip_links, desc="Downloading zip files"):
        file_url = url + link
        file_name = os.path.join('cvm_zip_files', link)
        
        if os.path.exists(file_name):
            print(f"File {link} already exists. Skipping download.")
            continue
        
        file_response = requests.get(file_url, stream=True)
        total_size = int(file_response.headers.get('content-length', 0))
        
        with open(file_name, 'wb') as file, tqdm(
            desc=link,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in file_response.iter_content(chunk_size=1024):
                size = file.write(data)
                progress_bar.update(size)
    
    print("All zip files have been downloaded.")

def is_valid_file(filename):
    # Keep files with the format dfp_cia_aberta_{year}.csv
    if filename.startswith('dfp_cia_aberta_') and filename.count('_') == 3:
        return True
    
    if 'ind' in filename.lower():
        return False
    if 'MD' in filename:
        return False
    if filename.startswith('dfp_cia_aberta_parecer_'):
        return False
    parts = filename.split('_')
    if len(parts) >= 5:
        statement = parts[3]
        if statement in ['DVA', 'DRA', 'DMPL', 'DFC_MD']:
            return False
    return True

def unify_csv_files():
    zip_dir = 'cvm_zip_files'
    output_dir = 'unified_cvm_data'
    temp_dir = 'temp_csv_files'

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract all CSV files from the downloaded zip files
    zip_files = [f for f in os.listdir(zip_dir) if f.endswith('.zip')]
    for zip_file in tqdm(zip_files, desc="Extracting zip files"):
        with zipfile.ZipFile(os.path.join(zip_dir, zip_file), 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

    # Move and filter CSV files
    csv_files = [f for f in os.listdir(temp_dir) if f.endswith('.csv')]
    for file in tqdm(csv_files, desc="Unifying CSV files"):
        if is_valid_file(file):
            src_path = os.path.join(temp_dir, file)
            dst_path = os.path.join(output_dir, file)
            if os.path.exists(dst_path):
                print(f"File {file} already exists in the unified folder. Skipping.")
            else:
                shutil.move(src_path, dst_path)

    # Clean up: remove temporary directory and cvm_zip_files
    shutil.rmtree(temp_dir)
    shutil.rmtree(zip_dir)

    print(f"Filtered CSV files have been unified into {output_dir}")
    return output_dir

def count_files_by_year(directory):
    year_count = Counter()
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            year = file.split('_')[-1].split('.')[0]
            year_count[year] += 1
    
    print("\nNumber of files by year:")
    for year, count in sorted(year_count.items()):
        print(f"{year}: {count}")

# Download the zip files
download_cvm_zip_files()

# Unify CSV files into a single folder
unified_dir = unify_csv_files()

# Count and print the number of files by year
count_files_by_year(unified_dir)