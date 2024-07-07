import os
import pandas as pd
import numpy as np
import unidecode

def read_files(file_type):
    files = os.listdir('unified_cvm_data')
    if file_type == 'BS':
        files = [file for file in files if 'dfp_cia_aberta_BPA_con_' in file or 'dfp_cia_aberta_BPP_con_' in file]
    elif file_type == 'IS':
        files = [file for file in files if 'dfp_cia_aberta_DRE_con_' in file]
    elif file_type == 'CF':
        files = [file for file in files if 'dfp_cia_aberta_DFC_MI_con_' in file]
    else:
        return None

    df = pd.concat([pd.read_csv(f'unified_cvm_data/{file}', sep=';', encoding='latin1', dtype={'VL_CONTA': str})
                    for file in files], ignore_index=True)

    df['VL_CONTA'] = pd.to_numeric(df['VL_CONTA'].str.replace(',', '.'), errors='coerce')
    df['CD_CVM'] = df['CD_CVM'].astype(int)

    columns_to_drop = ['CNPJ_CIA', 'VERSAO', 'DT_INI_EXERC']
    if 'ORDEM_EXERC' in df.columns:
        df = df[df['ORDEM_EXERC'] != 'PENÚLTIMO']
        columns_to_drop.append('ORDEM_EXERC')

    df = df.drop(columns=columns_to_drop, errors='ignore')

    if file_type == 'BS':
        df['GRUPO_DFP'] = df['GRUPO_DFP'].replace({
            'DF Consolidado - Balanço Patrimonial Ativo': 'BSA',
            'DF Consolidado - Balanço Patrimonial Passivo': 'BSP'
        })
    elif file_type == 'IS':
        df['GRUPO_DFP'] = df['GRUPO_DFP'].replace('DF Consolidado - Demonstração do Resultado', 'DRE - Con')
    elif file_type == 'CF':
        df['GRUPO_DFP'] = df['GRUPO_DFP'].replace('DF Consolidado - Demonstração do Fluxo de Caixa (Método Indireto)', 'FC-MI')

    return df

def read_files_ref():
    files = [file for file in os.listdir('unified_cvm_data') if 'dfp_cia_aberta_20' in file]
    df = pd.concat([pd.read_csv(f'unified_cvm_data/{file}', sep=';', encoding='latin1') for file in files], ignore_index=True)
    df = df.drop(columns=['CNPJ_CIA', 'VERSAO', 'ID_DOC', 'DT_RECEB', 'LINK_DOC'], errors='ignore')
    df['CD_CVM'] = df['CD_CVM'].astype(int)
    return df.drop_duplicates(subset='CD_CVM')

def aggregate_df(df):
    return df.groupby(['GRUPO_DFP','CD_CVM','DENOM_CIA','CD_CONTA', 'DS_CONTA','ST_CONTA_FIXA', 'DT_FIM_EXERC'], as_index=False)['VL_CONTA'].sum()

def remove_accents(text): 
     return unidecode.unidecode(text)
