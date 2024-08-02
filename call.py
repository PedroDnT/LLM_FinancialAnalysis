import openai
import pandas as pd
import re
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from unidecode import unidecode
from utils import get_financial_statements_batch, get_company_name_by_cd_cvm
import pandas as pd
import numpy as np
from typing import Dict, Any
import statistics
import os
from openai import OpenAI
import time
from tenacity import retry, stop_after_attempt, wait_fixed
from tqdm import tqdm  # Import tqdm for progress bar

# Constants for rate limiting
TPM_LIMIT = 450000  # Tokens per minute
BATCH_QUEUE_LIMIT = 1350000  # Batch queue limit in tokens
ESTIMATED_TOKENS_PER_REQUEST = 1000  # Estimate of tokens per request

def get_financial_data(CD_CVM_list: List[int]) -> Dict[str, Any]:
    """Fetches and returns financial data for the given CD_CVM list without including CD_CVM in the return JSON keys as a dictionary."""
    income, balance, _ = get_financial_statements_batch(CD_CVM_list)
    
    financial_data = {
        "income_statements": [],
        "balance_sheets": []
    }
    
    for code in CD_CVM_list:
        income_data = income[code].to_dict(orient='records')
        balance_data = balance[code].to_dict(orient='records')
        
        # Function to check if DS_CONTA is valid and not all values are NaN, 0, or 0.0
        def is_valid_ds_conta(item):
            return (isinstance(item.get('DS_CONTA'), str) and 
                    item['DS_CONTA'].strip() != '' and 
                    not all(pd.isna(value) or float(value) == 0 or float(value) == 0.0 for key, value in item.items() if key != 'DS_CONTA' and value is not None))

        # Filter and decode the 'DS_CONTA' column
        income_data = [item for item in income_data if is_valid_ds_conta(item)]
        balance_data = [item for item in balance_data if is_valid_ds_conta(item)]
        
        for item in income_data:
            item['DS_CONTA'] = unidecode(item['DS_CONTA'])
        for item in balance_data:
            item['DS_CONTA'] = unidecode(item['DS_CONTA'])
        
        financial_data["income_statements"].append(income_data)
        financial_data["balance_sheets"].append(balance_data)
    
    return financial_data

def create_prompt_template() -> ChatPromptTemplate:
    """Creates a prompt template for the financial prediction task."""
    template = """
    Analyze the provided financial data for the target year {target_year} and provide a concise prediction. Use the provided income statements and balance sheets data for your analysis.

    Financial data:
    Income Statements: {financial_data[income_statements]}
    Balance Sheets: {financial_data[balance_sheets]}
    Target year: {target_year}
    """
    return ChatPromptTemplate.from_template(template)

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def get_financial_prediction_with_retry(financial_data: Dict[str, Any], n_years: int) -> Dict[int, Any]:
    return get_financial_prediction(financial_data, n_years)

def get_financial_prediction(financial_data: Dict[str, Any], n_years: int = None) -> Dict[int, Any]:
    try:
        print("Starting get_financial_prediction...")

        # Debug print to check the structure of financial_data
        #print(f"Financial data structure: {financial_data.keys()}")
        if "income_statements" not in financial_data or not financial_data["income_statements"]:
            print("No income statements found in financial data.")
            return {}
        if not financial_data["income_statements"][0]:
            print("Income statements list is empty.")
            return {}

        available_years = sorted([int(year.split('-')[0]) for year in financial_data["income_statements"][0][0].keys() if year.startswith('20')])
        
        target_years = []
        for year in reversed(available_years):
            if year - 5 in available_years:
                target_years.append(year)
            else:
                print(f"Skipping year {year} due to insufficient historical data.")
        target_years.reverse()
        
        if not target_years:
            #print("Not enough historical data for prediction. At least 6 years of data are required.")
            return {}
        
        if n_years is not None:
            target_years = target_years[-4:]
        
        #print(f"Target years determined: {target_years}")

        # Function to check if DS_CONTA is valid and not all values are NaN, 0, or 0.0
        def is_valid_ds_conta(item):
            return (isinstance(item.get('DS_CONTA'), str) and 
                    item['DS_CONTA'].strip() != '' and 
                    not all(pd.isna(value) or float(value) == 0 or float(value) == 0.0 for key, value in item.items() if key != 'DS_CONTA' and value is not None))

        # Function to check if year values are valid and not all values are NaN, 0, or 0.0
        def is_valid_year_value(item):
            return not all(pd.isna(value) or float(value) == 0 or float(value) == 0.0 for key, value in item.items() if key.startswith('20') and value is not None)

        # Clean the columns of years
        def clean_year_columns(financial_data):
            for key, value in financial_data.items():
                for statement in value:
                    for item in statement:
                        for year in list(item.keys()):
                            if year.startswith('20') and not is_valid_year_value({year: item[year]}):
                                del item[year]
            return financial_data

        prompts = []
        for year in target_years:
            
            prompt_template = create_prompt_template()
            data_up_to = year - 1
            data_from = year - 4
            filtered_financial_data = {
                key: [
                    [{k: v for k, v in item.items() if k == 'DS_CONTA' or (k.startswith('20') and data_from <= int(k.split('-')[0]) <= data_up_to)}
                     for item in statement if is_valid_ds_conta(item)]
                    for statement in value
                ]
                for key, value in financial_data.items()
            }
            prompt = prompt_template.format(financial_data=filtered_financial_data, target_year=year)
            prompts.append(prompt)
        
        #print("Prompts created.")

        openai_api = ChatOpenAI(model="gpt-4o-mini", temperature=1)
        
        def process_prompt(prompt, year):
            try:
                print(f"Sending prompt for year {year}...")
                response = openai_api.generate([
                    [
                        {"role": "system", "content": """Analyze the provided financial statements and estimate earnings for the target year {target_year}.
                        Use the past 6 years of financial data {financial_data}.

                        Steps:
                        1. Review historical revenue and profit trends.
                        2. Perform a ratio analysis.
                        3. Given tour analysys in previous steps, think step by step to gather the best estimate of earnings direction and provide rationale based on your analysis.

                        Response format:
                        Panel A ||| [text]
                        Panel B ||| [text]
                        Panel C ||| [text]
                        Direction ||| [1/-1]
                        Magnitude ||| [large/moderate/small]
                        Confidence ||| [0.00 to 1.00]

                        Guidelines:
                        - Be precise and concise.
                        - Use 1 for increase, -1 for decrease.
                        - Use large, moderate, or small for magnitude.
                        - Provide a confidence score between 0.00 and 1.00.
                        - Do not include Direction, Magnitude, or Confidence in Panel C.
                        - Separate sections with '|||' delimiter.
                        - Do not define any formula or ratios on response.
                        """},
                        {"role": "user", "content": prompt}
                    ]
                ], logprobs=True)
                return year, response
            except Exception as e:
                print(f"Error processing year {year}: {str(e)}")
                return year, None

        predictions = {}
        with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
            future_to_year = {executor.submit(process_prompt, prompt, target_years[i]): target_years[i] for i, prompt in enumerate(prompts)}
            for future in as_completed(future_to_year):
                year = future_to_year[future]
                try:
                    result_year, response = future.result()
                    if response is not None:
                        predictions[result_year] = response
                except Exception as e:
                    print(f"Error processing future for year {year}: {str(e)}")
        
        print("Predictions received.")
        return predictions
    except Exception as e:
        print(f"An error occurred in get_financial_prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}

def calculate_confidence(logprobs):
    if not logprobs:
        return np.nan
    flat_logprobs = [logprob for sublist in logprobs for logprob in sublist]
    avg_logprob = sum(flat_logprobs) / len(flat_logprobs)
    return avg_logprob

def calculate_median_logprob(logprobs):
    if not logprobs:
        return np.nan
    flat_logprobs = [logprob for sublist in logprobs for logprob in sublist]
    median_logprob = statistics.median(flat_logprobs)
    return median_logprob

def calculate_std_logprob(logprobs):
    if not logprobs:
        return np.nan
    flat_logprobs = [logprob for sublist in logprobs for logprob in sublist]
    std_logprob = statistics.stdev(flat_logprobs)
    return std_logprob

def parse_financial_prediction(prediction_dict: Dict[int, Any], cd_cvm: int) -> pd.DataFrame:
    parsed_data = []
    for year, llm_result in prediction_dict.items():
        # Extract the generation text (OpenAI)
        generation = llm_result.generations[0][0]
        text = generation.text
        completion_tokens = llm_result.llm_output['token_usage']['completion_tokens']
        prompt_tokens = llm_result.llm_output['token_usage']['prompt_tokens']
        model_name = llm_result.llm_output['model_name']
        logprobs = generation.generation_info['logprobs']['content']
        logprob_values = [token_info['logprob'] for token_info in logprobs]

        # Extract the panels
        panel_a = text.split('Panel A |||')[1].split('Panel B |||')[0].strip()
        panel_b = text.split('Panel B |||')[1].split('Panel C |||')[0].strip()
        panel_c = text.split('Panel C |||')[1].split('Direction |||')[0].strip()
        
        # Extract direction, magnitude, and confidence
        direction = text.split('Direction |||')[1].split('Magnitude |||')[0].strip()
        magnitude = text.split('Magnitude |||')[1].split('Confidence |||')[0].strip()
        confidence_str = text.split('Confidence |||')[1].strip()
        
        # Clean up confidence string and convert to float
        confidence_str = confidence_str.split('\n')[0].strip('[]')
        try:
            confidence = float(confidence_str)
        except ValueError:
            print(f"Warning: Could not convert confidence to float for year {year}. Using NaN.")
            confidence = np.nan
        
        # Calculate logprob metrics if available
        avg_logprob = calculate_confidence([logprob_values]) if logprob_values else np.nan
        median_logprob = calculate_median_logprob([logprob_values]) if logprob_values else np.nan
        std_logprob = calculate_std_logprob([logprob_values]) if logprob_values else np.nan
        
        # Create the Year_CD_CVM column
        year_cd_cvm = f"{year}_{cd_cvm}"
        
        parsed_data.append({
            'Year': year,
            'CD_CVM': cd_cvm,
            'Year_CD_CVM': year_cd_cvm,
            'Panel A': panel_a.replace('\n', ' '),
            'Panel B': panel_b.replace('\n', ' '),
            'Panel C': panel_c.replace('\n', ' '),
            'Prediction Direction': direction,
            'Magnitude': magnitude,
            'Confidence': confidence,
            'Completion Tokens': completion_tokens,
            'Prompt Tokens': prompt_tokens,
            'Model Name': model_name,
            'Average Logprob': avg_logprob,
            'Median Logprob': median_logprob,
            'Std Logprob': std_logprob
        })
    
    return pd.DataFrame(parsed_data)

def get_financial_prediction_list(CD_CVM_list: List[int], n_years: int) -> pd.DataFrame:
    """
    Generates financial predictions for a list of CD_CVM codes and target years.
    
    Args:
    CD_CVM_list (List[int]): List of CD_CVM codes to process.
    n_years (int): Number of most recent years to predict for each CD_CVM code.
    
    Returns:
    pd.DataFrame: A DataFrame containing predictions for all CD_CVM codes and target years.
    """
    all_predictions = []
    
    # Initialize the progress bar
    with tqdm(total=len(CD_CVM_list), desc="Processing CD_CVM codes") as pbar:
        for cd_cvm in CD_CVM_list:
            print(f"Processing CD_CVM: {cd_cvm}")
            financial_data = get_financial_data([cd_cvm])
            
            try:
                predictions = get_financial_prediction_with_retry(financial_data, n_years)
            except Exception as e:
                print(f"Failed to get predictions for CD_CVM: {cd_cvm} after retries. Error: {str(e)}")
                pbar.update(1)  # Update the progress bar even if there's an error
                continue
            
            if predictions:
                df = parse_financial_prediction(predictions, cd_cvm)
                df['CD_CVM'] = cd_cvm
                all_predictions.append(df)
            else:
                print(f"No predictions generated for CD_CVM: {cd_cvm}")
            
            pbar.update(1)  # Update the progress bar after processing each CD_CVM
    
    if all_predictions:
        combined_df = pd.concat(all_predictions, ignore_index=True)
        return post_added_data(combined_df)
    else:
        return pd.DataFrame()

def post_added_data(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds an actual_earnings_direction column and a NAME column to the predictions DataFrame.
    
    Args:
    predictions_df (pd.DataFrame): DataFrame returned by get_financial_prediction_list
    
    Returns:
    pd.DataFrame: Updated DataFrame with actual_earnings_direction and NAME columns
    """
    def normalize_string(s):
        return unidecode(s).lower()

    def strip_markdown(text):
        # Remove bold and italic markers
        text = re.sub(r'\*\*|__', '', text)
        text = re.sub(r'\*|_', '', text)
        # Remove links
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        # Remove backticks
        text = re.sub(r'`', '', text)
        # Remove any remaining special characters
        text = re.sub(r'[#>~\-=|]', '', text)
        return text.strip()

    def get_actual_direction(row):
        cd_cvm = row['CD_CVM']
        year = row['Year']
        
        try:
            financial_data = get_financial_data([cd_cvm])
            if not financial_data or 'income_statements' not in financial_data or not financial_data['income_statements']:
                print(f"No financial data found for CD_CVM: {cd_cvm}")
                return np.nan
            
            income_statement = financial_data['income_statements'][0]
            
            earnings_metrics = [
                'Resultado Liquido das Operacoes Continuadas',
                'Lucro/Prejuizo Consolidado do Periodo',
                'Lucro/Prejuizo do Periodo'
            ]
            
            normalized_metrics = [normalize_string(metric) for metric in earnings_metrics]
            
            earnings_row = None
            for item in income_statement:
                normalized_ds_conta = normalize_string(item['DS_CONTA'])
                if normalized_ds_conta in normalized_metrics:
                    earnings_row = item
                    #print(f"Using earnings metric: {item['DS_CONTA']}")
                    break
            
            if earnings_row is None:
                #print(f"No suitable earnings metric found for CD_CVM: {cd_cvm}")
                #print(f"Available metrics: {[item['DS_CONTA'] for item in income_statement]}")
                return np.nan
            
            #print(f"Debug: Earnings row for CD_CVM {cd_cvm}: {earnings_row}")
            
            current_year_earnings = earnings_row.get(f'{year}-12-31')
            previous_year_earnings = earnings_row.get(f'{year-1}-12-31')
            
            #print(f"Debug: Current year earnings ({year}): {current_year_earnings}")
            #print(f"Debug: Previous year earnings ({year-1}): {previous_year_earnings}")
            
            if current_year_earnings is None or previous_year_earnings is None:
                print(f"Missing earnings data for CD_CVM: {cd_cvm}, Year: {year}")
                return np.nan
            try:
                current_year_earnings = float(current_year_earnings)
                previous_year_earnings = float(previous_year_earnings)
            except ValueError:
                print(f"Error converting earnings to float for CD_CVM: {cd_cvm}, Year: {year}")
                return np.nan
            return 1 if current_year_earnings > previous_year_earnings else -1
        except Exception as e:
            print(f"Error processing CD_CVM: {cd_cvm}, Year: {year}. Error: {str(e)}")
            return np.nan
    # Apply the function to each row
    predictions_df['actual_earnings_direction'] = predictions_df.apply(get_actual_direction, axis=1)
    # Add the NAME column
    predictions_df['NAME'] = predictions_df['CD_CVM'].apply(get_company_name_by_cd_cvm)
    # Strip markdown from Panel A, B, and C
    for panel in ['Panel A', 'Panel B', 'Panel C']:
        if panel in predictions_df.columns:
            predictions_df[panel] = predictions_df[panel].apply(strip_markdown)
    return predictions_df