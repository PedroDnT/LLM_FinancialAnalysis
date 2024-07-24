import os
import json
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import pandas as pd
import numpy as np
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.callbacks import get_openai_callback
from utils import *
from unidecode import unidecode
import re

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
        
        # Decode the 'DS_CONTA' column
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
    Analyze the provided financial data for the target year {target_year} and provide a concise prediction. Follow these instructions strictly:

    1. Use at least 5 years of historical data prior to the target year.
    2. Focus on 'Resultado Líquido das Operações Continuadas' (Net Income from Continuing Operations) as the main earnings metric.
    
    Your response must follow this exact structure:

    Panel A ||| [Trend Analysis: Analyze trends over the past five years, focusing on 'Resultado Líquido das Operações Continuadas'. Describe the overall trend without listing specific values.]
    Panel B ||| [Ratio Analysis: Analyze key financial ratios over the past five years. Describe trends and changes in ratios without listing specific values for each year.]
    Panel C ||| [Rationale: Summarize your analyses and explain your prediction reasoning. Focus on qualitative insights rather than quantitative data.]
    Direction ||| [increase/decrease]
    Magnitude ||| [large/moderate/small]
    Confidence ||| [0.00 to 1.00]

    Additional guidelines:
    - Be precise and concise.
    - For Magnitude, use one of: large, moderate, or small.
    - For Confidence, provide a number between 0.00 and 1.00.
    - Do not include formulas, calculations, or lists of values.
    - Describe trends and changes without mentioning specific numerical values.
    - Use comparative terms (e.g., "increased", "decreased", "remained stable") instead of listing values.
    - Use '|||' as a delimiter between sections.
    - Return responses in English.

    Financial data: {financial_data}
    Target year: {target_year}
    """
    return ChatPromptTemplate.from_template(template)

def get_financial_prediction(financial_data: Dict[str, Any], n_years: int) -> Dict[int, Any]:
    """Calls the prompt template and returns the entire response in a dictionary for a given CD_CVM."""
    try:
        print("Starting get_financial_prediction...")

        # Determine the available years based on the data
        available_years = sorted([int(year.split('-')[0]) for year in financial_data["income_statements"][0][0].keys() if year.startswith('20')])
        
        # Select the last n_years for prediction, ensuring at least 5 years of data for each prediction
        target_years = []
        for year in reversed(available_years[-n_years:]):
            if year - 5 in available_years:
                target_years.append(year)
            else:
                print(f"Skipping year {year} due to insufficient historical data.")
        target_years.reverse()  # Reverse to maintain chronological order
        
        if not target_years:
            print("Not enough historical data for prediction. At least 5 years of data are required.")
            return {}
        
        print(f"Target years determined: {target_years}")

        # Create a prompt for each target year
        prompts = []
        for year in target_years:
            prompt_template = create_prompt_template()
            # Use data up to the year before the target year, ensuring at least 5 years of data
            data_up_to = year - 1
            data_from = min(year - 6, available_years[0])  # Ensure we use at least 5 years of data
            filtered_financial_data = {
                key: [
                    [{k: v for k, v in item.items() if k == 'DS_CONTA' or (k.startswith('20') and data_from <= int(k.split('-')[0]) <= data_up_to)}
                     for item in statement
                ]
                for statement in value
            ]
            for key, value in financial_data.items()
        }
        prompt = prompt_template.format(financial_data=filtered_financial_data, target_year=year)
        prompts.append(prompt)
        
        print("Prompts created.")

        # Initialize the OpenAI API
        openai_api = ChatOpenAI(model="gpt-4o", temperature=1)
        
        # Get the predictions from the OpenAI API for each target year
        predictions = {}
        for i, prompt in enumerate(prompts):
            try:
                print(f"Sending prompt for year {target_years[i]}...")
                response = openai_api.generate([
                    [
                        {"role": "system", "content": "As a Brazilian experienced equity research analyst, your task is to analyze the provided financial statements and predict future earnings for the specified target period."},
                        {"role": "user", "content": prompt}
                    ]
                ])
                
                # Print the response for debugging
                print(f"Response from OpenAI API for year {target_years[i]}: {response}")
                
                # Store the entire response in the dictionary
                predictions[target_years[i]] = response
            except Exception as e:
                print(f"Error processing year {target_years[i]}: {str(e)}")
                continue

        print("Predictions received.")
        return predictions
    except Exception as e:
        print(f"An error occurred in get_financial_prediction: {str(e)}")
        print(f"Financial data structure: {financial_data.keys()}")
        print(f"First item in income_statements: {financial_data['income_statements'][0][0].keys()}")
        return {}

def parse_financial_prediction(prediction_dict: Dict[int, Any]) -> pd.DataFrame:
    try:
        parsed_data = []
        for year, prediction in prediction_dict.items():
            if isinstance(prediction, dict) and 'generations' in prediction:
                text = prediction['generations'][0][0].text
            elif isinstance(prediction, list) and prediction and isinstance(prediction[0], dict) and 'text' in prediction[0]:
                text = prediction[0]['text']
            elif hasattr(prediction, 'generations') and prediction.generations:
                text = prediction.generations[0].text
            else:
                print(f"Unexpected prediction format for year {year}")
                continue

            panels = text.split('|||')
            if len(panels) >= 6:
                parsed_data.append({
                    'Year': year,
                    'Panel A': panels[1].strip(),
                    'Panel B': panels[2].strip(),
                    'Panel C': panels[3].strip(),
                    'Direction': panels[4].strip(),
                    'Magnitude': panels[5].strip(),
                    'Confidence': float(panels[6].strip()) if len(panels) > 6 else None
                })
            else:
                print(f"Unexpected number of panels in prediction for year {year}")

        return pd.DataFrame(parsed_data)
    except Exception as e:
        print(f"An error occurred while parsing financial predictions: {str(e)}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error

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
    
    for cd_cvm in CD_CVM_list:
        print(f"Processing CD_CVM: {cd_cvm}")
        financial_data = get_financial_data([cd_cvm])
        predictions = get_financial_prediction(financial_data, n_years)
        
        if predictions:
            df = parse_financial_prediction(predictions)
            df['CD_CVM'] = cd_cvm
            all_predictions.append(df)
        else:
            print(f"No predictions generated for CD_CVM: {cd_cvm}")
    
    if all_predictions:
        return pd.concat(all_predictions, ignore_index=True)
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
            
            print(f"Debug: Income statement structure for CD_CVM {cd_cvm}:")
            print(f"Type: {type(income_statement)}")
            print(f"Number of items: {len(income_statement)}")
            print(f"Sample content: {income_statement[:2]}")
            
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
                    print(f"Using earnings metric: {item['DS_CONTA']}")
                    break
            
            if earnings_row is None:
                print(f"No suitable earnings metric found for CD_CVM: {cd_cvm}")
                print(f"Available metrics: {[item['DS_CONTA'] for item in income_statement]}")
                return np.nan
            
            print(f"Debug: Earnings row for CD_CVM {cd_cvm}: {earnings_row}")
            
            current_year_earnings = earnings_row.get(f'{year}-12-31')
            previous_year_earnings = earnings_row.get(f'{year-1}-12-31')
            
            print(f"Debug: Current year earnings ({year}): {current_year_earnings}")
            print(f"Debug: Previous year earnings ({year-1}): {previous_year_earnings}")
            
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