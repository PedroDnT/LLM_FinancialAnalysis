import os
import json
from typing import Dict, Any, List, Tuple, Optional
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
from functools import lru_cache

@lru_cache(maxsize=None)
def get_financial_data(CD_CVM_list: Tuple[int, ...]) -> Dict[str, Any]:
    """Fetches and returns financial data for the given CD_CVM list."""
    income, balance, _ = get_financial_statements_batch(list(CD_CVM_list))
    
    financial_data = {
        "income_statements": [],
        "balance_sheets": []
    }
    
    for code in CD_CVM_list:
        income_data = income[code].to_dict(orient='records')
        balance_data = balance[code].to_dict(orient='records')
        
        # Decode the 'DS_CONTA' column
        for data in (income_data, balance_data):
            for item in data:
                item['DS_CONTA'] = unidecode(item['DS_CONTA'])
        
        financial_data["income_statements"].append(income_data)
        financial_data["balance_sheets"].append(balance_data)
    
    return financial_data

def create_prompt_template() -> ChatPromptTemplate:
    """Creates a prompt template for the financial prediction task."""
    template = """
    Analyze the provided financial data for the target year {target_year} and provide a concise prediction. Follow these instructions strictly:

    1. Do not include any introductory text or pleasantries.
    2. Start directly with the analysis sections as outlined below.
    3. Provide all sections in the exact order and format specified.
    4. Use at least 5 years of historical data prior to the target year for your analysis.
    5. Analyze both income statements and balance sheets in your prediction.
    6. Focus on predicting the 'Resultado Líquido das Operações Continuadas' (Net Income from Continuing Operations) as the main earnings metric.
    
    Your response must follow this exact structure:

    Panel A ||| [Trend Analysis: Analyze relevant trends over at least the past five years,.]
    Panel B ||| [Ratio Analysis: Calculate and analyze key financial ratios over at least the past five years, interpreting their implications for future earnings.]
    Panel C ||| [Rationale: Summarize your analyses and explain your prediction reasoning concisely, considering the long-term trends and focusing on 'Resultado Líquido das Operações Continuadas'.]
    Direction ||| [increase/decrease]
    Magnitude ||| [large/moderate/small]
    Confidence ||| [0.00 to 1.00]

    Additional guidelines:
    - Be precise, focused and cocise in your explanations.
    - For Magnitude, you must use exactly one of these words: large, moderate, or small. Do not skip this or use any other terms.
    - For Confidence, provide a single number between 0.00 and 1.00.
    - Do not include formulas or calculations in your response.
    - Use '|||' as a delimiter between section headers and content.
    - Ensure your analysis covers at least 5 years of historical data.
    - Return responses in English.
    - No need to define fomulas or calculations in your response. Just mention the ratio or the value by name.
    - When referring to earnings, always use 'Resultado Líquido das Operaçes Continuadas' as the key metric, but call it just earnings.

    Financial data: {financial_data}
    Target year: {target_year}
    """
    return ChatPromptTemplate.from_template(template)

def get_financial_prediction(financial_data: Dict[str, Any], n_years: Optional[int] = None) -> Dict[int, Any]:
    """
    Calls the prompt template and returns the entire response in a dictionary for a given CD_CVM.
    If n_years is not provided, it predicts for all available years with at least 5 years of historical data.
    """
    try:
        print("Starting get_financial_prediction...")

        available_years = sorted(set(int(year.split('-')[0]) for year in financial_data["income_statements"][0][0].keys() if year.startswith('20')))
        
        if n_years is None:
            target_years = [year for year in available_years if year - 5 in available_years]
        else:
            target_years = [year for year in reversed(available_years[-n_years:]) if year - 5 in available_years]
            target_years.reverse()
        
        if not target_years:
            print("Not enough historical data for prediction. At least 5 years of data are required.")
            return {}
        
        print(f"Target years determined: {target_years}")

        prompt_template = create_prompt_template()
        openai_api = ChatOpenAI(model="gpt-4o", temperature=1)
        
        predictions = {}
        for year in target_years:
            try:
                data_up_to = year - 1
                data_from = year - 6
                filtered_financial_data = {
                    key: [
                        [{k: v for k, v in item.items() if k == 'DS_CONTA' or (k.startswith('20') and data_from <= int(k.split('-')[0]) <= data_up_to)}
                         for item in statement]
                        for statement in value
                    ]
                    for key, value in financial_data.items()
                }
                prompt = prompt_template.format(financial_data=filtered_financial_data, target_year=year)
                
                print(f"Sending prompt for year {year}...")
                response = openai_api.generate([
                    [
                        {"role": "system", "content": "As a Brazilian experienced equity research analyst, your task is to analyze the provided financial statements and predict future earnings for the specified target period."},
                        {"role": "user", "content": prompt}
                    ]
                ])
                
                print(f"Response from OpenAI API for year {year}: {response}")
                predictions[year] = response
            except Exception as e:
                print(f"Error processing year {year}: {str(e)}")

        print("Predictions received.")
        return predictions
    except Exception as e:
        print(f"An error occurred in get_financial_prediction: {str(e)}")
        print(f"Financial data structure: {financial_data.keys()}")
        print(f"First item in income_statements: {financial_data['income_statements'][0][0].keys()}")
        return {}

def parse_financial_prediction(prediction_dict: Dict[int, Any]) -> pd.DataFrame:
    parsed_data = []
    for year, llm_result in prediction_dict.items():
        generation = llm_result.generations[0][0]
        text = generation.text
        
        # Extract panels and prediction using the new delimiter
        panels = re.split(r'Panel [A-C] \|\|\|', text)
        panel_a = panels[1].strip() if len(panels) > 1 else ''
        panel_b = panels[2].strip() if len(panels) > 2 else ''
        panel_c = panels[3].strip() if len(panels) > 3 else ''
        
        # Extract direction, magnitude, and confidence
        direction_match = re.search(r'Direction \|\|\| (\w+)', text, re.IGNORECASE)
        direction = 1 if direction_match and 'increase' in direction_match.group(1).lower() else -1
        
        magnitude_match = re.search(r'Magnitude \|\|\| (\w+)', text, re.IGNORECASE)
        if magnitude_match:
            magnitude = magnitude_match.group(1).lower()
            if magnitude not in ['large', 'moderate', 'small']:
                print(f"Warning: Unexpected magnitude value '{magnitude}' for year {year}. Setting to 'moderate'.")
                magnitude = 'moderate'
        else:
            print(f"Warning: No magnitude found for year {year}. Setting to 'moderate'.")
            magnitude = 'moderate'
        
        confidence_match = re.search(r'Confidence \|\|\| (\d+\.\d+)', text, re.IGNORECASE)
        try:
            confidence = float(confidence_match.group(1)) if confidence_match else 0.0
            confidence = round(max(0.00, min(1.00, confidence)), 2)  # Ensure it's between 0.00 and 1.00
        except (ValueError, AttributeError):
            confidence = 0.0
        
        # Extract token usage and model information
        completion_tokens = llm_result.llm_output['token_usage']['completion_tokens']
        prompt_tokens = llm_result.llm_output['token_usage']['prompt_tokens']
        model_name = llm_result.llm_output['model_name']
        
        parsed_data.append({
            'Year': year,
            'Panel A': panel_a.replace('\n', ' '),
            'Panel B': panel_b.replace('\n', ' '),
            'Panel C': panel_c.replace('\n', ' '),
            'Prediction Direction': direction,
            'Magnitude': magnitude,
            'Confidence': confidence,
            'Completion Tokens': completion_tokens,
            'Prompt Tokens': prompt_tokens,
            'Model Name': model_name
        })
    
    return pd.DataFrame(parsed_data)

def get_financial_prediction_list(CD_CVM_list: List[int], n_years: Optional[int] = None) -> pd.DataFrame:
    """Generates financial predictions for a list of CD_CVM codes and target years."""
    all_predictions = []
    
    for cd_cvm in CD_CVM_list:
        print(f"Processing CD_CVM: {cd_cvm}")
        financial_data = get_financial_data((cd_cvm,))
        predictions = get_financial_prediction(financial_data, n_years)
        
        if predictions:
            df = parse_financial_prediction(predictions)
            df['CD_CVM'] = cd_cvm
            all_predictions.append(df)
        else:
            print(f"No predictions generated for CD_CVM: {cd_cvm}")
    
    return pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()

def post_added_data(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Adds an actual_earnings_direction column and a NAME column to the predictions DataFrame."""
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
            financial_data = get_financial_data((cd_cvm,))
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