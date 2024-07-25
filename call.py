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

    1. Do not include any introductory text or pleasantries.
    2. Start directly with the analysis sections as outlined below.
    3. Provide all sections in the exact order and format specified.
    4. Use at least 5 years of historical data prior to the target year for your analysis.
    5. Analyze both income statements and balance sheets in your prediction.
    6. Focus on predicting the 'Resultado Líquido das Operações Continuadas' (Net Income from Continuing Operations) as the main earnings metric.
    
    Your response must follow this exact structure, with each section on a new line:

    Panel A ||| [Trend Analysis]
    Panel B ||| [Ratio Analysis]
    Panel C ||| [Rationale]
    Direction ||| [increase/decrease]
    Magnitude ||| [large/moderate/small]
    Confidence ||| [0.00 to 1.00]

    Additional guidelines:
    - Be precise, focused and concise in your explanations.
    - For Magnitude, you must use exactly one of these words: large, moderate, or small.
    - For Confidence, provide a single number between 0.00 and 1.00.
    - Do not include the Direction, Magnitude, or Confidence information in the Panel C section.
    - Ensure each section is clearly separated by the '|||' delimiter.
    - Do not skip any sections or change their order.

    Financial data: {financial_data}
    Target year: {target_year}
    """
    return ChatPromptTemplate.from_template(template)

def get_financial_prediction(financial_data: Dict[str, Any], n_years: int) -> Dict[int, Any]:
    try:
        print("Starting get_financial_prediction...")

        available_years = sorted([int(year.split('-')[0]) for year in financial_data["income_statements"][0][0].keys() if year.startswith('20')])
        
        target_years = []
        for year in reversed(available_years[-n_years:]):
            if year - 5 in available_years:
                target_years.append(year)
            else:
                print(f"Skipping year {year} due to insufficient historical data.")
        target_years.reverse()
        
        if not target_years:
            print("Not enough historical data for prediction. At least 5 years of data are required.")
            return {}
        
        print(f"Target years determined: {target_years}")

        prompts = []
        for year in target_years:
            prompt_template = create_prompt_template()
            data_up_to = year - 1
            data_from = min(year - 6, available_years[0])
            filtered_financial_data = {
                key: [
                    [{k: v for k, v in item.items() if k == 'DS_CONTA' or (k.startswith('20') and data_from <= int(k.split('-')[0]) <= data_up_to)}
                     for item in statement]
                    for statement in value
                ]
                for key, value in financial_data.items()
            }
            prompt = prompt_template.format(financial_data=filtered_financial_data, target_year=year)
            prompts.append(prompt)
        
        print("Prompts created.")

        openai_api = ChatOpenAI(model="gpt-4o", temperature=1)
        
        def process_prompt(prompt, year):
            try:
                print(f"Sending prompt for year {year}...")
                response = openai_api.generate([
                    [
                        {"role": "system", "content": "As a Brazilian experienced equity research analyst, your task is to analyze the provided financial statements and predict future earnings for the specified target period."},
                        {"role": "user", "content": prompt}
                    ]
                ])
                print(f"Response from OpenAI API for year {year}: {response}")
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
        print(f"Financial data structure: {financial_data.keys()}")
        print(f"First item in income_statements: {financial_data['income_statements'][0][0].keys()}")
        return {}


def parse_financial_prediction(prediction_dict: Dict[int, Any], cd_cvm: int) -> pd.DataFrame:
    parsed_data = []
    for year, llm_result in prediction_dict.items():
        # Extract the generation text
        generation = llm_result.generations[0][0]
        text = generation.text

        # Extract panels and prediction using the new delimiter
        panels = re.split(r'Panel [A-C] \|\|\|', text)
        panel_a = panels[1].strip() if len(panels) > 1 else ''
        panel_b = panels[2].strip() if len(panels) > 2 else ''
        panel_c = panels[3].strip() if len(panels) > 3 else ''
        
        # Extract direction, magnitude, and confidence
        direction_match = re.search(r'Direction\s*\|\|\|\s*(\w+)', text, re.IGNORECASE)
        direction = 1 if direction_match and 'increase' in direction_match.group(1).lower() else -1
        
        magnitude_match = re.search(r'Magnitude\s*\|\|\|\s*(\w+)', text, re.IGNORECASE)
        magnitude = magnitude_match.group(1).lower() if magnitude_match else 'moderate'
        
        confidence_match = re.search(r'Confidence\s*\|\|\|\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
        confidence = float(confidence_match.group(1)) if confidence_match else 0.0
        confidence = round(max(0.00, min(1.00, confidence)), 2)  # Ensure it's between 0.00 and 1.00
        
        # If confidence is 0, try to extract it from the text
        if confidence == 0.0:
            confidence_alt_match = re.search(r'Confidence:?\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
            if confidence_alt_match:
                confidence = float(confidence_alt_match.group(1))
                confidence = round(max(0.00, min(1.00, confidence)), 2)
        
        # Extract token usage and model information
        completion_tokens = llm_result.llm_output['token_usage']['completion_tokens']
        prompt_tokens = llm_result.llm_output['token_usage']['prompt_tokens']
        
        # Extract full model name including version
        model_name = generation.message.response_metadata.get('model_name', 'Unknown')
        
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
            'Model Name': model_name
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
    
    for cd_cvm in CD_CVM_list:
        print(f"Processing CD_CVM: {cd_cvm}")
        financial_data = get_financial_data([cd_cvm])
        predictions = get_financial_prediction(financial_data, n_years)
        
        if predictions:
            df = parse_financial_prediction(predictions, cd_cvm)
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