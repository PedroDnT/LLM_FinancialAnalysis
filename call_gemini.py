import os
from typing import Dict, Any, List
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from unidecode import unidecode
from langchain_core.prompts import ChatPromptTemplate
from utils import get_financial_statements_batch, get_company_name_by_cd_cvm
import google.generativeai as genai

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
    Analyze the provided financial data for the target year {target_year} and provide a concise prediction.FOllow the steps
    Steps:
    1. Review historical financial data and provide a concise analysis of the company's financial performance.
    2. Perform a ratio analysis and provide a concise analysis of the ratios.
    3. Given Your analysys in previous steps, make a conclusion on the best estimate of earnings direction and provide rationale based on your analysis.

    Response format:
        Panel A ||| [text from step 1]
        Panel B ||| [text from step 2]
        Panel C ||| [text from step 3]
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
    Financial data:
    Income Statements: {financial_data[income_statements]}
    Balance Sheets: {financial_data[balance_sheets]}
    Target year: {target_year}
    """
    return ChatPromptTemplate.from_template(template)

def gemini_pro_completion(prompt, model):
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {str(e)}")
        return None

def get_financial_prediction(financial_data: Dict[str, Any], n_years: int = None) -> Dict[int, Any]:
    try:
        print("Starting get_financial_prediction...")

        # Ensure the API key is set
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("API key for Google Generative AI is not set. Please set the 'GOOGLE_API_KEY' environment variable.")

        # Configure the API key and initialize the model
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')

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
            return {}
        
        if n_years is not None:
            target_years = target_years[-4:]
        
        def is_valid_ds_conta(item):
            return (isinstance(item.get('DS_CONTA'), str) and 
                    item['DS_CONTA'].strip() != '' and 
                    not all(pd.isna(value) or float(value) == 0 or float(value) == 0.0 for key, value in item.items() if key != 'DS_CONTA' and value is not None))

        def is_valid_year_value(item):
            return not all(pd.isna(value) or float(value) == 0 or float(value) == 0.0 for key, value in item.items() if key.startswith('20') and value is not None)

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

        def process_prompt(prompt, year):
            try:
                print(f"Sending prompt for year {year}...")
                return year, prompt
            except Exception as e:
                print(f"Error processing year {year}: {str(e)}")
                return year, None

        predictions = {}
        with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
            future_to_year = {executor.submit(process_prompt, prompt, target_years[i]): target_years[i] for i, prompt in enumerate(prompts)}
            for future in as_completed(future_to_year):
                year = future_to_year[future]
                try:
                    result_year, prompt = future.result()
                    if prompt is not None:
                        response = gemini_pro_completion(prompt, model)
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

def parse_financial_prediction(prediction_dict: Dict[int, str]) -> pd.DataFrame:
    parsed_data = []
    for year, text in prediction_dict.items():
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
        
        parsed_data.append({
            'Year': year,
            'Panel A': panel_a.replace('\n', ' '),
            'Panel B': panel_b.replace('\n', ' '),
            'Panel C': panel_c.replace('\n', ' '),
            'Prediction Direction': direction,
            'Magnitude': magnitude,
            'Confidence': confidence
        })
    
    return pd.DataFrame(parsed_data)