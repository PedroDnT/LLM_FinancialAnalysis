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
from langchain_groq import ChatGroq
from ratelimit import limits, sleep_and_retry
import json
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Dict, List, Any
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, validator

# Constants for rate limiting
TPM_LIMIT = 450000  # Tokens per minute
BATCH_QUEUE_LIMIT = 1350000  # Batch queue limit in tokens
ESTIMATED_TOKENS_PER_REQUEST = 1000  # Estimate of tokens per request

# Define the rate limit
CALLS_PER_MINUTE = 16  # 1000 RPM / 60 seconds
CALLS_PER_DAY = 14000
TOKENS_PER_MINUTE = 130000
ESTIMATED_TOKENS_PER_CALL = 15000  # 10000 input + 5000 output

@sleep_and_retry
@limits(calls=CALLS_PER_MINUTE, period=60)
def rate_limited_api_call(cd_cvm, financial_data, n_years):
    return get_financial_prediction(financial_data, n_years)

def get_financial_data(CD_CVM_list: List[int]) -> Dict[str, Any]:
    """Fetches and returns financial data for the given CD_CVM list without including CD_CVM in the return JSON keys as a dictionary."""
    income, balance, _ = get_financial_statements_batch(CD_CVM_list)
    
    financial_data = {
        "income_statements": [],
        "balance_sheets": []
    }
    
    for code in CD_CVM_list:
        if code not in income or code not in balance:
            print(f"Warning: Financial data not found for CD_CVM {code}. Skipping.")
            continue
        
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

system_prompt = """
            You are a Brazilian financial analyst specializing in analyzing financial statements and forecasting earnings direction. Your task is to analyze financial statements, specifically using the balance sheet and income statement, to predict future returns. Use your expertise to identify the most relevant metrics and indices for this analysis and base your predictions solely on this data. Apply a chain of thought approach to carefully reason through each step of your analysis. Structure your response in the following three panels:

            Panel A: Trend Analysis
            Identify Key Trends: Begin by identifying the most significant trends in the financial statements.
            Focus on Impact: Analyze how these trends are likely to impact future earnings, considering both positive and negative implications.
            Document Observations: Clearly document your observations and the rationale behind selecting these trends, linking them directly to potential earnings outcomes for the target year.
            
            Panel B: Ratio Analysis
            Select Key Ratios: Choose and calculate the financial ratios that are most relevant for predicting future earnings.
            Interpret Ratios: Carefully interpret these ratios in the context of the company’s overall financial health and potential for future earnings growth.
            Explain the Significance: Provide a detailed explanation of how these ratios influence your earnings predictions, supported by your calculations and reasoning.
            
            Panel C: Integrated Analysis and Summary
            Combine Insights: Integrate the insights from your trend and ratio analyses to form a comprehensive view of the company’s financial outlook.
            Evaluate Overall Position: Assess the company’s overall financial position, considering both strengths and weaknesses.
            Predict Future Returns: Offer an informed prediction of expected returns, fully considering the factors analyzed in the previous panels. Ensure your prediction is logical, coherent, and supported by the analysis provided.

            Response format:
                Panel A ||| [text from Panel A analysis]
                Panel B ||| [text from Panel B analysis]
                Panel C ||| [text from Panel C analysis]
                Direction ||| [1/-1]
                Magnitude ||| [large/moderate/small]
                Confidence ||| [0.00 to 1.00]

                Guidelines:
                - Remember to follow the chain of thought process by logically connecting each observation and calculation to your final prediction. Provide your response in a clear, structured format as outlined below.
                - Do not include introductory text, titles, subtitles or disclaimers
                - DO not include Panel Name in text
                - The data is in Portuguese and data follow the standard financial statements format by Comissao de Valores Mobiliarios (CVM). Answer in English. 
                - Use 1 for increase, -1 for decrease.
                - Use large, moderate, or small for magnitude.
                - Provide a confidence score between 0.00 and 1.00.
                - Do not include Direction, Magnitude, or Confidence in Panel C.
                - Separate sections with '|||' delimiter.
                - Do not define any formula or ratios on response.
                - No need to use full name of accounts or define calculations.
    """

from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser

def create_prompt_template() -> ChatPromptTemplate:
    parser = PydanticOutputParser(pydantic_object=FinancialAnalysis)
    
    template = """
    Analyze the provided financial data for the target year {target_year} perform a comprehensive analysis, divided into three panels, 
    integrating all analysis into a single prediction and provide an summary of the analysis and prediction on earnings direction. 
    Use the provided income statements and balance sheets data for your analysis. 

    {format_instructions}

    Financial data: {financial_data}
    Target year: {target_year}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    return prompt.partial(format_instructions=parser.get_format_instructions())

class FinancialAnalysis(BaseModel):
    panel_a: str = Field(description="Trend analysis of financial statements and its impact on earnings")
    panel_b: str = Field(description="Ratio analysis of financial ratios and its impact on earnings")
    panel_c: str = Field(description="Integrated analysis and summary of predictions")
    direction: int = Field(description="Predicted earnings direction (1 for increase, -1 for decrease)")
    magnitude: str = Field(description="Predicted magnitude of change (large, moderate, or small)")
    confidence: float = Field(description="Confidence score between 0.00 and 1.00")

    @validator("direction")
    def direction_must_be_valid(cls, v):
        if v not in [1, -1]:
            raise ValueError("Direction must be either 1 (increase) or -1 (decrease)")
        return v

    @validator("magnitude")
    def magnitude_must_be_valid(cls, v):
        if v not in ["large", "moderate", "small"]:
            raise ValueError("Magnitude must be either 'large', 'moderate', or 'small'")
        return v

    @validator("confidence")
    def confidence_must_be_between_0_and_1(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        return v

def is_valid_ds_conta(item):
    return (isinstance(item.get('DS_CONTA'), str) and 
            item['DS_CONTA'].strip() != '' and 
            not all(pd.isna(value) or float(value) == 0 or float(value) == 0.0 
                    for key, value in item.items() 
                    if key != 'DS_CONTA' and value is not None))

def is_valid_year_value(item):
    return not all(pd.isna(value) or float(value) == 0 or float(value) == 0.0 
                   for key, value in item.items() 
                   if key.startswith('20') and value is not None)

def clean_year_columns(financial_data):
    for key, value in financial_data.items():
        for statement in value:
            for item in statement:
                for year in list(item.keys()):
                    if year.startswith('20') and not is_valid_year_value({year: item[year]}):
                        del item[year]
    return financial_data

def process_prompt_groq(prompt, year):
    try:
        print(f"Sending prompt for year {year}...")
        messages = ["system", system_prompt, "human", prompt]
        response=llm.invoke(messages)
        return year, response
    except Exception as e:
        print(f"Error processing year {year}: {str(e)}")
        return year, None

def get_financial_prediction(financial_data: Dict[str, Any], n_years: int = 3) -> Dict[int, Dict[str, Any]]:
    try:
        print("Starting get_financial_prediction...")

        if isinstance(financial_data, str):
            financial_data = json.loads(financial_data)

        if "income_statements" not in financial_data or not financial_data["income_statements"]:
            print("No income statements found in financial data.")
            return {}
        if not financial_data["income_statements"][0]:
            print("Income statements list is empty.")
            return {}

        available_years = sorted([int(year.split('-')[0]) for year in financial_data["income_statements"][0][0].keys() if year.startswith('20')])
        
        target_years = [year for year in reversed(available_years) if year - 5 in available_years]
        target_years.reverse()
        
        if not target_years:
            return {}
        
        if n_years is not None:
            target_years = target_years[-n_years:]

        model_name = "mixtral-8x7b-32768"
        chat = ChatGroq(model_name=model_name, temperature=1)
        parser = PydanticOutputParser(pydantic_object=FinancialAnalysis)

        predictions = {}
        for year in target_years:
            prompt_template = create_prompt_template()
            data_up_to = year - 1
            data_from = year - 5
            filtered_financial_data = {
                key: [
                    [{k: v for k, v in item.items() if k == 'DS_CONTA' or (k.startswith('20') and data_from <= int(k.split('-')[0]) <= data_up_to)}
                     for item in statement if is_valid_ds_conta(item)]
                    for statement in value
                ]
                for key, value in financial_data.items()
            }
            
            filtered_financial_data_json = json.dumps(filtered_financial_data)
            
            human_message = HumanMessage(content=prompt_template.format(
                financial_data=filtered_financial_data_json, 
                target_year=year,
                format_instructions=parser.get_format_instructions()
            ))
            system_message = SystemMessage(content=system_prompt)

            messages = [system_message, human_message]
            
            response = chat.generate([messages])
            
            # Extract token usage from the LLMResult
            token_usage = response.llm_output.get('token_usage', {})
            
            parsed_response = parser.parse(response.generations[0][0].text)
            predictions[year] = {
                'analysis': parsed_response,
                'token_usage': token_usage,
                'model': model_name
            }

        print("Predictions received.")
        return predictions
    except Exception as e:
        print(f"An error occurred in get_financial_prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}

def calculate_std_logprob(logprobs):
    if not logprobs:
        return np.nan
    flat_logprobs = [logprob for sublist in logprobs for logprob in sublist]
    std_logprob = statistics.stdev(flat_logprobs)
    return std_logprob

def parse_financial_prediction(prediction_dict: Dict[int, Dict[str, Any]], cd_cvm: int) -> pd.DataFrame:
    parsed_data = []
    for year, prediction_data in prediction_dict.items():
        analysis = prediction_data['analysis']
        token_usage = prediction_data['token_usage']
        model = prediction_data['model']
        year_cd_cvm = f"{year}_{cd_cvm}"
        
        parsed_data.append({
            'Year': year,
            'CD_CVM': cd_cvm,
            'Year_CD_CVM': year_cd_cvm,
            'Panel A': analysis.panel_a.replace('\n', ' '),
            'Panel B': analysis.panel_b.replace('\n', ' '),
            'Panel C': analysis.panel_c.replace('\n', ' '),
            'Prediction Direction': analysis.direction,
            'Magnitude': analysis.magnitude,
            'Confidence': analysis.confidence,
            'Prompt Tokens': token_usage['prompt_tokens'],
            'Completion Tokens': token_usage['completion_tokens'],
            'Total Tokens': token_usage['total_tokens'],
            'Model Name': model  # Add the model to the parsed data
        })
    
    return pd.DataFrame(parsed_data)

def get_financial_prediction_list_groq(CD_CVM_list: List[int], n_years: int=None) -> pd.DataFrame:
    """
    Generates financial predictions for a list of CD_CVM codes and target years.
    
    Args:
    CD_CVM_list (List[int]): List of CD_CVM codes to process.
    n_years (int): Number of most recent years to predict for each CD_CVM code.
    
    Returns:
    pd.DataFrame: A DataFrame containing predictions for all CD_CVM codes and target years.
    """
    all_predictions = []
    daily_call_count = 0
    daily_token_count = 0
    start_time = time.time()

    for cd_cvm in CD_CVM_list:
        print(f"Processing CD_CVM: {cd_cvm}")
        try:
            # Check daily limits
            if daily_call_count >= CALLS_PER_DAY:
                print("Daily call limit reached. Stopping processing.")
                break
            if daily_token_count + ESTIMATED_TOKENS_PER_CALL > TOKENS_PER_MINUTE * 1440:  # 1440 minutes in a day
                print("Daily token limit reached. Stopping processing.")
                break

            financial_data = get_financial_data([cd_cvm])
            
            if not financial_data:
                print(f"No financial data found for CD_CVM: {cd_cvm}. Skipping.")
                continue
            
            # Make the rate-limited API call
            predictions = rate_limited_api_call(cd_cvm, financial_data, n_years)
            
            # Update counters
            daily_call_count += 1
            daily_token_count += ESTIMATED_TOKENS_PER_CALL

            if predictions:
                df = parse_financial_prediction(predictions, cd_cvm)
                all_predictions.append(df)
            else:
                print(f"No predictions generated for CD_CVM: {cd_cvm}")

            # Check if we need to pause for token limit
            elapsed_time = time.time() - start_time
            if elapsed_time < 60 and daily_token_count >= TOKENS_PER_MINUTE:
                sleep_time = 60 - elapsed_time
                print(f"Token limit reached. Sleeping for {sleep_time:.2f} seconds.")
                time.sleep(sleep_time)
                daily_token_count = 0
                start_time = time.time()

        except Exception as e:
            print(f"Error processing CD_CVM: {cd_cvm}. Error: {str(e)}")
            print(f"Skipping CD_CVM: {cd_cvm}")
            continue
    
    if all_predictions:
        combined_df = pd.concat(all_predictions, ignore_index=True)
        return post_added_data(combined_df)
    else:
        print("No valid predictions were generated for any CD_CVM.")
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

