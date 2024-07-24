import os
import re
from typing import Dict, Any, List, Tuple
import random
import requests
from dotenv import load_dotenv
from sklearn.metrics import precision_score, f1_score
import pandas as pd
from utils import get_company_name_by_cd_cvm, get_financial_statements_batch, calculate_actual_results
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables.base import RunnableSequence
from langchain.callbacks import get_openai_callback
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import OutputParserException, HumanMessage
from pydantic import BaseModel, Field
import js

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

class PredictionOutput(BaseModel):
    company: str = Field(alias="Company")
    model: str = Field(alias="Model")
    trend_analysis: str = Field(alias="TREND ANALYSIS")
    ratio_analysis: str = Field(alias="RATIO ANALYSIS")
    rationale: str = Field(alias="RATIONALE")
    direction: str = Field(alias="DIRECTION")
    magnitude: str = Field(alias="MAGNITUDE")
    confidence: float = Field(alias="CONFIDENCE LEVEL")
    actual_direction: int = Field(alias="ACTUAL DIRECTION")
    cd_cvm: str = Field(alias="CD_CVM")
    target_period: str = Field(alias="TARGET PERIOD")
    token_usage: int = Field(alias="TOKEN USAGE")

output_parser = PydanticOutputParser(pydantic_object=PredictionOutput)

prompt_template = PromptTemplate(
    input_variables=['company_name', 'cd_cvm', 'financial_data', 'target_period', 'model', 'provider'],
    template="""
    Act as a financial expert analyzing a Brazilian company. Your task is to analyze the provided financial statements for {company_name} (CVM Code: {cd_cvm}) and predict future earnings for the specified target period. 
    You MUST provide analysis and prediction for the target period by performing the following actions:

    1. Trend Analysis (Panel A): Analyze relevant trends over the past three years for this specific Brazilian company.
    2. Ratio Analysis (Panel B): Calculate and analyze financial ratios you consider relevant, provide economic interpretations of ratios and implications for future earnings.
    3. Rationale (Panel C): Summarize your analyses on trend and ratio to make a prediction specific to {company_name}. Explain your prediction reasoning concisely.
    4. Prediction: Given your previous analyses, work out a unified analysis and predict the earnings direction (increase/decrease), magnitude (large/moderate/small), and confidence (0.0-1.0) for {company_name}.

    Provide your response in the following format:
    Panel A - Trend Analysis: [Summary of trend analysis]
    Panel B - Ratio Analysis: [Summary of ratio analysis]
    Panel C - Rationale: [Summary of rationale for prediction]
    Direction: [increase/decrease]
    Magnitude: [large/moderate/small]
    Confidence: [0.00 to 1.00]
    Company: {company_name}
    Model: {provider}/{model}
    CD_CVM: {cd_cvm}
    TARGET PERIOD: {target_period}
    TOKEN USAGE: [token usage]
    """
)

def predict_earnings(cd_cvm, financial_data: str, target_period: str, model: str, provider: str) -> Tuple[Dict[str, Any], int]:
    company_name = get_company_name_by_cd_cvm(cd_cvm)
    if provider == "openai":
        llm = ChatOpenAI(api_key=OPENAI_API_KEY, model=model)
        chain = RunnableSequence(first=prompt_template, last=llm)
        with get_openai_callback() as cb:
            response = chain.invoke({
                "company_name": company_name,
                "cd_cvm": cd_cvm,
                "financial_data": financial_data,
                "target_period": target_period,
                "model": model,
                "provider": provider
            })
            response_content = response.content if hasattr(response, 'content') else response
            print(f"Response content: {response_content}")
            print(f"Response content type: {type(response_content)}")
            if not response_content:
                raise ValueError("Response content is empty, cannot parse.")
            response_content = response_content.strip("```").strip()
            prediction = manual_parse_response(response_content)
            token_usage = cb.total_tokens if cb else 0

    elif provider == "openrouter":
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt_template.format(
                company_name=company_name, 
                cd_cvm=cd_cvm, 
                financial_data=financial_data, 
                target_period=target_period,
                model=model,
                provider=provider
            )}],
            "temperature": 0,
            "repetition_penalty": 1,
            "max_tokens": 750,
            "seed": random.randint(0, 100), 
        }
        response = requests.post(OPENROUTER_BASE_URL, headers=headers, json=data)
        if response.status_code != 200:
            raise ValueError(f"Request to OpenRouter failed with status code {response.status_code}: {response.text}")
        response_json = response.json()
        try:
            response_content = response_json['choices'][0]['message']['content']
            print(f"Response content: {response_content}")
            prediction = manual_parse_response(response_content)
            token_usage = response_json.get("usage", {}).get("total_tokens", 0)
        except KeyError as e:
            print(f"Output parsing failed: {e}")
            prediction = manual_parse_response(response_content)
            token_usage = response_json.get("usage", {}).get("total_tokens", 0)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    return prediction, token_usage

def manual_parse_response(response: str) -> Dict[str, Any]:
    print(f"Response content before manual parsing: {response}")
    # Use regex or string manipulation to parse the response
    try:
        parsed_response = {}
        parsed_response['TREND ANALYSIS'] = re.search(r"Panel A - Trend Analysis:\s*(.*)", response).group(1)
        parsed_response['RATIO ANALYSIS'] = re.search(r"Panel B - Ratio Analysis:\s*(.*)", response).group(1)
        parsed_response['RATIONALE'] = re.search(r"Panel C - Rationale:\s*(.*)", response).group(1)
        parsed_response['DIRECTION'] = re.search(r"Direction:\s*(.*)", response).group(1)
        parsed_response['MAGNITUDE'] = re.search(r"Magnitude:\s*(.*)", response).group(1)
        parsed_response['CONFIDENCE LEVEL'] = float(re.search(r"Confidence:\s*(.*)", response).group(1))
        parsed_response['Company'] = re.search(r"Company:\s*(.*)", response).group(1)
        parsed_response['Model'] = re.search(r"Model:\s*(.*)", response).group(1)
        parsed_response['CD_CVM'] = re.search(r"CD_CVM:\s*(.*)", response).group(1)
        parsed_response['TARGET PERIOD'] = re.search(r"TARGET PERIOD:\s*(.*)", response).group(1)
        parsed_response['TOKEN USAGE'] = int(re.search(r"TOKEN USAGE:\s*(.*)", response).group(1))
        return parsed_response
    except AttributeError as e:
        print(f"Manual parsing failed: {e}")
        raise ValueError("Response is not in valid format and cannot be parsed manually.")

def parse_prediction(prediction: Dict[str, Any]) -> Dict[str, Any]:
    return prediction

def run_predictions(cd_cvm_list: List[str], model: str, provider: str, n_years: int = 1) -> pd.DataFrame:
    results = []
    income_statements, balance_sheets, cash_flows = get_financial_statements_batch(cd_cvm_list)

    for cd_cvm in cd_cvm_list:
        income_statement = income_statements.get(cd_cvm)
        balance_sheet = balance_sheets.get(cd_cvm)
        cash_flow = cash_flows.get(cd_cvm)
        company_name = get_company_name_by_cd_cvm(cd_cvm)
        if income_statement is None or balance_sheet is None or cash_flow is None:
            continue

        company_name = get_company_name_by_cd_cvm(cd_cvm)
        actual_results, sorted_dates = calculate_actual_results(income_statement)

        for target_period, actual_result in actual_results:
            target_index = sorted_dates.index(target_period)
            relevant_data = {
                'income_statement': income_statement[sorted_dates[max(0, target_index - n_years * 5):target_index]].to_dict(),
                'balance_sheet': balance_sheet[sorted_dates[max(0, target_index - n_years * 5):target_index]].to_dict(),
                'cash_flow_statement': cash_flow[sorted_dates[max(0, target_index - n_years * 5):target_index]].to_dict()
            }
            financial_data = json.dumps(relevant_data)
            company_name = get_company_name_by_cd_cvm(cd_cvm)
            prediction, token_usage = predict_earnings(cd_cvm, financial_data, target_period, model, provider)
            prediction['Company'] = company_name
            prediction['Model'] = f"{provider}/{model}"
            prediction['ACTUAL DIRECTION'] = actual_result
            prediction['CD_CVM'] = cd_cvm
            prediction['TARGET PERIOD'] = target_period
            prediction['TOKEN USAGE'] = token_usage

            results.append(prediction)

    return pd.DataFrame(results)

def evaluate_predictions(df):
    precision = precision_score(df['ACTUAL DIRECTION'], df['DIRECTION'], average='binary', pos_label=1).round(3)
    f1 = f1_score(df['ACTUAL DIRECTION'], df['DIRECTION'], average='binary', pos_label=1).round(3)
    Company = df['Company'].unique()   
    Model = df['Model'].unique()
    N_Pred = len(df)
    average_confidence = df['CONFIDENCE LEVEL'].mean().round(3)
    
    results_df = pd.DataFrame({
        'Company': Company,
        'Model': Model,
        'Precision': [precision],
        'F1 Score': [f1],
        'Average Confidence Level': [average_confidence],
        'Number of Predictions': [N_Pred],
    })
    
    return results_df

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python call.py <cvm_code> <model> <provider>")
        sys.exit(1)
    
    cvm_code = sys.argv[1]
    model = sys.argv[2]
    provider = sys.argv[3]
    
    results = run_predictions([cvm_code], model, provider)
    print(results)
