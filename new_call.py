import os
import json
from typing import Dict, Any, List, Tuple
import langchain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import pandas as pd
import requests
from utils import *
import random
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

class PredictionOutput(BaseModel):
    trend_analysis: str = Field(alias="Panel A - Trend Analysis")
    ratio_analysis: str = Field(alias="Panel B - Ratio Analysis")
    rationale: str = Field(alias="Panel C - Rationale")
    direction: str = Field(alias="Direction")
    magnitude: str = Field(alias="Magnitude")
    confidence: float = Field(alias="Confidence")

output_parser = PydanticOutputParser(pydantic_object=PredictionOutput)

prompt_template = PromptTemplate(
    input_variables=['company_name', 'cd_cvm', 'financial_data', 'target_period'],
    template="""
    Act as a financial expert analyzing a Brazilian company. Your task is to analyze the provided financial statements for {company_name} (CVM Code: {cd_cvm}) and predict future earnings for the specified target period. 
    You MUST provide analysis and prediction for the target period by performing the following actions:

    1. Trend Analysis (Panel A): Analyze relevant trends over the past three years for this specific Brazilian company.
    2. Ratio Analysis (Panel B): Calculate and analyze financial ratios you consider relevant, provide economic interpretations of ratios and implications for future earnings.
    3. Rationale (Panel C): Summarize your analyses on trend and ratio to make a prediction specific to {company_name}. Explain your prediction reasoning concisely.
    4. Prediction: Given your previous analyses, work out a unified analysis and predict the earnings direction (increase/decrease), magnitude (large/moderate/small), and confidence (0.0-1.0) for {company_name}.

    Be concise in your explanation and address the company by "the firm" and not by {company_name}.

    Provide your response in this format for the target period:
    Panel A - Trend Analysis: [Summary of trend analysis]
    Panel B - Ratio Analysis: [Summary of ratio analysis]
    Panel C - Rationale: [Summary of rationale for prediction]
    Direction: [increase/decrease]
    Magnitude: [large/moderate/small]
    Confidence: [0.00 to 1.00]

    Info to analyze:
    Company: {company_name}
    CVM Code: {cd_cvm}
    Financial data: {financial_data}
    Target period: {target_period}
    """,
    output_parser=output_parser
)

def predict_earnings(cd_cvm, financial_data: str, target_period: str, model: str, provider: str) -> Tuple[str, int]:
    company_name = get_company_name_by_cd_cvm(cd_cvm)
    if provider == "openai":
        llm = ChatOpenAI(api_key=OPENAI_API_KEY, model=model)
        chain = LLMChain(llm=llm, prompt=prompt_template)
        with get_openai_callback() as cb:
            response = chain.run({
                'company_name': company_name,
                'cd_cvm': cd_cvm,
                'financial_data': financial_data,
                'target_period': target_period
            })
            try:
                prediction = output_parser.parse(response)
            except Exception as e:
                print(f"Output parsing failed: {e}")
                # Manually parse the response if it's not in JSON format
                prediction = manual_parse_response(response)
            token_usage = cb.total_tokens

    elif provider == "openrouter":
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt_template.format(company_name=company_name, cd_cvm=cd_cvm, financial_data=financial_data, target_period=target_period)}],
            "temperature":0,
            #"top_p":1,
            "repetition_penalty":1,
            "max_tokens": 750,
            "seed": random.randint(0, 100), 
        }
        response = requests.post(OPENROUTER_BASE_URL, headers=headers, json=data)
        if response.status_code != 200:
            raise ValueError(f"Request to OpenRouter failed with status code {response.status_code}: {response.text}")
        response_json = response.json()
        try:
            prediction = output_parser.parse(response_json['choices'][0]['message']['content'])  # Adjust according to actual API response structure
        except OutputParserException as e:
            print(f"Output parsing failed: {e}")
            prediction = None
        token_usage = response_json["usage"]["total_tokens"]
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    return prediction if isinstance(prediction, dict) else prediction.dict(), token_usage

def manual_parse_response(response: str) -> Dict[str, Any]:
    sections = {
        "trend_analysis": "",
        "ratio_analysis": "",
        "rationale": "",
        "direction": "",
        "magnitude": "",
        "confidence": 0.0
    }
    current_section = None

    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("Panel A - Trend Analysis:"):
            current_section = "trend_analysis"
            sections[current_section] = line[len("Panel A - Trend Analysis:"):].strip()
        elif line.startswith("Panel B - Ratio Analysis:"):
            current_section = "ratio_analysis"
            sections[current_section] = line[len("Panel B - Ratio Analysis:"):].strip()
        elif line.startswith("Panel C - Rationale:"):
            current_section = "rationale"
            sections[current_section] = line[len("Panel C - Rationale:"):].strip()
        elif line.startswith("Direction:"):
            current_section = "direction"
            sections[current_section] = line[len("Direction:"):].strip()
        elif line.startswith("Magnitude:"):
            current_section = "magnitude"
            sections[current_section] = line[len("Magnitude:"):].strip()
        elif line.startswith("Confidence:"):
            current_section = "confidence"
            sections[current_section] = float(line[len("Confidence:"):].strip())
        elif current_section:
            sections[current_section] += " " + line

    return sections

def parse_prediction(prediction: Dict[str, Any]) -> Dict[str, Any]:
    return prediction

def run_predictions(cd_cvm_list: List[str], model: str, provider: str) -> pd.DataFrame:
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
                'income_statement': income_statement[sorted_dates[max(0, target_index-5):target_index]].to_dict(),
                'balance_sheet': balance_sheet[sorted_dates[max(0, target_index-5):target_index]].to_dict(),
                'cash_flow_statement': cash_flow[sorted_dates[max(0, target_index-5):target_index]].to_dict()
            }
            financial_data = json.dumps(relevant_data)
            company_name = get_company_name_by_cd_cvm(cd_cvm)
            prediction, token_usage = predict_earnings(cd_cvm, financial_data, target_period, model, provider)
            print(f"Unparsed response for {cd_cvm} during {target_period}:\n{prediction}\n")
            parsed_prediction = parse_prediction(prediction)
            print(f"Parsed prediction for {cd_cvm} during {target_period}:\n{parsed_prediction}\n")
            
            results.append({
                'Company': company_name,
                'Model': f"{provider}/{model}",
                'TREND ANALYSIS': parsed_prediction['trend_analysis'],
                'RATIO ANALYSIS': parsed_prediction['ratio_analysis'].replace('\n', ' '),
                'RATIONALE': parsed_prediction['rationale'],
                'DIRECTION': 1 if parsed_prediction['direction'] == 'increase' else (-1 if parsed_prediction['direction'] == 'decrease' else 0),
                'MAGNITUDE': parsed_prediction['magnitude'],
                'CONFIDENCE LEVEL': parsed_prediction['confidence'],
                'ACTUAL DIRECTION': actual_result,
                'CD_CVM': cd_cvm,
                'TARGET PERIOD': target_period,
                'TOKEN USAGE': token_usage
            })

    return pd.DataFrame(results)

from sklearn.metrics import precision_score, f1_score
import pandas as pd

def evaluate_predictions(df):
    # Calculate precision and f1 score
    precision = precision_score(df['ACTUAL DIRECTION'], df['DIRECTION'], average='binary', pos_label=1).round(3)
    f1 = f1_score(df['ACTUAL DIRECTION'], df['DIRECTION'], average='binary', pos_label=1).round(3)
    Company = df['Company'].unique()   
    Model = df['Model'].unique()
    #number of predictions
    N_Pred = len(df)
    # Calculate average confidence level
    average_confidence = df['CONFIDENCE LEVEL'].mean().round(3)
    
    # Create a DataFrame to return the results
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
    
    results = main(cvm_code, model, provider)
    print(results)
