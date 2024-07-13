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


# --- Functions related to prediction logic ---

def calculate_actual_results(income_statement: pd.DataFrame) -> Tuple[List[Tuple[str, int]], List[str]]:
    """Calculates actual earnings direction based on income statement."""
    earnings_column = 'Resultado Líquido das Operações Continuadas'
    results = []
    if 'DS_CONTA' not in income_statement.columns:
        raise ValueError("Expected 'DS_CONTA' column in income statement")
    earnings_rows = income_statement[income_statement['DS_CONTA'] == earnings_column]
    date_columns = [col for col in earnings_rows.columns if col.startswith('20') and col.endswith('-12-31')]
    sorted_dates = sorted(date_columns)
    for i in range(5, len(sorted_dates)):
        current_earnings = earnings_rows[sorted_dates[i]].values[0]
        previous_earnings = earnings_rows[sorted_dates[i-1]].values[0]
        if pd.notnull(current_earnings) and pd.notnull(previous_earnings):
            result = 1 if current_earnings > previous_earnings else -1
            period = sorted_dates[i]
            results.append((period, result))
    return results, sorted_dates

def create_prompt_template() -> ChatPromptTemplate:
    """Creates a prompt template for the financial prediction task."""
    template = """
    As a financial expert, your task analyze the provided financial statements and predict future earnings for the specified target period. You MUST provide analysis and prediction for the target period by performing the following actions:

    1. Trend Analysis (Panel A): Analyze relevant trends over the past three years.
    2. Ratio Analysis (Panel B): Calculate and analyze  financial ratios you consider relevant and provide economic interpretations of the computed ratios interpret them and implications for future earnings.
    3. Rationale (Panel C): Summarize your analyzes on trend and ration to make a prediction. Explain your prediction reasoning concisely.
    4. Prediction: Given your previous analyzes, workout a unified analysis and predict the earnings direction (increase/decrease), magnitude (large/moderate/small), and confidence (0.0-1.0).

    Directives: Direction will be interpreted as 1 for increase and -1 for decrease.
    You MUST provide all sections (Panel A, B, C, Direction, Magnitude, and Confidence) for the target period.
    Be precise in your explanation, focus on explaining rationales and analysis.

    Provide your analysis in this format for the target period:
    Panel A - Trend Analysis: [Summary of trend analysis]
    Panel B - Ratio Analysis: [Summary of ratio analysis]
    Panel C - Rationale: [Summary of rationale for prediction]
    Direction: [increase/decrease]
    Magnitude: [large/moderate/small]
    Confidence: [0.00 to 1.00]

    Info to analyze:

    Financial data: {financial_data}
    Target period: {target_period}

    """
    return ChatPromptTemplate.from_template(template)

def get_llm(model_name: str, **kwargs) -> ChatOpenAI:
    """Gets a language model instance."""
    base_kwargs = {
        "temperature": 0,
        "model_kwargs": {"logprobs": True, "top_p": 1},
        **kwargs
    }
    return ChatOpenAI(
            model=model_name,
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            **base_kwargs
        )

def get_financial_prediction(financial_data: str, target_period: str, chain: RunnableSequence) -> Dict[str, Any]:
    """Makes a financial prediction using the provided chain."""
    with get_openai_callback() as cb:
        response = chain.invoke({"financial_data": financial_data, "target_period": target_period})

    prediction = {
        'trend_analysis': 'Analysis not provided',
        'ratio_analysis': 'Analysis not provided',
        'rationale': 'Rationale not provided',
        'direction': 0,
        'magnitude': 'unknown',
        'confidence': 0.0,
        'log_prob': 0.0  # Initialize log_prob
    }

    # Check if response is an AIMessage object
    if hasattr(response, 'content'):
        response_text = response.content
    else:
        response_text = str(response)

    # Split the response into sections
    sections = response_text.split('Panel')
    for section in sections:
        if 'A - Trend Analysis:' in section:
            prediction['trend_analysis'] = section.split('A - Trend Analysis:', 1)[1].strip()
        elif 'B - Ratio Analysis:' in section:
            prediction['ratio_analysis'] = section.split('B - Ratio Analysis:', 1)[1].strip()
        elif 'C - Rationale:' in section:
            prediction['rationale'] = section.split('C - Rationale:', 1)[1].strip()

    # Extract direction, magnitude, and confidence
    lines = response_text.split('\n')
    for line in lines:
        if line.startswith('Direction:'):
            direction = line.split(':', 1)[1].strip().lower()
            prediction['direction'] = 1 if 'increase' in direction else (-1 if 'decrease' in direction else 0)
        elif line.startswith('Magnitude:'):
            prediction['magnitude'] = line.split(':', 1)[1].strip().lower()
        elif line.startswith('Confidence:'):
            try:
                prediction['confidence'] = float(line.split(':', 1)[1].strip())
            except ValueError:
                prediction['confidence'] = 0.0

    # Clean up the magnitude field
    if 'cagr' in prediction['magnitude']:
        prediction['magnitude'] = 'unknown'

    prediction['token_usage'] = {
        'total_tokens': cb.total_tokens,
        'prompt_tokens': cb.prompt_tokens,
        'completion_tokens': cb.completion_tokens
    }

    return prediction

def run_predictions(cd_cvm_list: List[str], models_to_test: List[tuple]) -> pd.DataFrame:
    """Runs predictions for a list of CD_CVM codes using multiple models."""
    results = []

    # Get financial statements for all companies at once
    income_statements, balance_sheets, cash_flows = get_financial_statements_batch(cd_cvm_list)

    for model_name, model_kwargs in models_to_test:
        print(f"\nTesting model: {model_name}")

        llm = get_llm(model_name, **model_kwargs)
        prompt = create_prompt_template()
        chain = prompt | llm

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_to_prediction = {}
            for cd_cvm in cd_cvm_list:
                income_statement = income_statements.get(cd_cvm)
                balance_sheet = balance_sheets.get(cd_cvm)
                cash_flow = cash_flows.get(cd_cvm)

                if income_statement is None or balance_sheet is None or cash_flow is None:
                    continue

                actual_results, sorted_dates = calculate_actual_results(income_statement)
                # Prepare financial data for all periods
                financial_statements = {
                    'income_statement': income_statement.to_dict(orient='records'),
                    'balance_sheet': balance_sheet.to_dict(orient='records'),
                    'cash_flow_statement': cash_flow.to_dict(orient='records')
                }

                for target_period, actual_result in actual_results:
                    # Find the index of the target period
                    target_index = sorted_dates.index(target_period)
                    # Use only data from periods before the target period
                    relevant_data = {k: v for k, v in financial_statements.items() if k in sorted_dates[max(0, target_index-5):target_index]}
                    financial_data = json.dumps(relevant_data)

                    future = executor.submit(get_financial_prediction, financial_data, target_period, chain)
                    future_to_prediction[(cd_cvm, target_period, actual_result)] = future
                    company_name = get_company_name_by_cd_cvm(cd_cvm)

            for (cd_cvm, target_period, actual_result), future in future_to_prediction.items():
                try:
                    prediction = future.result()
                    results.append({
                        'Company': company_name,
                        'Model': model_name,
                        'TREND ANALYSIS': prediction['trend_analysis'],
                        'RATIO ANALYSIS': prediction['ratio_analysis'].replace('\n', ' '),  # Clean the RATIO ANALYSIS
                        'RATIONALE': prediction['rationale'],
                        'DIRECTION': prediction['direction'],
                        'MAGNITUDE': prediction['magnitude'],
                        'CONFIDENCE LEVEL': prediction['confidence'],
                        'LOG PROBABILITY': prediction['log_prob'],  # Add log probability to results
                        'ACTUAL DIRECTION': actual_result,
                        'CD_CVM': cd_cvm,
                        'TARGET PERIOD': target_period
                    })
                except Exception as exc:
                    print(f'Generated an exception for {cd_cvm}, {target_period}: {exc}')
                    print(f'Exception type: {type(exc)}')
                    print(f'Exception details: {str(exc)}')

    return pd.DataFrame(results)

# --- Main execution ---

def main(cvm_code):
    """Main function to run predictions."""
    models_to_test = [
        ('anthropic/claude-3.5-sonnet:beta', {}),
        # Add more models here
    ]
    return run_predictions([cvm_code], models_to_test)

if __name__ == "__main__":
    main()