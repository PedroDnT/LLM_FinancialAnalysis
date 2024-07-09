import os
import json
import time
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from sklearn.metrics import precision_score, f1_score
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
from D_fetcher import get_company_name_by_cd_cvm, execute_query

def calculate_actual_results(income_statement: pd.DataFrame) -> Tuple[List[Tuple[str, int]], List[str]]:
    start_time = time.time()
    earnings_column = 'Resultado Líquido das Operações Continuadas'
    results = []
    
    if 'DS_CONTA' not in income_statement.columns:
        raise ValueError("Expected 'DS_CONTA' column in income statement")
    
    earnings_rows = income_statement[income_statement['DS_CONTA'] == earnings_column]
    date_columns = sorted([col for col in earnings_rows.columns if col.startswith('20') and col.endswith('-12-31')])
    
    for i in range(5, len(date_columns)):
        current_earnings = earnings_rows[date_columns[i]].values[0]
        previous_earnings = earnings_rows[date_columns[i-1]].values[0]
        
        if pd.notnull(current_earnings) and pd.notnull(previous_earnings):
            result = 1 if current_earnings > previous_earnings else -1
            results.append((date_columns[i], result))
    
    print(f"Time taken to calculate actual results: {time.time() - start_time} seconds")
    return results, date_columns

def create_prompt_template() -> ChatPromptTemplate:
    template = """
    As a financial expert, analyze the provided financial statements and predict future earnings for the specified target period. Provide analysis and prediction for the target period as follows:

    1. Trend Analysis (Panel A): Analyze relevant trends over the past three years.
    2. Ratio Analysis (Panel B): Calculate and analyze financial ratios you consider relevant. Interpret them and discuss implications for future earnings.
    3. Rationale (Panel C): Summarize your analyses on trend and ratio to make a prediction. Explain your prediction reasoning concisely.
    4. Prediction: Given your previous analyses, provide a unified analysis and predict the earnings direction (increase/decrease), magnitude (large/moderate/small), and confidence (0.0-1.0).
    
    Directives: Direction will be interpreted as 1 for increase and -1 for decrease.
    Provide all sections (Panel A, B, C, Direction, Magnitude, and Confidence) for the target period.
    Be precise in your explanation, focus on explaining rationales and analysis, no need to explain formulas nor justify the Confidence or ratios formulas.

    You MUST Provide your analysis in this format for the every target period:
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
    start_time = time.time()
    with get_openai_callback() as cb:
        response = chain.invoke({"financial_data": financial_data, "target_period": target_period})

    prediction = {
        'trend_analysis': 'Analysis not provided',
        'ratio_analysis': 'Analysis not provided',
        'rationale': 'Rationale not provided',
        'direction': 0,
        'magnitude': 'unknown',
        'confidence': 0.0,
        'log_prob': 0.0
    }

    response_text = response.content if hasattr(response, 'content') else str(response)

    sections = response_text.split('Panel')
    for section in sections:
        if 'A - Trend Analysis:' in section:
            prediction['trend_analysis'] = section.split('A - Trend Analysis:', 1)[1].strip()
        elif 'B - Ratio Analysis:' in section:
            prediction['ratio_analysis'] = section.split('B - Ratio Analysis:', 1)[1].strip()
        elif 'C - Rationale:' in section:
            prediction['rationale'] = section.split('C - Rationale:', 1)[1].strip()

    for line in response_text.split('\n'):
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

    prediction['magnitude'] = 'unknown' if 'cagr' in prediction['magnitude'] else prediction['magnitude']
    prediction['token_usage'] = {
        'total_tokens': cb.total_tokens,
        'prompt_tokens': cb.prompt_tokens,
        'completion_tokens': cb.completion_tokens
    }

    print(f"Time taken for prediction: {time.time() - start_time} seconds")
    print(f"Token usage: Prompt: {cb.prompt_tokens}, Completion: {cb.completion_tokens}, Total: {cb.total_tokens}")
    return prediction

def process_company_data(cd_cvm: str, model_name: str, chain: RunnableSequence, financial_statements: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
    start_time = time.time()
    results = []
    income_statement = financial_statements['income_statement'].get(cd_cvm)
    balance_sheet = financial_statements['balance_sheet'].get(cd_cvm)
    cash_flow = financial_statements['cash_flow'].get(cd_cvm)

    if income_statement is None or balance_sheet is None or cash_flow is None:
        return results

    actual_results, sorted_dates = calculate_actual_results(income_statement)
    financial_data = {
        'income_statement': income_statement.to_dict(orient='records'),
        'balance_sheet': balance_sheet.to_dict(orient='records'),
        'cash_flow_statement': cash_flow.to_dict(orient='records')
    }

    company_name = get_company_name_by_cd_cvm(cd_cvm)

    for target_period, actual_result in actual_results:
        target_index = sorted_dates.index(target_period)
        relevant_data = {k: v for k, v in financial_data.items() if k in sorted_dates[max(0, target_index-5):target_index]}
        financial_data_json = json.dumps(relevant_data)

        try:
            prediction = get_financial_prediction(financial_data_json, target_period, chain)
            results.append({
                'Company': company_name,
                'Model': model_name,
                'TREND ANALYSIS': prediction['trend_analysis'],
                'RATIO ANALYSIS': prediction['ratio_analysis'].replace('\n', ' '),
                'RATIONALE': prediction['rationale'],
                'DIRECTION': prediction['direction'],
                'MAGNITUDE': prediction['magnitude'],
                'CONFIDENCE LEVEL': prediction['confidence'],
                'LOG PROBABILITY': prediction['log_prob'],
                'ACTUAL DIRECTION': actual_result,
                'CD_CVM': cd_cvm,
                'TARGET PERIOD': target_period
            })
        except Exception as exc:
            print(f'Generated an exception for {cd_cvm}, {target_period}: {exc}')
            print(f'Exception type: {type(exc)}')
            print(f'Exception details: {str(exc)}')

    return results

def run_predictions(cd_cvm_list: List[str], models_to_test: List[tuple]) -> pd.DataFrame:
    all_results = []
    
    # Fetch financial statements for all companies
    income_statements = execute_query(cd_cvm_list, 'ist')
    balance_sheets = execute_query(cd_cvm_list, 'bs')
    cash_flows = execute_query(cd_cvm_list, 'cf')
    
    financial_statements = {
        'income_statement': income_statements,
        'balance_sheet': balance_sheets,
        'cash_flow': cash_flows
    }

    for model_name, model_kwargs in models_to_test:
        print(f"\nTesting model: {model_name}")
        llm = get_llm(model_name, **model_kwargs)
        chain = create_prompt_template() | llm

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(process_company_data, cd_cvm, model_name, chain, financial_statements) 
                       for cd_cvm in cd_cvm_list]
            for future in futures:
                all_results.extend(future.result())

    return pd.DataFrame(all_results)
def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    metrics = []
    for cvm_code in df['CD_CVM'].unique():
        for model in df['Model'].unique():
            filtered_df = df[(df['CD_CVM'] == cvm_code) & (df['Model'] == model)]
            if not filtered_df.empty:
                precision = precision_score(filtered_df['ACTUAL DIRECTION'], filtered_df['DIRECTION'], average='weighted', zero_division=0)
                f1 = f1_score(filtered_df['ACTUAL DIRECTION'], filtered_df['DIRECTION'], average='weighted', zero_division=0)
                avg_confidence = filtered_df['CONFIDENCE LEVEL'].mean()
                mean_log_prob = filtered_df['LOG PROBABILITY'].mean()
                metrics.append({
                    'CD_CVM': cvm_code,
                    'Model': model,
                    'Precision': round(precision, 2),
                    'F1 Score': round(f1, 2),
                    'Average Confidence Level': round(avg_confidence, 2),
                })
    return pd.DataFrame(metrics)

def main(cvm_code: str, models_to_test: List[tuple]) -> pd.DataFrame:
    return run_predictions([cvm_code], models_to_test)

if __name__ == "__main__":
    # Example usage:
    # models_to_test = [("gpt-3.5-turbo", {}), ("gpt-4", {})]
    # result_df = main("12345", models_to_test)
    # print(result_df)
    pass