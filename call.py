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

    1. Do not include any introductory text or pleasantries.
    2. Start directly with the analysis sections as outlined below.
    3. Provide all sections in the exact order and format specified.
    4. Use at least 5 years of historical data prior to the target year for your analysis.

    Your response must follow this exact structure:

    Panel A ||| [Trend Analysis: Analyze relevant trends over at least the past five years.]
    Panel B ||| [Ratio Analysis: Calculate and analyze key financial ratios over at least the past five years, interpreting their implications for future earnings.]
    Panel C ||| [Rationale: Summarize your analyses and explain your prediction reasoning concisely, considering the long-term trends.]
    Direction ||| [increase/decrease]
    Magnitude ||| [large/moderate/small]
    Confidence ||| [0.00 to 1.00]

    Additional guidelines:
    - Be precise and focused in your explanations.
    - For Magnitude, use only one of these words: large, moderate, or small.
    - For Confidence, provide a single number between 0.00 and 1.00.
    - Do not include formulas or calculations in your response.
    - Use '|||' as a delimiter between section headers and content.
    - Ensure your analysis covers at least 5 years of historical data.
    

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
             for item in statement]
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
        magnitude = magnitude_match.group(1).lower() if magnitude_match else 'N/A'
        
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