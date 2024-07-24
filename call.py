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

## AIDER HERE CREATE FUNCTION THAT RETURNS CALLS THE PROMPT TEMPLATE AND RETURNS THE RESULT IN A PANDAS DATAFRAME FOR A CD_CVM PASSED AS INPUT