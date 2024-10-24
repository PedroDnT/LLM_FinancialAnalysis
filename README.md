**Objective:**

LLM_FinancialAnalysis is a project aimed at replicating the methodology used in the paper “Financial Analysis with LLM." This repository focuses on applying the methodology specifically to Brazilian public companies, aiming to replicate the study's methodology and learn how to interact with the OpenAI API.

**Description:**

In this project, I built a pipeline using OpenAI API to analyze financial statements stored in an SQL database. With this data, the model was prompted to analyze and predict key financial metrics. The objective was to test the ability of financial LLMs to predict future financial performance, particularly focusing on earnings.

**Here's a step-by-step breakdown of the process:**

1. **Data Retrieval:** Using SQL, I retrieve financial data such as income statements and balance sheets for a list of companies based on their codes.
2. **Prompt Creation:** I designed a custom prompt template for the LLM, ensuring analysis of trends over at least five years of historical data.
3. **Prediction Generation:** The financial data and prompts are passed to OpenAI's GPT-4 model, which generates earnings predictions for different years, including trend analysis, key financial ratios, and reasoning behind the predictions.
