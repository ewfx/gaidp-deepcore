from dotenv import load_dotenv
import os
import streamlit as st
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from langchain_groq import ChatGroq
from langchain.agents.agent_types import AgentType
import io
#from langchain.agents import create_pandas_dataframe_agent
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents import AgentExecutor
from langchain.tools import Tool

load_dotenv()

#######################################################

def detect_anomalies(csv_df, profiling_rules):
    # ... (detect_anomalies function remains the same) ...
    try:
        df = csv_df

        # Basic Checks (can be expanded)
        anomalies = pd.DataFrame()

        # Transaction Amount vs Reported Amount Rule with Cross Currency Conversion
        def check_transaction_amount(row):
            if pd.isna(row['TransactionAmount']) or pd.isna(row['ReportedAmount']):
                return False  # Skip rows with missing values

            if row['Currency'] != row['ReportedCurrency']:
                deviation = abs(row['TransactionAmount'] - row['ReportedAmount']) / row['ReportedAmount']
                return deviation > 0.01  # 1% deviation
            else:
                return row['TransactionAmount'] != row['ReportedAmount']

        amount_anomalies = df[df.apply(check_transaction_amount, axis=1)]
        amount_anomalies['AnomalyType'] = 'Transaction Amount Mismatch'
        anomalies = pd.concat([anomalies, amount_anomalies], ignore_index=True)

        # Account Balance Rule
        balance_anomalies = df[
            (df['AccountBalance'] < 0) & (df['OverdraftFlag'] != 'OD')
        ]
        balance_anomalies['AnomalyType'] = 'Negative Balance (Non-OD)'
        anomalies = pd.concat([anomalies, balance_anomalies], ignore_index=True)

        # Country Jurisdiction Validation (Example, expand with real data)
        valid_countries = ['US', 'CA', 'GB', 'DE', 'JP', 'IN']  # Example valid countries
        country_anomalies = df[~df['Country'].isin(valid_countries)]
        country_anomalies['AnomalyType'] = 'Invalid Country'
        anomalies = pd.concat([anomalies, country_anomalies], ignore_index=True)

        # Cross Border Transaction Remarks
        cross_border_anomalies = df[
            (df['TransactionType'] == 'CrossBorder') &
            (df['TransactionAmount'] > 10000) &
            (pd.isna(df['TransactionRemarks']))
        ]
        cross_border_anomalies['AnomalyType'] = 'Missing Cross-Border Remarks'
        anomalies = pd.concat([anomalies, cross_border_anomalies], ignore_index=True)

        # Unsupervised Learning (Clustering and Outliers)
        if not df.empty:
            numerical_features = df.select_dtypes(include=['number'])
            if not numerical_features.empty:
                # DBSCAN Clustering
                dbscan = DBSCAN(eps=0.5, min_samples=5)
                clusters = dbscan.fit_predict(numerical_features)
                df['Cluster'] = clusters
                cluster_anomalies = df[df['Cluster'] == -1]  # -1 indicates outliers
                cluster_anomalies['AnomalyType'] = 'Clustering Outlier'
                anomalies = pd.concat([anomalies, cluster_anomalies], ignore_index=True)

                # Isolation Forest
                isolation_forest = IsolationForest(contamination=0.05)
                outliers = isolation_forest.fit_predict(numerical_features)
                df['Outlier'] = outliers
                isolation_anomalies = df[df['Outlier'] == -1]
                isolation_anomalies['AnomalyType'] = 'Isolation Forest Outlier'
                anomalies = pd.concat([anomalies, isolation_anomalies], ignore_index=True)

        return anomalies
    except Exception as e:
        print(f"Error processing CSV: {e}")
        return None

def langchain_agent_anomaly_detection(prompt,csv_df, profiling_rules, output_csv_path):
    """
    Uses a LangChain agent with Groq to detect and save anomalies.
    """
    #llm = ChatGroq(temperature=0)
    llm = ChatGroq(model_name="llama-3.3-70b-versatile")
    pandas_agent = create_pandas_dataframe_agent(llm, csv_df, verbose=True, allow_dangerous_code=True,handle_parsing_errors=True)

    #prompt = f"""
    #Analyze the following customer account data based on these profiling rules:

    #{profiling_rules}

    #Identify and extract rows that violate these rules. Additionally, use unsupervised learning techniques like clustering (DBSCAN) and outlier detection (Isolation Forest) to find further anomalies.
    #Return a CSV representation of the identified anomalies.
    #"""

    try:
        response = pandas_agent.run(prompt)
        anomaly_df = pd.read_csv(io.StringIO(response))
        anomaly_df.to_csv(output_csv_path, index=False)
        print(f"Anomalies saved to {output_csv_path}")
        return response

    except Exception as e:
        print(f"Error during LangChain agent execution: {e}")
        anomaly_df = detect_anomalies(csv_df, profiling_rules)
        if anomaly_df is not None:
            anomaly_df.to_csv(output_csv_path, index=False)
            print(f"Anomalies saved to {output_csv_path} using function.")
        else:
            print("Failed to save anomalies.")

# Example Usage
output_file = "anomalies.csv"

#########################################################
rules = """
Transaction amount should always match reported amount except when the transaction involves cross currency conversion, in which case a permissiable deviation of upto 1% is allowed.
Account balance should never be negative except incase of overdraft accounts explicitly marked with an OD flag.
Country should be an accepted jurisdiction based on bank regulation.
Cross border transaction should include mandatory transaction remarks if the amount exceeds $5000.
"""
##########################################################

#API_KEY= os.environ['OPENAI_API_KEY']
# create an LLM by instantiating OpenAI object, and passing API token
#llm = OpenAI(openai_api_key=API_KEY,temperature=0.0)
#llm=ChatGroq(model_name="llama-3.3-70b-versatile")
output_file = "anomalies.csv"

st.title("Prompt-driven data analysis")
uploaded_file = st.file_uploader("Upload a CSV file for analysis", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head(20))

    # new code below...
    prompt = st.text_area("Enter your prompt:")

    # Generate output
    if st.button("Generate"):
        if prompt:
            st.write(langchain_agent_anomaly_detection(prompt, df, rules, output_file))
        else:
            st.warning("Please enter a prompt.")