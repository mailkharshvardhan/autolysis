# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "seaborn",
#   "matplotlib",
#   "httpx",
#   "openai",
#   "chardet",
#   "scikit-learn",
# ]
# ///

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import openai
import chardet
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

def test_openai_api():
    try:
        response = openai.Completion.create(
            model="gpt-4o-mini",
            prompt="Say hello.",
            max_tokens=5
        )
        print("OpenAI API is working. Response:", response['choices'][0]['text'].strip())
    except Exception as e:
        print(f"Error connecting to OpenAI API: {e}")

def detect_encoding(file_path):
    """
    Detects the encoding of a file using chardet.
    """
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    encoding = result['encoding']
    confidence = result['confidence']
    print(f"Detected encoding: {encoding} with confidence {confidence*100:.2f}%")
    return encoding

def load_data(file_path):
    """
    Loads CSV data into a pandas DataFrame with proper encoding handling.
    """
    encoding = detect_encoding(file_path)
    try:
        df = pd.read_csv(file_path, encoding=encoding)
        print(f"File '{file_path}' loaded successfully with encoding '{encoding}'.")
    except Exception as e:
        print(f"Error reading the file: {file_path}. Error: {e}")
        return None
    return df

def analyze_data(df):
    """
    Performs data analysis: summary statistics, missing values, and correlation matrix.
    Filters out non-numeric columns for correlation.
    """
    # Summary statistics for all columns
    summary = df.describe(include='all').to_dict()
    
    # Missing values per column
    missing_values = df.isnull().sum().to_dict()
    
    # Select only numeric columns for correlation
    df_numeric = df.select_dtypes(include=[np.number])
    
    # Calculate the correlation matrix
    correlation_matrix = df_numeric.corr().to_dict()
    
    return summary, missing_values, correlation_matrix

def create_visualizations(df):
    """
    Creates and saves visualizations: Correlation Heatmap and Distribution Plot.
    Only uses numeric data for visualizations.
    """
    # Select only numeric columns
    df_numeric = df.select_dtypes(include=[np.number])
    
    if df_numeric.empty:
        print("No numeric data available for visualizations.")
        return
    
    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    correlation_matrix = df_numeric.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()
    print("Saved 'correlation_heatmap.png'.")
    
    # Distribution Plot for the first numerical column
    first_numeric_col = df_numeric.columns[0]
    plt.figure(figsize=(8, 6))
    sns.histplot(df_numeric[first_numeric_col].dropna(), kde=True)
    plt.title(f'Distribution of {first_numeric_col}')
    plt.tight_layout()
    plt.savefig('distribution_plot.png')
    plt.close()
    print("Saved 'distribution_plot.png'.")

def generate_narrative(df, summary, missing_values, correlation_matrix):
    """
    Generates a narrative summary of the data analysis using OpenAI's GPT model.
    """
    prompt = f"""
    Dataset Overview:
    Columns: {list(df.columns)}
    
    Summary Statistics:
    {summary}
    
    Missing Values:
    {missing_values}
    
    Correlation Matrix:
    {correlation_matrix}
    
    Based on this analysis, write a detailed narrative on the dataset.
    - Describe the data.
    - Mention any significant insights from the summary statistics and correlations.
    - Suggest what actions can be taken based on the insights.
    """
    
    try:
        response = openai.Completion.create(
            model="gpt-4o-mini",  # Ensure you're using the correct model supported by AIPROXY
            prompt=prompt,
            max_tokens=1000,
            temperature=0.7,
        )
        narrative = response['choices'][0]['text'].strip()
        print("Narrative generated successfully.")
        return narrative
    except Exception as e:
        print(f"Error generating narrative: {e}")
        return "Narrative generation failed."

def generate_report(narrative):
    """
    Generates a Markdown report with the narrative and embedded visualizations.
    """
    with open('README.md', 'w') as f:
        f.write("# Automated Data Analysis Report\n\n")
        f.write("## Data Analysis\n\n")
        f.write(narrative)
        f.write("\n\n### Visualizations\n")
        f.write("![Correlation Heatmap](correlation_heatmap.png)\n")
        f.write("![Distribution Plot](distribution_plot.png)\n")
    print("Saved 'README.md'.")

def main(file_path):
    """
    Main function to orchestrate the data analysis workflow.
    """
    # Load the data
    df = load_data(file_path)
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Perform data analysis
    summary, missing_values, correlation_matrix = analyze_data(df)
    
    # Create visualizations
    create_visualizations(df)
    
    # Generate LLM narrative
    narrative = generate_narrative(df, summary, missing_values, correlation_matrix)
    
    # Generate the final report
    generate_report(narrative)
def main(file_path):
    """
    Main function to orchestrate the data analysis workflow.
    """
    # Test OpenAI API
    test_openai_api()
    
    # Load the data
    df = load_data(file_path)
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Perform data analysis
    summary, missing_values, correlation_matrix = analyze_data(df)
    
    # Create visualizations
    create_visualizations(df)
    
    # Generate LLM narrative
    narrative = generate_narrative(df, summary, missing_values, correlation_matrix)
    
    # Generate the final report
    generate_report(narrative)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if not os.path.isfile(file_path):
            print(f"File '{file_path}' does not exist. Please provide a valid CSV file.")
        else:
            main(file_path)
    else:
        print("Usage: uv run autolysis.py <dataset.csv>")

