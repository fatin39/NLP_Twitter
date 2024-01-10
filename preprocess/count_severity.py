import pandas as pd

# Load the provided dataset
file_path = '/Users/nurfatinaqilah/Documents/streamlit-test/dataset/severity_depression.csv'
df = pd.read_csv(file_path)

# Analyzing the distribution of actual labels and predicted severities
label_distribution = df['label'].value_counts()
severity_distribution = df['severity_prediction'].value_counts()

(label_distribution, severity_distribution)
