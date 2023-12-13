from textblob import TextBlob
import pandas as pd

# Load the preprocessed data
df = pd.read_csv('data/processed_and_tokenized_data.csv')

# Convert 'processed_text' to string to handle NaN or non-string values
df['processed_text'] = df['processed_text'].astype(str)

# Function to classify severity based on polarity score
def classify_severity(polarity):
    if polarity < -0.6:
        return 'Severe'
    elif polarity < -0.3:
        return 'Moderate'
    elif polarity < 0:
        return 'Mild'
    else:
        return 'Not Depressed'
    
# Calculate polarity for each tweet and classify
df['polarity'] = df['processed_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['severity'] = df['polarity'].apply(classify_severity)

# Verify the results
print(df[['text','label','processed_text', 'polarity', 'severity']].head())

# # Save the results to a new CSV file
# df.to_csv('severity_analysis_results.csv', index=False)
