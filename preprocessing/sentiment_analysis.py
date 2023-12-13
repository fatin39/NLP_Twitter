import pandas as pd
from textblob import TextBlob, Word
import nltk

# Load the preprocessed data
df = pd.read_csv('data/processed_data.csv')

# Ensure NLTK resources are available
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

# Convert 'processed_text' to string and handle NaN values
df['processed_text'] = df['processed_text'].astype(str)

# Lemmatization
df['lemmatized'] = df['processed_text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

# Tokenization
df['tokens'] = df['lemmatized'].apply(lambda x: TextBlob(x).words)

# Verify the results
print(df[['text','processed_text', 'label', 'lemmatized', 'tokens']].head())

# Save the DataFrame to a new CSV file
df.to_csv('processed_and_tokenized_data.csv', index=False)

""" 
Employ TextBlob, a python library to conduct sentiment analysis on the processed data
The objective is to classify each tweet as either exhibiting signs of depression or not, based on its sentiment score. 

Sentiment analysis is an NLP technique that entails analyzing the emotional tone of text, with the goal of automatically classifying it as positive, negative, or neutral, depending on the words and phrases used. 

TextBlob utilizes a machine learning algorithm to analyze text and assign a sentiment score ranging from -1 to +1, where a score of -1 indicates a very negative sentiment, +1 indicates a very positive sentiment, and 0 indicates a neutral sentiment.

By analyzing tweets using TextBlob's sentiment analysis, we can categorize them as depressed or non-depressed based on their sentiment scores. For example, a tweet with a sentiment score of -0.8 might be classified as depressed, while a tweet with a sentiment score of +0.5 may be classified as non-depressed.
"""


