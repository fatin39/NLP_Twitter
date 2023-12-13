import pandas as pd
import os

# Combine all datasets
#df_combined = pd.concat([df_sentiment140, df_depressive_tweets, df_non_depressive_tweets, df_suicide_notes])

# Shuffle the combined dataset
#df_combined = df_combined.sample(frac=1).reset_index(drop=True)

# Print the first few rows of the combined DataFrame to verify
#print(df_combined.head())
sentiment_analysis_path = 'data/sentiment_analysis.csv'

# Specify the encoding and column names
# Latin-1, also called ISO-8859-1,
# is an 8-bit character set endorsed by the International Organization for Standardization (ISO)
encoding = 'ISO-8859-1'
columns = ['label', 'ids', 'date', 'flag', 'user', 'text']

# Read the CSV file into a DataFrame
# DataFrame will contain your data with the specified column names and encoding.
sentiment_analysis_df = pd.read_csv(sentiment_analysis_path, encoding = encoding, names = columns)

# # First 5 data of the dataset
# print(df.head())

# # Last 5 data of the dataset
# print(df.tail())

# # print the shape of the dataset
# print(df.shape)

# # To find any NaN values
# print(df.isna().sum())

# # To find any NULL values
# print(df.isnull().sum())

# # info of the data
# print(df.info())

# print(df["label"].value_counts())

# Data Augmentation

# The only data we need
sentiment_analysis_df = sentiment_analysis_df[['text','label']]

# 0 represents as negative and 4 as positive, not ideal
# change to 0 and 1

sentiment_analysis_df['label'] = sentiment_analysis_df['label'].replace(4,1)

# print("Sentiment_analysis.csv")
# print(df.head())

# -----

# Add depressed_tweets.csv data
depressed_tweets_path = 'data/depressed_tweets.csv'

# Load the dataset
# Assuming the tweet text is in a column named 'tweet'
depressed_tweets_df = pd.read_csv(depressed_tweets_path)

# Rename the 'tweet' column to 'text'
depressed_tweets_df.rename(columns={'tweet': 'text'}, inplace=True)

# Add a label column for depressive tweets (2)
depressed_tweets_df['label'] = 2

# The only data we need
depressed_tweets_df = depressed_tweets_df[['text','label']]

# print("depressed_tweets.csv")

# # # First 5 data of the dataset
# print(depressed_tweets_df.head())

# ----
# Add nondepressed_tweets.csv data
nondepressed_tweets_path = 'data/nondepressed_tweets.csv'

# Load the dataset
# Assuming the tweet text is in a column named 'tweet'
nondepressed_tweets_df = pd.read_csv(nondepressed_tweets_path)

# Rename the 'tweet' column to 'text'
nondepressed_tweets_df.rename(columns={'tweet': 'text'}, inplace=True)

# Add a label column for nondepressive tweets (1)
nondepressed_tweets_df['label'] = 1

# The only data we need
nondepressed_tweets_df = nondepressed_tweets_df[['text','label']]

# print("nondepressed_tweets.csv")

# # First 5 data of the dataset
# print(nondepressed_tweets_df.head())

# ----
# Add nondepressed_tweets.csv data
suicide_text_path = 'data/suicide_text.csv'

# Load the dataset
# Assuming the tweet text is in a column named 'tweet'
suicide_text_df = pd.read_csv(suicide_text_path)

# Map 'class' to a numerical 'label'
suicide_text_df['label'] = suicide_text_df['class'].map({'suicide': 2, 'non-suicide': 0})

# Select only the 'text' and 'label' columns
suicide_text_df = suicide_text_df[['text', 'label']]

# print("suicide_text.csv")

# # Display the first few rows to verify
# print(suicide_text_df.head())

# ---- 

# sentiment_analysis_df: Sentiment140 data with labels 0 (negative) and 1 (positive)
# depressed_tweets_df: Depressive tweets data with label 2
# nondepressed_tweets_df : Non-Depressed tweets data with label 1
# suicide_text_df: Suicide-related text data with labels 0 (non-suicide) and 2 (suicide)

# # Combine all datasets into one DataFrame by concatenating them
# df = pd.concat([sentiment_analysis_df, depressed_tweets_df, nondepressed_tweets_df, suicide_text_df])

# # Shuffle the dataset
# df = df.sample(frac=1).reset_index(drop=True)

# Combine all datasets into one DataFrame by concatenating them
df = pd.concat([sentiment_analysis_df, depressed_tweets_df, nondepressed_tweets_df, suicide_text_df])

# Shuffle the dataset
df = df.sample(frac=1).reset_index(drop=True)

# Preprocessing
import re
import preprocessor as p
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download the required NLTK data
nltk.download('stopwords')
nltk.download('punkt')

json_file_path = 'data/english_contractions.json'
contractions = pd.read_json(json_file_path, typ='series').to_dict()

# Compile regular expression patterns
urlPattern = re.compile(r'https?://\S+|www\.\S+')
userPattern = re.compile(r'@(\w+)')
smileemoji = re.compile(r':\)')
sademoji = re.compile(r':\(')
neutralemoji = re.compile(r':\|')
lolemoji = re.compile(r'lol')

# Compile a regular expression pattern for contractions
c_re = re.compile('(%s)' % '|'.join(contractions.keys()))

def expand_contractions(text, c_re=c_re):
    def replace(match):
        return contractions[match.group(0)]
    return c_re.sub(replace, text)

def replace_urls(text):
    return re.sub(urlPattern, '<url>', text)

def replace_usernames(text):
    return re.sub(userPattern, '<user>', text)

def replace_emojis(text):
    text = re.sub(r'<3', '<heart>', text)
    text = re.sub(smileemoji, '<smile>', text)
    text = re.sub(sademoji, '<sadface>', text)
    text = re.sub(neutralemoji, '<neutralface>', text)
    text = re.sub(lolemoji, '<lolface>', text)
    return text

def remove_punctuation(text):
    symbols_re = re.compile('[^0-9a-z #+_]')
    return symbols_re.sub(' ', text)

def remove_non_alphanumeric(text):
    alphaPattern = re.compile("[^a-z0-9<>]")
    return re.sub(alphaPattern, ' ', text)

def add_spaces_for_slash(text):
    return re.sub(r'/', ' / ', text)

def preprocess_tweet(text):
    # Convert tweet to lowercase
    text = str(text).lower()

    # Remove URLs
    text = re.sub(urlPattern, '<url>', text)

    # Remove usernames
    text = re.sub(userPattern, '<user>', text)

    # Expand contractions
    text = expand_contractions(text)

    # Remove emojis
    text = replace_emojis(text)

    # Remove non-alphanumeric characters
    text = remove_non_alphanumeric(text)

    # Adding space on either side of '/' to separate words (After replacing URLs).
    text = add_spaces_for_slash(text)

    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = nltk.word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return ' '.join(filtered_sentence)

def clean_tweets(text):
    cleaned_tweets = []
    for text in text:
        # Preprocess the tweet
        text = preprocess_tweet(text)

        # Remove stopwords
        text = remove_stopwords(text)

        # Adding space on either side of '/' to separate words (After replacing URLs).
        text = add_spaces_for_slash(text)

        cleaned_tweets.append(text)

    return cleaned_tweets

# Apply the preprocessing to the 'text' column of the DataFrame
# This will create a new column 'processed_text' with the cleaned and preprocessed tweets
df['processed_text'] = df['text'].apply(lambda x: remove_stopwords(preprocess_tweet(x)))

# Verify the results
print(df[['text', 'processed_text', 'label']].head())
df.to_csv('processed_data.csv, index = False')