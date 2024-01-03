import pandas as pd
import os

severity_depression_path = '/Users/nurfatinaqilah/Documents/streamlit-test/data/severity_depression.csv'

# Read the CSV file into a DataFrame
# DataFrame will contain your data with the specified column names and encoding.
severity_depression_df = pd.read_csv(severity_depression_path)

# # First 5 data of the dataset
# print(severity_depression_df.head())

# # Last 5 data of the dataset
# print(severity_depression_df.tail())

# # print the shape of the dataset
# print(severity_depression_df.shape)

# # To find any NaN values
# print(severity_depression_df.isna().sum())

# # To find any NULL values
# print(severity_depression_df.isnull().sum())

# # info of the data
# print(severity_depression_df.info())

# print(severity_depression_df["label"].value_counts())

# Preprocessing
import re #RegEx
import preprocessor as p 
import nltk 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download the required NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# JSON file containing English contractions (like "don't" for "do not") and converting it into a dictionary
json_file_path = '/Users/nurfatinaqilah/Documents/streamlit-test/data/english_contractions.json'
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

# Expands contractions in the text using the contractions dictionary. 
def expand_contractions(text, c_re=c_re):
    def replace(match):
        return contractions[match.group(0)]
    return c_re.sub(replace, text)

# # Replace URLs and usernames with placeholder texts (<url> and <user>, respectively).
# def replace_urls(text):
#     return re.sub(urlPattern, '<url>', text)

# def replace_usernames(text):
#     return re.sub(userPattern, '<user>', text)

# Replaces specific emoji patterns with text placeholders (like <smile>, <sadface>, etc.).
def replace_emojis(text):
    text = re.sub(r'<3', '<heart>', text)
    text = re.sub(smileemoji, '<smile>', text)
    text = re.sub(sademoji, '<sadface>', text)
    text = re.sub(neutralemoji, '<neutralface>', text)
    text = re.sub(lolemoji, '<lolface>', text)
    return text

# def remove_punctuation(text):
#     symbols_re = re.compile('[^0-9a-z #+_]')
#     return symbols_re.sub(' ', text)

# def remove_non_alphanumeric(text):
#     alphaPattern = re.compile("[^a-z0-9<>]")
#     return re.sub(alphaPattern, ' ', text)

# # Adds spaces around slashes to separate words joined by slashes.
# def add_spaces_for_slash(text):
#     return re.sub(r'/', ' / ', text)

# comprehensive function that applies all the above preprocessing steps to a single text entry.
# To preprocess a single tweet.
def preprocess_tweet(text):
    # # Remove URLs
    # text = re.sub(urlPattern, '<url>', text)

    # # Remove usernames
    # text = re.sub(userPattern, '<user>', text)

    # Expand contractions
    text = expand_contractions(text)
    
    # Convert tweet to lowercase
    text = str(text).lower()

    # Remove emojis
    text = replace_emojis(text)

    # # Remove non-alphanumeric characters
    # text = remove_non_alphanumeric(text)

    # # Adding space on either side of '/' to separate words (After replacing URLs).
    # text = add_spaces_for_slash(text)

    return text

# removes common words (stopwords) that add little semantic value to the text, using NLTK’s list of English stopwords.
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

        cleaned_tweets.append(text)

    return cleaned_tweets

# Apply the preprocessing to the 'text' column of the DataFrame
# This will create a new column 'processed_text' with the cleaned and preprocessed tweets
# used here as part of a lambda function within df.apply(), allowing it to preprocess each tweet in the DataFrame one by one
severity_depression_df['processed_text'] = severity_depression_df['text'].apply(lambda x: remove_stopwords(preprocess_tweet(x)))

# Verify the results
print(severity_depression_df[['text', 'processed_text', 'label']].head())

severity_depression_df.to_csv('/Users/nurfatinaqilah/Documents/streamlit-test/dataset/severity_processed3.csv', index = False)
