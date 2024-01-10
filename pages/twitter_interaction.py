import streamlit as st
from transformers import AutoTokenizer, DistilBertForSequenceClassification, DistilBertTokenizer
import torch
import re
from datetime import datetime
from tweety import Twitter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model_path = '/Users/nurfatinaqilah/Documents/streamlit-test/dataset/final_distilbert_model'
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()

class TwitterUser:
    def __init__(self):
        self.app = Twitter("session")

    def get_tweets(self, username, pages=1):
        tweets = self.app.get_tweets(username=username, pages=pages)
        return tweets

def predict_tweet_sentiment(tweets):
    encodings = tokenizer.batch_encode_plus(
        tweets,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    with torch.no_grad():
        inputs = encodings['input_ids']
        attention_mask = encodings['attention_mask']
        outputs = model(inputs, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=1)
        return predictions.numpy()

def format_tweet_display(tweet, sentiment):
    # Remove links and media from tweet text
    tweet_text = re.sub(r'http\S+', '', tweet.text)

    # Format the date to 'dd-mm-yyyy'
    formatted_date = tweet.date.strftime('%d-%m-%Y')

    # Map sentiment numbers to labels and colors
    sentiment_labels = {0: 'Normal/Negative/Non-Depressed', 1: 'Depressed', 2: 'Suicidal'}
    sentiment_colors = {0: 'white', 1: '#fabebe', 2: 'red'}  # light red for 1, red for 2, white for 0
    sentiment_label = sentiment_labels[sentiment]
    sentiment_color = sentiment_colors[sentiment]

    return formatted_date, tweet_text, sentiment_label, sentiment_color


def generate_word_cloud(text_data):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(text_data))
    return wordcloud

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

def remove_punctuation(text):
    symbols_re = re.compile('[^0-9a-z #+_]')
    return symbols_re.sub(' ', text)

def remove_non_alphanumeric(text):
    alphaPattern = re.compile("[^a-z0-9<>]")
    return re.sub(alphaPattern, ' ', text)

# Adds spaces around slashes to separate words joined by slashes.
def add_spaces_for_slash(text):
    return re.sub(r'/', ' / ', text)

# comprehensive function that applies all the above preprocessing steps to a single text entry.
# To preprocess a single tweet.
def preprocess_tweet(text):
    # # Convert tweet to lowercase
    # text = str(text).lower()

    # Remove URLs
    text = re.sub(urlPattern, '', text)

    # Remove usernames
    text = re.sub(userPattern, '', text)

    # Expand contractions
    text = expand_contractions(text)

    # Remove emojis
    text = replace_emojis(text)

    # Remove non-alphanumeric characters
    text = remove_non_alphanumeric(text)

    # Adding space on either side of '/' to separate words (After replacing URLs).
    text = add_spaces_for_slash(text)

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

def twitter_app():
    st.title("Twitter Sentiment Analysis")
    twitter_user = TwitterUser()
    username = st.text_input('Enter Twitter Username:')
    pages = st.number_input('Number of Tweet Pages to Fetch', min_value=1, value=1, step=1)

    # Checkbox for showing visualizations - define at the top
    show_visualizations = st.checkbox("Show Visualizations")

    fetch_tweets_button = st.button('Analyze Tweets')

    if fetch_tweets_button and username:
        tweets = twitter_user.get_tweets(username, pages=pages)
        preprocessed_tweets = [remove_stopwords(preprocess_tweet(tweet.text)) for tweet in tweets]

        predictions = predict_tweet_sentiment(preprocessed_tweets)
        
        for tweet, prediction in zip(tweets, predictions):
            formatted_date, tweet_text, sentiment_label, sentiment_color = format_tweet_display(tweet, prediction)
            
            # Use columns to organize the output
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Date**: {formatted_date}")
            with col2:
                st.markdown(f"**Tweet**: {tweet_text}")
            with col3:
                st.markdown(f"<span style='color: {sentiment_color};'>**Sentiment**: {sentiment_label}</span>", unsafe_allow_html=True)

            st.markdown("---")  # Add a separator
        
        if show_visualizations:
           # Generate and display WordCloud
            st.subheader("Word Cloud for All Tweets")
            all_tweets_text = [tweet.text for tweet in tweets]
            wordcloud = generate_word_cloud(all_tweets_text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            st.pyplot(plt)

# The main app execution
if __name__ == "__main__":
    twitter_app()