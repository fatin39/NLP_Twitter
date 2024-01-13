import streamlit as st
from transformers import AutoTokenizer, DistilBertForSequenceClassification, DistilBertTokenizer
import torch
import re
# from datetime import datetime
from tweety import Twitter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import altair as alt
from vega_datasets import data


# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model_path = '/Users/nurfatinaqilah/Documents/streamlit-test/dataset/final_distilbert_model'
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()

class TwitterUser:
    def __init__(self):
        self.app = Twitter("session")

    # def get_tweets(self, username, pages=1):
    #     tweets = self.app.get_tweets(username=username, pages=pages)
    #     return tweets
    
    # def get_tweets(self, username, pages=1):
    #     # Assuming the 'get_tweets' method returns a list of tweet objects
    #     # and each tweet object has a 'text' attribute and a 'date' attribute
    #     tweets = self.app.get_tweets(username=username, pages=pages)
    #     return [{'text': tweet.text, 'date': tweet.date} for tweet in tweets]
    def get_tweets(self, username, pages=1):
    # Directly return the list of dictionaries without transforming it
        return self.app.get_tweets(username=username, pages=pages)

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
    
    # Sentiment label explanations
    with st.expander("Sentiment Labels Explained"):
        st.write("""
            - **0**: Normal/Negative/Non-Depressed - These tweets are generally neutral or express negative sentiments but are not indicative of depression.
            - **1**: Depressed - These tweets suggest the user may be experiencing feelings of sadness or depression.
            - **2**: Suicidal - These tweets indicate that the user might be having suicidal thoughts or severe depression.
        """)
    # Initialize 'tweets' to ensure it's in the proper scope
    tweets = []
    tweet_objects = []
    predictions = []  # Initialize predictions
    twitter_user = TwitterUser()
    username = st.text_input('Enter Twitter Username:')
    pages = st.number_input('Number of Tweet Pages to Fetch', min_value=1, value=1, step=1)

    # Checkbox for showing visualizations - define at the top
    show_visualizations = st.checkbox("Show Visualizations")

    fetch_tweets_button = st.button('Analyze Tweets')

    if fetch_tweets_button and username:
        tweet_objects = twitter_user.get_tweets(username, pages=pages)
        if tweet_objects:
            # Processing tweets and predictions
            preprocessed_tweets = [remove_stopwords(preprocess_tweet(tweet['text'])) for tweet in tweet_objects]
            predictions = predict_tweet_sentiment(preprocessed_tweets)

            # Loop through each tweet and prediction to display them
            for tweet, prediction in zip(tweet_objects, predictions):
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
        
        preprocessed_tweets = [remove_stopwords(preprocess_tweet(tweet['text'])) for tweet in tweet_objects]
        predictions = predict_tweet_sentiment(preprocessed_tweets)

        # Create a DataFrame for visualization
        df_tweets = pd.DataFrame({
            'text': [tweet['text'] for tweet in tweet_objects],
            'sentiment': predictions,
            'date': [tweet['date'] for tweet in tweet_objects]
        })
        
        # Organize tweets by sentiment
        sentiment_tweets = {0: [], 1: [], 2: []}
        for tweet, pred in zip(preprocessed_tweets, predictions):
            sentiment_tweets[pred].append(tweet)

        if show_visualizations:
            for sentiment, tweets in sentiment_tweets.items():
                if tweets:  # Only generate a word cloud if there are tweets
                    st.subheader(f"Word Cloud for Sentiment {sentiment}")
                    wordcloud = generate_word_cloud(tweets)
                    plt.figure(figsize=(10, 5))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis("off")
                    st.pyplot(plt)
            
            # Sentiment Distribution Chart
            # Sort DataFrame by date
            df_tweets = df_tweets.sort_values('date')

            # Create the point chart for the individual tweets
            points = alt.Chart(df_tweets).mark_point().encode(
                x=alt.X('date:T', axis=alt.Axis(format='%Y-%m-%d'), title='Date'),
                y=alt.Y('sentiment:N', title='Sentiment Label'),
                color=alt.Color('sentiment:N', legend=alt.Legend(title="Sentiment")),
                tooltip=['date:T', 'sentiment:N', 'text:N']
            )

            # Create the line chart to connect points
            lines = alt.Chart(df_tweets).mark_line().encode(
                x='date:T',
                y='sentiment:N',
                color='sentiment:N'
            )

            # Combine the charts
            combined_chart = points + lines

            # Display the chart
            st.altair_chart(combined_chart, use_container_width=True)
                
            import plotly.express as px

            #Sentiment Distribution Chart
            sentiment_counts = df_tweets['sentiment'].value_counts().reset_index()
            sentiment_counts.columns = ['sentiment', 'count']
            sentiment_chart = alt.Chart(sentiment_counts).mark_bar().encode(
                x=alt.X('sentiment:N', title='Sentiment'),
                y=alt.Y('count:Q', title='Number of Tweets'),
                color='sentiment:N',
                tooltip=['sentiment', 'count']
            ).interactive()
            st.altair_chart(sentiment_chart, use_container_width=True)    
                            
# The main app execution
if __name__ == "__main__":
    twitter_app()