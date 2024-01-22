import streamlit as st
from transformers import AutoTokenizer, DistilBertForSequenceClassification
import torch
import re
from datetime import datetime
from tweety import Twitter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import altair as alt
import plotly.express as px
import lime
from lime.lime_text import LimeTextExplainer
from scipy.special import softmax


# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model_path = '/Users/nurfatinaqilah/Documents/streamlit-test/dataset/finals_distilbert_model'
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()

class TwitterUser:

    def __init__(self):
        # username: 'Haven39_'
        # password: 'FYPpurposes'
        self.app = Twitter("session")
        # Replace with your actual Twitter username and password
        self.app.sign_in("Haven39_", "FYPpurposes")
     
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
    ## First, check if 'tweet' is a dictionary and has a 'text' key
    if isinstance(tweet, dict) and 'text' in tweet:
        tweet_text = re.sub(r'http\S+', '', tweet['text'])
    else:
        # If 'tweet' is not a dictionary or doesn't have a 'text' key, handle the error
        tweet_text = "Invalid tweet data"

    if isinstance(tweet, dict) and 'text' in tweet and 'date' in tweet:
        # Process the tweet as expected
        tweet_text = re.sub(r'http\S+', '', tweet['text'])
        formatted_date = tweet['date'].strftime('%d-%m-%Y')
        # ... rest of the code ...
    else:
        # Handle the unexpected tweet format
        tweet_text = "Invalid tweet data"
        formatted_date = "Invalid date"

    # Map sentiment numbers to labels and colors
    sentiment_labels = {0: 'Normal/Negative/Non-Depressed', 1: 'Depressed', 2: 'Suicidal'}
    sentiment_colors = {0: '#1f77b4', 1: '#ff9999', 2: '#d62728'}  # light red for 1, red for 2, white for 0
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
urlPattern = re.compile(r'https?://\S+|www\.\S+http')
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
    # Convert tweet to lowercase
    text = str(text).lower()

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
    # Define sentiment_colors at the top of the twitter_app function
    sentiment_colors = {0: '#1f77b4', 1: '#ff9999', 2: '#d62728'}

    st.title("Twitter Sentiment Analysis")
    
    # Sentiment label explanations
    with st.expander("Sentiment Labels Explained"):
        st.write("""
            - **0**: Normal/Negative/Non-Depressed - These tweets are generally neutral or express negative sentiments but are not indicative of depression.
            - **1**: Depressed - These tweets suggest the user may be experiencing feelings of sadness or depression.
            - **2**: Suicidal - These tweets indicate that the user might be having suicidal thoughts or severe depression.
        """)
    
    tweets = []
    tweet_objects = []
    predictions = []  # Initialize predictions
    twitter_user = TwitterUser()
    username = st.text_input('Enter Twitter Username:')
    pages = st.number_input('Number of Tweet Pages to Fetch', min_value=1, value=1, step=1)

    fetch_tweets_button = st.button('Analyze Tweets')

    if fetch_tweets_button and username:
        tweet_objects = twitter_user.get_tweets(username, pages=pages)
        if tweet_objects:
            # Filter out tweets that are empty, contain only whitespace, or have images
            tweet_objects = [tweet for tweet in tweet_objects if tweet['text'].strip() and not tweet.get('has_media_field', False)]
            # Filter out tweets that are empty or contain only links
            tweet_objects = [tweet for tweet in tweet_objects if tweet['text'].strip() and 'http' not in tweet['text']]

            # Sort tweets by date in descending order (latest first)
            tweet_objects = sorted(tweet_objects, key=lambda x: x['date'], reverse=True)

            # Processing tweets and predictions
            preprocessed_tweets = [remove_stopwords(preprocess_tweet(tweet['text'])) for tweet in tweet_objects]
            predictions = predict_tweet_sentiment(preprocessed_tweets)

            # # Loop through each tweet and prediction to display them
            # for tweet, prediction in zip(tweet_objects, predictions):
            #     formatted_date, tweet_text, sentiment_label, sentiment_color = format_tweet_display(tweet, prediction)
            #     # Use columns to organize the output
            #     col1, col2, col3 = st.columns(3)
            #     with col1:
            #         st.markdown(f"**Date**: {formatted_date}")
            #     with col2:
            #         st.markdown(f"**Tweet**: {tweet_text}")
            #     with col3:
            #         st.markdown(f"<span style='color: {sentiment_color};'>**Sentiment**: {sentiment_label}</span>", unsafe_allow_html=True)

            #     st.markdown("---")  # Add a separator
                    
        # Visualization code goes here
        df_tweets = pd.DataFrame({
            'text': [tweet['text'] for tweet in tweet_objects if tweet['text'].strip()],
            'sentiment': predictions,
            'date': [tweet['date'] for tweet in tweet_objects if tweet['text'].strip()]
        })

        # # Sort DataFrame by date
        # df_tweets.sort_values('date', inplace=True)
        
        # Visualization containers
        # st.header("Visualizations")
        vis_col1, vis_col2 = st.columns(2)
        
        sentiment_colors = {0: '#1f77b4', 1: '#ff9999', 2: '#d62728'}
    
        with vis_col1:
            # Sentiment Distribution Pie Chart
            sentiment_counts = df_tweets['sentiment'].value_counts().reset_index()
            sentiment_counts.columns = ['sentiment', 'count']
            fig_pie = px.pie(
                sentiment_counts, 
                values='count', 
                names='sentiment', 
                title='Sentiment Distribution',
                color='sentiment',
                color_discrete_map=sentiment_colors  # Use the updated color mapping
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with vis_col2:
            # Tweet Volume Over Time Chart
            df_volume = df_tweets.groupby(df_tweets['date'].dt.strftime('%Y-%m')).size().reset_index(name='tweet_count')
            tweet_volume_chart = alt.Chart(df_volume).mark_bar().encode(
                x=alt.X('date:O', axis=alt.Axis(title='Date', labelAngle=0)),
                y=alt.Y('tweet_count:Q', axis=alt.Axis(title='Tweet Count', tickMinStep=1))
            ).properties(title='Tweet Volume Over Time')
            st.altair_chart(tweet_volume_chart, use_container_width=True)
        
        
        # Sample data preparation
        #Assume df_tweets is your DataFrame and it has 'date' and 'sentiment' columns
        df_tweets['date'] = pd.to_datetime(df_tweets['date'])
        df_tweets['sentiment'] = df_tweets['sentiment'].astype(int)  # Ensure sentiment is an integer type

        # Group by date and sentiment to count the tweets
        df_sentiment_time = df_tweets.groupby(
            [df_tweets['date'].dt.to_period('M'), 'sentiment']
        ).size().reset_index(name='count')

        # Convert Period to Timestamp for Altair
        df_sentiment_time['date'] = df_sentiment_time['date'].dt.to_timestamp()

        # Check the DataFrame structure
        print(df_sentiment_time.head())

        # Define color scale for sentiments
        color_scale = alt.Scale(domain=[0, 1, 2], range=['#1f77b4', '#ff9999', '#d62728'])

        # Create the area chart for sentiment distribution over time
        sentiment_distribution_chart = alt.Chart(df_sentiment_time).mark_area().encode(
            x='date:T',
            y='count:Q',
            color=alt.Color('sentiment:N', scale=color_scale),
            tooltip=['date:T', 'sentiment:N', 'count:Q']
        ).properties(
            title='Sentiment Distribution Over Time'
        )

        # To display the chart in a Streamlit app, you would use:
        st.altair_chart(sentiment_distribution_chart, use_container_width=True)

        # --------------
        # Sort DataFrame by date
        
        df_tweets = df_tweets.sort_values('date')
        
        # Create the point chart for individual tweets
        points = alt.Chart(df_tweets).mark_point().encode(
            x=alt.X('date:T', title='Date'),
            y=alt.Y('sentiment:N', title='Sentiment Label'),
            color=alt.Color('sentiment:N', scale=color_scale, legend=alt.Legend(title="Sentiment")),
            tooltip=['date:T', 'sentiment:N', 'text:N']
        ).properties(
            title='Sentiment Label Over Time'
        )

        # Create the line chart to connect points in chronological order
        # The line color is not encoded by sentiment here to ensure a continuous line across different sentiments
        lines = alt.Chart(df_tweets).mark_line(interpolate='monotone').encode(
            x='date:T',
            y='sentiment:N'
        )

        # Combine the charts to overlay the points on the lines
        combined_chart = points + lines

        # Add interactive zoom and pan
        zoom = alt.selection_interval(bind='scales', encodings=['x'])

        # Apply zoom to your chart
        combined_chart = combined_chart.add_selection(
            zoom
        )

        # Display the combined chart
        st.altair_chart(combined_chart, use_container_width=True)

        # # Ensure 'date' column is datetime and sort by date
        # df_tweets['date'] = pd.to_datetime(df_tweets['date'], errors='coerce')
        # df_tweets = df_tweets.dropna(subset=['date']).sort_values('date')

        # # Define color scale for sentiments
        # color_scale = alt.Scale(domain=[0, 1, 2], range=['#1f77b4', '#ff9999', '#d62728'])
        
        # # Group by month and sentiment, then reset index to use in Altair
        # df_sentiment_time = df_tweets.groupby([df_tweets['date'].dt.to_period('M'), 'sentiment']).size().unstack(fill_value=0).reset_index()

        # # Convert Period to Timestamp for Altair
        # df_sentiment_time['date'] = df_sentiment_time['date'].dt.to_timestamp()

        # # Melt the dataframe to have 'date', 'sentiment', and 'count' columns
        # df_sentiment_time = df_sentiment_time.melt(id_vars=['date'], var_name='sentiment', value_name='count')

        # # Convert sentiments to string type for mapping
        # df_sentiment_time['sentiment'] = df_sentiment_time['sentiment'].astype(str)

        # # Create the area chart
        # sentiment_distribution_chart = alt.Chart(df_sentiment_time).mark_area().encode(
        #     x='date:T',
        #     y='count:Q',
        #     color=alt.Color('sentiment:N', scale=color_scale),
        #     tooltip=['date:T', 'sentiment:N', 'count:Q']
        # ).properties(
        #     title='Sentiment Distribution Over Time'
        # )

        # # Display the sentiment distribution chart
        # st.altair_chart(sentiment_distribution_chart, use_container_width=True)

        # # Create the point chart for individual tweets
        # points = alt.Chart(df_tweets).mark_point().encode(
        #     x=alt.X('date:T', title='Date'),
        #     y=alt.Y('sentiment:N', title='Sentiment Label'),
        #     color=alt.Color('sentiment:N', scale=color_scale),
        #     tooltip=['date:T', 'sentiment:N', 'text:N']
        # ).properties(
        #     title='Sentiment Label Over Time'
        # )

        # # Create the line chart to connect points with smooth lines
        # lines = alt.Chart(df_tweets).mark_line(interpolate='monotone').encode(
        #     x='date:T',
        #     y='sentiment:N',
        #     color=alt.Color('sentiment:N', scale=color_scale)
        # )

        # # Combine the charts to overlay the points on the lines
        # combined_chart = points + lines

        # # Add interactive zoom and pan
        # zoom = alt.selection_interval(bind='scales', encodings=['x'])

        # # Apply zoom to your chart
        # combined_chart = combined_chart.add_selection(
        #     zoom
        # )

        # # Display the combined chart
        # st.altair_chart(combined_chart, use_container_width=True)



        # # Before grouping by date, ensure the 'date' column is a datetime object.
        # df_tweets['date'] = pd.to_datetime(df_tweets['date'], errors='coerce')
        # df_tweets.dropna(subset=['date'], inplace=True)  # Drop rows where 'date' could not be converted

        # # Now, continue with your groupby operation
        # df_sentiment_time = df_tweets.groupby([df_tweets['date'].dt.to_period('M'), 'sentiment']).size().unstack(fill_value=0).reset_index()
        # df_sentiment_time['date'] = df_sentiment_time['date'].dt.to_timestamp()  # Convert Period to Timestamp for Altair

        # # Melt the dataframe to have 'date', 'sentiment', and 'count' columns
        # df_sentiment_time = df_sentiment_time.melt(id_vars=['date'], var_name='sentiment', value_name='count')

        # # Convert sentiments to string type for mapping
        # df_sentiment_time['sentiment'] = df_sentiment_time['sentiment'].astype(str)
        # # Sort DataFrame by date
        # df_tweets.sort_values('date', inplace=True)

        # # Define color mapping for sentiments
        # color_scale = alt.Scale(domain=[0, 1, 2], range=['#1f77b4', '#ff9999', '#d62728'])

        # # Plot the sentiment distribution over time
        # sentiment_distribution_chart = alt.Chart(df_sentiment_time).mark_area().encode(
        #     x='date:T',
        #     y='count:Q',
        #     color=alt.Color('sentiment:N', scale=alt.Scale(domain=[0, 1, 2], range=[sentiment_colors[0], sentiment_colors[1], sentiment_colors[2]])),
        #     tooltip=['date:T', 'sentiment:N', 'count:Q']
        # ).properties(
        #     title='Sentiment Distribution Over Time'
        # )
        # st.altair_chart(sentiment_distribution_chart, use_container_width=True)
        
        # # Create the point chart for individual tweets
        # points = alt.Chart(df_tweets).mark_point().encode(
        #     x=alt.X('date:T', title='Date'),
        #     y=alt.Y('sentiment:N', title='Sentiment Label'),
        #     color=alt.Color('sentiment:N', scale=color_scale, legend=alt.Legend(title="Sentiment")),
        #     tooltip=['date:T', 'sentiment:N', 'text:N']
        # ).properties(
        #     title='Sentiment Label Over Time'
        # )

        # # Create the line chart to connect points in chronological order
        # # The line color is not encoded by sentiment here to ensure a continuous line across different sentiments
        # lines = alt.Chart(df_tweets).mark_line(interpolate='monotone').encode(
        #     x='date:T',
        #     y='sentiment:N'
        # )

        # # Combine the charts to overlay the points on the lines
        # combined_chart = points + lines

        # # Display the combined chart
        # st.altair_chart(combined_chart, use_container_width=True)
        
        # # Add interactive zoom and pan
        # zoom = alt.selection_interval(bind='scales', encodings=['x'])

        # # Apply zoom to your chart
        # chart = alt.Chart(df_tweets).mark_line().encode(
        #     x='date:T',
        #     y='sentiment:N',
        #     color='sentiment:N'
        # ).add_params(
        #     zoom
        # )

        # st.altair_chart(chart, use_container_width=True)
        
        # Add a new column for tweet length
        df_tweets['length'] = df_tweets['text'].apply(lambda x: len(x))

        # Visualize tweet length distribution
        fig_length = px.box(
            df_tweets, 
            title="Tweet Length Distribution",
            x='sentiment', 
            y='length', 
            color='sentiment',
            category_orders={"sentiment": [0, 1, 2]},  # Add this line to specify the order
            color_discrete_map=sentiment_colors,  # Use the defined color mapping
            labels={'sentiment': 'Sentiment', 'length': 'Tweet Length'}
        )
        fig_length.update_traces(marker=dict(size=2))  # Optional: Adjust marker size for better visibility
        st.plotly_chart(fig_length)

        # Word Clouds for Each Sentiment
        wordcloud_col1, wordcloud_col2, wordcloud_col3 = st.columns(3)
        for sentiment, col in zip(range(3), [wordcloud_col1, wordcloud_col2, wordcloud_col3]):
            with col:
                st.subheader(f"Word Cloud for Sentiment {sentiment}")
                words = ' '.join(df_tweets[df_tweets['sentiment'] == sentiment]['text'])
                
                # Check if there are words available for the word cloud
                if words.strip():  # This checks if the string is not empty
                    wordcloud = WordCloud(width=300, height=300, background_color='white').generate(words)
                    fig, ax = plt.subplots()
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                else:
                    st.write("No words available for word cloud.")


        # Sentiment Predictions at the bottom
        st.header("Sentiment Predictions")
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
                        
# The main app execution
if __name__ == "__main__":
    twitter_app()
#         # Generate Word Cloud for each sentiment
#         for sentiment, group in df_tweets.groupby('sentiment'):
#             st.subheader(f"Word Cloud for Sentiment {sentiment}")
#             words = ' '.join(group['text'])
#             wordcloud = WordCloud(width=800, height=400).generate(words)
#             plt.figure(figsize=(10, 5))
#             plt.imshow(wordcloud, interpolation='bilinear')
#             plt.axis('off')
#             st.pyplot(plt)
            
#         # Sentiment Distribution Pie Chart
#         sentiment_counts = df_tweets['sentiment'].value_counts().reset_index()
#         sentiment_counts.columns = ['sentiment', 'count']
#         fig = px.pie(sentiment_counts, values='count', names='sentiment', title='Sentiment Distribution')
#         st.plotly_chart(fig)
            
#         # Sort DataFrame by date
#         df_tweets.sort_values('date', inplace=True)

#         # Define color mapping for sentiments
#         color_scale = alt.Scale(domain=[0, 1, 2], range=['#909191', '#fc8181', 'red'])

#         # Create the point chart for individual tweets
#         points = alt.Chart(df_tweets).mark_point().encode(
#             x=alt.X('date:T', title='Date'),
#             y=alt.Y('sentiment:N', title='Sentiment Label'),
#             color=alt.Color('sentiment:N', scale=color_scale, legend=alt.Legend(title="Sentiment")),
#             tooltip=['date:T', 'sentiment:N', 'text:N']
#         )

#         # Create the line chart to connect points in chronological order
#         # The line color is not encoded by sentiment here to ensure a continuous line across different sentiments
#         lines = alt.Chart(df_tweets).mark_line().encode(
#             x='date:T',
#             y='sentiment:N'
#         )

#         # Combine the charts to overlay the points on the lines
#         combined_chart = points + lines

#         # Display the combined chart
#         st.altair_chart(combined_chart, use_container_width=True)

#         #Tweet Volume Over Time Chart
#         df_volume = df_tweets.groupby(df_tweets['date'].dt.strftime('%Y-%m')).size().reset_index(name='tweet_count')

#         # Create the bar chart
#         tweet_volume_chart = alt.Chart(df_volume).mark_bar().encode(
#             x=alt.X('date:O', axis=alt.Axis(title='Date', labelAngle=0)),  # Ordinal scale for bar chart
#             y=alt.Y('tweet_count:Q', axis=alt.Axis(title='Tweet Count', tickMinStep=1)),  # Set minimum step for ticks to 1
#         ).properties(
#         title='Tweet Volume Over Time'
#         )

#         st.altair_chart(tweet_volume_chart, use_container_width=True)

#         # Trying new chart
#         # Sentiment Distribution Over Time
#         # First, let's create a sentiment distribution over time
#         df_sentiment_time = df_tweets.groupby([df_tweets['date'].dt.to_period('M'), 'sentiment']).size().unstack(fill_value=0).reset_index()

#         # Melt the dataframe to have 'date', 'sentiment', and 'count' columns
#         df_sentiment_time = df_sentiment_time.melt(id_vars=['date'], var_name='sentiment', value_name='count')

#         # Plot the sentiment distribution over time
#         sentiment_distribution_chart = alt.Chart(df_sentiment_time).mark_area().encode(
#             x='date:T',
#             y='count:Q',
#             color='sentiment:N',
#             tooltip=['date:T', 'sentiment:N', 'count:Q']
#         ).properties(
#             title='Sentiment Distribution Over Time'
#         )

#         st.altair_chart(sentiment_distribution_chart, use_container_width=True)

        
# # The main app execution
# if __name__ == "__main__":
#     twitter_app()