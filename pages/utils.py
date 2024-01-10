import re
import torch
from transformers import AutoTokenizer, DistilBertForSequenceClassification
from wordcloud import WordCloud
from tweety import Twitter

class TwitterUser:
    def __init__(self):
        self.app = Twitter("session")
        self.logged_in_user_info = None

    def login(self, username, password):
        if not username or not password:
            raise ValueError("Username and password are required.")
        
        try:
            # Use plain text password
            self.app.sign_in(username, password)
            self.logged_in_user_info = self.app.user
            return True
        except Exception as e:
            st.error(f"Login failed. Error: {str(e)}")
            return False

    def get_user_info(self, target_username):
        return self.app.get_user_info(target_username)

    def get_tweets(self, username, pages=1):
        # Fetch a specific number of pages of tweets
        try:
            tweets = self.app.get_tweets(username=username, pages=pages)
            return tweets
        except Exception as e:
            raise Exception(f"Error retrieving tweets: {str(e)}")

def predict_tweet_sentiment(tweets, model):
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
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
    sentiment_colors = {0: '#f1f2eb', 1: '#898df5', 2: 'red'}  # light red for 1, red for 2, white for 0
    sentiment_label = sentiment_labels[sentiment]
    sentiment_color = sentiment_colors[sentiment]

    return formatted_date, tweet_text, sentiment_label, sentiment_color

def generate_word_cloud(text_data):
    # Generate a word cloud from the provided text data
    text = " ".join(text_data)
    wordcloud = WordCloud(width=800, height=400).generate(text)
    return wordcloud