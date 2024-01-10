import hashlib
import sqlite3
from tweety import Twitter  # Import the tweety library

# Your Twitter credentials
TWITTER_USERNAME = "Haven39_"
TWITTER_PASSWORD = "FYPpurposes"

class TwitterUser:
    def __init__(self):
        self.app = Twitter("session")
        self.logged_in_user_info = None

    # def login(self, username, password):
    #     if not username or not password:
    #         raise ValueError("Username and password are required.")
        
    #     try:
    #         self.app.sign_in(username, password)
    #         self.logged_in_user_info = self.app.user
    #         return True
    #     except Exception as e:
    #         print(f"Login failed. Error: {str(e)}")
    #         return False

    def login(self):
        try:
            self.app.login_with_access_token(TWITTER_USERNAME, TWITTER_PASSWORD)
            self.logged_in_user_info = self.app.user
            return True
        except Exception as e:
            print(f"Login failed. Error: {str(e)}")
            return False
        
    def get_user_info(self, target_username):
        return self.app.get_user_info(target_username)

    def get_tweets(self, username, pages=5):
        return self.app.get_tweets(username=username, pages=pages)

def get_encrypted_password(password):
    sha256 = hashlib.sha256()
    sha256.update(password.encode())
    return sha256.hexdigest()

# DATABASE
def create_tweets_table_if_not_exists(conn):
    # Create a table to store tweet text if it doesn't already exist
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tweets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tweet_text TEXT
        )
    ''')
    conn.commit()

def store_tweets_in_database(conn, tweets):
    # Store the tweets in the database
    cursor = conn.cursor()
    for tweet in tweets:
        cursor.execute("INSERT INTO tweets (tweet_text) VALUES (?)", (tweet,))
    conn.commit()

def connect_to_database():
    # Connect to the SQLite database
    conn = sqlite3.connect('tweets.db')
    return conn

# ...

if __name__ == "__main__":
    # Example usage:
    twitter_user = TwitterUser()
    
    # Input for Twitter credentials
    username = input("Enter your Twitter username: ")
    password = input("Enter your Twitter password: ")

    if twitter_user.login(username, password):
        target_username = input("Enter the Twitter username to extract tweets from: ")

        # Input for the maximum number of tweets to retrieve
        max_tweets = int(input("Enter the maximum number of tweets to retrieve (e.g., 100): "))

        tweets = twitter_user.get_tweets(target_username, pages=5)

        # Limit the number of tweets to the specified maximum
        tweets = tweets[:max_tweets]

        # Connect to the database
        conn = connect_to_database()

        # Create the tweets table if it doesn't exist
        create_tweets_table_if_not_exists(conn)

        # Store the tweets in the database
        store_tweets_in_database(conn, [tweet.text for tweet in tweets])

        # Close the database connection when done
        conn.close()
