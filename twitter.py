from tweety import Twitter
import getpass
import hashlib
from tweety.types.twDataTypes import SelfThread

class TwitterUser:
    def __init__(self):
        self.app = Twitter("session")
        self.logged_in_user_info = None

    def login(self, username, password):
        if not username or not password:
            raise ValueError("Username and password are required.")
        
        try:
            self.app.sign_in(username, password)
            self.logged_in_user_info = self.app.user
            print("Login successful.")
        except Exception as e:
            print(f"Login failed. Error: {str(e)}")

    def get_logged_in_user_info(self):
        return self.logged_in_user_info

    def get_user_info(self, target_username):
        return self.app.get_user_info(target_username)

    def print_user_info(self, user_info):
        print(f"ID: {user_info.id}")
        print(f"Username: {user_info.username}")
        print(f"Name: {user_info.name}")
        print(f"Followers: {user_info.followers_count}")
        print(f"Following: {user_info.friends_count}")
        print(f"Verified: {user_info.verified}")
        print(f"Profile Image URL: {user_info.profile_image_url_https}")

    def get_tweets(self, user_info, pages=5):
        return self.app.get_tweets(username=user_info, pages=pages)

    def process_tweets(self, tweets):
        for tweet in tweets:
            if isinstance(tweet, SelfThread):
                for tw in tweet.tweets:
                    if not tw.is_retweet:
                        print(tw.text)
                    if not tw.is_quoted:
                        print(tw.text)
                    if not tw.is_reply:
                        print(tw.text)
            else:
                if not tweet.is_retweet:
                    print(tweet.text)

def get_encrypted_password(password):
    # You can choose your preferred encryption method here.
    # Here, we're using SHA-256 as an example.
    sha256 = hashlib.sha256()
    sha256.update(password.encode())
    encrypted_password = sha256.hexdigest()
    return encrypted_password

def main():
    twitter_user = TwitterUser()

    username = input("Enter your Twitter username: ")
    password = getpass.getpass("Enter your Twitter password (hidden): ")
    encrypt_password = input("Encrypt password? (y/n): ").strip().lower()

    if encrypt_password == 'y':
        encrypted_password = get_encrypted_password(password)
    else:
        encrypted_password = password

    try:
        twitter_user.login(username, encrypted_password)
    except ValueError as ve:
        print(f"Error: {str(ve)}")
        return

    if twitter_user.get_logged_in_user_info():
        target_username = input("Enter the Twitter username to analyze: ")

        twitter_user_info = twitter_user.get_logged_in_user_info()
        target_user_info = twitter_user.get_user_info(target_username)

        print("Logged-In User Info:")
        twitter_user.print_user_info(twitter_user_info)
        print("\nTarget User Info:")
        twitter_user.print_user_info(target_user_info)

        tweets = twitter_user.get_tweets(target_user_info, pages=5)
        twitter_user.process_tweets(tweets)

if __name__ == "__main__":
    main()
