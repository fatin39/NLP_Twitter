import streamlit as st
from tweety import Twitter
import hashlib

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
            return True
        except Exception as e:
            st.error(f"Login failed. Error: {str(e)}")
            return False

    def get_user_info(self, target_username):
        return self.app.get_user_info(target_username)

    def get_tweets(self, username, pages=5):
        return self.app.get_tweets(username=username, pages=pages)

def get_encrypted_password(password):
    sha256 = hashlib.sha256()
    sha256.update(password.encode())
    return sha256.hexdigest()

def app():
    st.title("Twitter Data Retrieval")
    twitter_user = TwitterUser()

    # Form for login and analyzing tweets
    with st.form(key='twitter_interaction_form'):
        # Login inputs
        username = st.text_input('Twitter Username')
        password = st.text_input('Password', type='password')
        login_button = st.form_submit_button(label='Login')

        # Target user input and button
        target_username = st.text_input("Enter the Twitter username to analyze:")
        analyze_button = st.form_submit_button('Analyze Tweets')

    if login_button:
        encrypted_password = get_encrypted_password(password)
        if twitter_user.login(username, encrypted_password):
            st.success("Login successful.")
            display_logged_in_user_info(twitter_user)

    if analyze_button and target_username:
        display_target_user_info(twitter_user, target_username)
def display_logged_in_user_info(twitter_user):
    user_info = twitter_user.logged_in_user_info
    if user_info:
        st.subheader("Logged-In User Info:")
        st.image(user_info.profile_image_url_https, width=100)
        st.write(f"Username: {user_info.username}")
        st.write(f"Name: {user_info.name}")
        st.write(f"Followers: {user_info.followers_count}")
        st.write(f"Following: {user_info.friends_count}")
        st.write(f"Total Tweets: {user_info.statuses_count}")

def display_target_user_info(twitter_user, target_username):
    try:
        target_user_info = twitter_user.get_user_info(target_username)
        if target_user_info:
            st.write("Target User Info:")
            st.image(target_user_info.profile_image_url_https, width=100)
            st.write(f"Username: {target_user_info.username}")
            st.write(f"Name: {target_user_info.name}")
            st.write(f"Followers: {target_user_info.followers_count}")
            st.write(f"Following: {target_user_info.friends_count}")
            st.write(f"Total Tweets: {target_user_info.statuses_count}")

            tweets = twitter_user.get_tweets(target_username, pages=5)
            st.subheader("Tweets")
            for tweet in tweets:
                st.write(tweet.text)
        else:
            st.error("Failed to retrieve target user info.")
    except Exception as e:
        st.error(f"Error retrieving target user info: {e}")
 
    tweets = twitter_user.get_tweets(target_username, pages=5)
    st.subheader("Tweets")
    for tweet in tweets:
        st.write(tweet.text)

if __name__ == "__main__":
    app()
