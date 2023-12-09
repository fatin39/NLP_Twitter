# tweet is a twitter web scraper - kinda like a robot that tries to access the twitter
from tweety import Twitter
from tweety.types.twDataTypes import SelfThread

# you need to signup first
username="haven39_"
password="FYP_2023"

app = Twitter("session")
app.sign_in(username, password)

# check if sign up correct
# print(app.user)

# get tweets
target_username='kencanayoo'
user = app.get_user_info(target_username)
print(user.followers_count, user.friends_count, user.profile_image_url_https)
tweets = app.get_tweets(username=user, pages=5)

for tweet in tweets:
    if(isinstance(tweet, SelfThread)):
        for tw in tweet.tweets:
            if not tw.is_retweet:
                print(tw.text)
    else:    
        if not tweet.is_retweet:
            print(tweet.text)