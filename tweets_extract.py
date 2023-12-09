from tweety import Twitter

app = Twitter("session")
app.sign_in(username, password)
target_username = "elonmusk"

user = app.get_user_info(target_username)
all_tweets = app.get_tweets(user)

for tweet in all_tweets:
    print(tweet)
