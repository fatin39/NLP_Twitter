import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud


# Load the preprocessed data
df = pd.read_csv('data/severity_analysis_results.csv')


# Ensure all entries in 'processed_text' are strings
df['processed_text'] = df['processed_text'].astype(str)

# Function to plot the distribution of severity levels
def plot_severity_distribution(df):
    # Count the occurrences of each severity level
    severity_counts = df['severity'].value_counts()

    # Create a bar plot
    sns.barplot(x=severity_counts.index, y=severity_counts.values)

    # Adding labels and title
    plt.xlabel('Severity Level')
    plt.ylabel('Count')
    plt.title('Distribution of Severity Levels')
    plt.xticks(rotation=45)  # Rotating the x-labels for better readability

    # Show the plot
    plt.show()

# Function for word cloud
def generate_word_cloud(df, severity):
    text = " ".join(review for review in df[df['severity'] == severity]['processed_text'])
    wordcloud = WordCloud(background_color="white").generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f'Word Cloud for {severity} Severity')
    plt.show()

# Function for histogram of text length
def plot_text_length_histogram(df):
    df['text_length'] = df['processed_text'].apply(len)
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='text_length', hue='severity', element='step')
    plt.title('Distribution of Text Lengths by Severity')
    plt.xlabel('Text Length')
    plt.ylabel('Frequency')
    plt.show()

# Function for sentiment polarity distribution
def plot_sentiment_polarity(df):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='severity', y='polarity', data=df)
    plt.title('Distribution of Sentiment Polarity by Severity')
    plt.xlabel('Severity')
    plt.ylabel('Sentiment Polarity')
    plt.show()

# Now you can call this function with your DataFrame
plot_severity_distribution(df)

# Generate word clouds for each severity
for severity in df['severity'].unique():
    generate_word_cloud(df, severity)

# Plot histogram of text length
plot_text_length_histogram(df)

# Plot sentiment polarity distribution
plot_sentiment_polarity(df)

