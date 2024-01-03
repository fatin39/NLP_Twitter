import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

def app():
    st.title('Dataset')
    # Function to preprocess and load dataset details
    def preprocess_and_load_data(dataset_name):
        
        if dataset_name == "Sentiment 140":
            df = pd.read_csv('/Users/nurfatinaqilah/Documents/streamlit-test/data/sentiment_analysis.csv', encoding='ISO-8859-1', names=['label', 'ids', 'date', 'flag', 'user', 'text'])
            df = df[['text', 'label']]
            df['label'] = df['label'].replace(4, 1)
        
        elif dataset_name == "Depressed and Non-Depressed Tweets":
            depressed_tweets_df = pd.read_csv('/Users/nurfatinaqilah/Documents/streamlit-test/data/depressed_tweets.csv')
            depressed_tweets_df.rename(columns={'tweet': 'text'}, inplace=True)
            depressed_tweets_df['label'] = 1  # Label for depressed tweets

            nondepressed_tweets_df = pd.read_csv('/Users/nurfatinaqilah/Documents/streamlit-test/data/nondepressed_tweets.csv')
            nondepressed_tweets_df.rename(columns={'tweet': 'text'}, inplace=True)
            nondepressed_tweets_df['label'] = 0  # Label for non-depressed tweets
            df = pd.concat([depressed_tweets_df, nondepressed_tweets_df])

            # Select only the "text" and "label" columns
            df = df[['text', 'label']]
            
        elif dataset_name == "Reddit Depression & SuicideWatch":
            df = pd.read_csv('/Users/nurfatinaqilah/Documents/streamlit-test/data/reddit_depression_suicidewatch.csv')
            df = df[['text', 'label']]
            df['label'] = df['label'].replace({'depression': 1, 'SuicideWatch': 2})

        elif dataset_name == "Reddit Depression Only":
            df = pd.read_csv('/Users/nurfatinaqilah/Documents/streamlit-test/data/reddit_depression_only.csv')
            df.rename(columns={'post': 'text'}, inplace=True)
            df['label'] = 1
            df = df[['text', 'label']]
            
        elif dataset_name == "Reddit Depression Only 2":
            df = pd.read_csv('/Users/nurfatinaqilah/Documents/streamlit-test/data/reddit_depression_only2.csv')
            df.rename(columns={'post': 'text'}, inplace=True)
            df['label'] = 1
            df = df[['text', 'label']]
            
        elif dataset_name == "Suicide Text":
            df = pd.read_csv('/Users/nurfatinaqilah/Documents/streamlit-test/data/suicide_text.csv')
            df.rename(columns={'class': 'label'}, inplace=True)
            df['label'] = df['label'].map({'suicide': 2, 'non-suicide': 0})
            df = df[['text', 'label']]

        else:
            st.error("Selected dataset is not recognized.")
            return pd.DataFrame()  # Return empty DataFrame

        return df

    # Function to display dataset details and visualizations
    def display_dataset_details(df):
        st.subheader("Data Preview")
        st.write(df.head())
        st.write(df.tail())

    # Function to display label distribution with corrected sns.countplot usage
    def display_label_distribution(df):
        if df is not None and not df.empty:
            sns.set_style("whitegrid")
            st.subheader("Sentiment Distribution")
            fig, ax = plt.subplots(figsize=(8, 4))
            countplot = sns.countplot(x='label', data=df, hue='label', palette='Set2', ax=ax, legend=False)
            
        # Calculate and annotate bars with percentages
        total = len(df)
        for p in countplot.patches:
            height = p.get_height()  # Get the height of each bar
            percentage = f'{100 * height/total:.1f}%'  # Calculate the percentage
            ax.text(p.get_x() + p.get_width() / 2., height + 3, f'{height}\n({percentage})', ha="center") 

        st.pyplot(fig)

    # Function to display word length distribution
    def display_word_length_distribution(df):
        st.subheader("Word Length Distribution")
        if 'text' in df.columns:
            df['word_length'] = df['text'].apply(lambda x: len(str(x).split()))
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(df['word_length'], bins=30, ax=ax)
            st.pyplot(fig)
        else:
            st.write("No text data available for word length distribution.")

    # Function to generate and display word clouds for each label
    def generate_word_clouds(df):
        unique_labels = df['label'].unique()
        for label_value in unique_labels:
            st.subheader(f"Word Cloud for Label {label_value}")
            text = " ".join(review for review in df[df['label'] == label_value]['text'] if isinstance(review, str))
            if text:
                wordcloud = WordCloud(background_color='white').generate(text)
                fig, ax = plt.subplots()
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)
            else:
                st.write(f"No text data available for word cloud for Label {label_value}.")

    # Create a navigation bar with links to sections
    st.markdown(
        """
        <style>
            .navbar {
                background-color: #000000;
                padding: 10px;
                border-radius: 10px;
            }

            .navbar a {
                color: white;
                text-decoration: none;
                margin-right: 20px;
                font-weight: bold;
            }

            .navbar a:hover {
                text-decoration: underline;
            }
        </style>
        <div class="navbar">
            <a href="#overview">Overview</a>
            <a href="#dataset-used">Dataset</a>
            <a href="#data-preview">Data Preview</a>
            <a href="#sentiment-distribution">Sentiment Distribution</a>
            <a href="#word-length">Word Length</a>
            <a href="#word-clouds">Word Clouds</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Section 1: Dataset
    st.markdown("<h2 id='dataset-used'>Dataset</h2>", unsafe_allow_html=True)
    # Add content for the dataset used section here...
    dataset_names = ["Sentiment 140", "Depressed and Non-Depressed Tweets", "Reddit Depression & SuicideWatch", "Reddit Depression Only", "Reddit Depression Only 2", "Suicide Text"]
    selected_dataset = st.selectbox("Select a dataset to view:", dataset_names)

    # Section 2: Overview
    st.markdown("<h2 id='overview'>Overview</h2>", unsafe_allow_html=True)

    if selected_dataset == "Sentiment 140":
        st.markdown(
            "This is the Sentiment 140 dataset, which contains 1,600,000 tweets extracted using the Twitter API."
            "                                                                                                  "
            "In this analysis, we are only using tweets labeled as '0'. The label '0' represents normal, non-depressed,"
            "or negative sentiment texts. These texts are crucial for our model to identify and differentiate"
            "normal/negative sentiment from those indicative of depression. The aim is to train the model to effectively"
            "distinguish between normal/negative sentiment texts and potentially depressed ones."
        )
    
    elif selected_dataset == "Depressed and Non-Depressed Tweets":
        st.write("This is a dataset containing data from users on twitter that are depressed and users who are not, extracted using twitter API. https://www.kaggle.com/datasets/hyunkic/twitter-depression-dataset/data")
        # Add specific information about this dataset.

    elif selected_dataset == "Reddit Depression & SuicideWatch":
        st.write("This dataset includes posts from Reddit's 'depression' and 'SuicideWatch' forums.")
    
    elif selected_dataset == "Reddit Depression Only":
        st.write("This dataset includes posts from Reddit's 'depression' forum.")
    
    elif selected_dataset == "Reddit Depression Only 2":
        st.write("This dataset includes additional posts from Reddit's 'depression' forum.")

    elif selected_dataset == "Suicide Text":
        st.write("The dataset is a collection of posts from the ""SuicideWatch"" and ""depression"" subreddits of the Reddit platform. The posts are collected using Pushshift API. https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch")
        # Add specific information about this dataset.

    # Section 3: Data Preview
    st.markdown("<h2 id='data-preview'>Data Preview</h2>", unsafe_allow_html=True)
    with st.expander("Data Preview"):
        df = preprocess_and_load_data(selected_dataset)
        display_dataset_details(df)

    # Sentiment distribution section
    st.markdown("<h2 id='sentiment-distribution'>Sentiment Distribution</h2>", unsafe_allow_html=True)
    display_label_distribution(df)

    # Section 5: Word Length
    st.markdown("<h2 id='word-length'>Word Length</h2>", unsafe_allow_html=True)
    display_word_length_distribution(df)

    # Section 6: Word Clouds
    st.markdown("<h2 id='word-clouds'>Word Clouds</h2>", unsafe_allow_html=True)
    # Add content for the word clouds section here...
    # Generate and display word clouds for each label
    if not df.empty:
        generate_word_clouds(df)
