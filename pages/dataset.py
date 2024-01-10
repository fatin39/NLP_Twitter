import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

def dataset_app():
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

        # Add a selectbox for label filtering
        unique_labels = sorted(df['label'].unique())
        selected_labels = st.multiselect('Filter by label(s)', options=unique_labels, default=unique_labels)

        # Display up to 10 rows for each selected label
        for label in selected_labels:
            st.write(f"Label {label}:")
            filtered_df = df[df['label'] == label].head(10)  # Adjust the number of rows as needed
            st.dataframe(filtered_df)



    # Function to display label distribution with corrected sns.countplot usage
    import plotly.express as px

    def display_label_distribution(df):
        if df is not None and not df.empty:
            st.subheader("Sentiment Distribution")
            
            label_descriptions = {
                0: "Normal or non-depressed text",
                1: "Depressed text",
                2: "Suicidal text"
                # Add more descriptions as needed
            }
            
            fig = px.pie(df, names='label', title='Sentiment Distribution')
            fig.update_traces(textinfo='percent+label', hoverinfo='label+percent', 
                            hovertemplate="%{label}: <br> %{percent} <br> " + 
                            df['label'].apply(lambda x: label_descriptions.get(x, "")))
            st.plotly_chart(fig, use_container_width=True)

    import plotly.express as px

    def display_word_length_distribution(df):
        st.subheader("Word Length Distribution")
        if 'text' in df.columns:
            df['word_length'] = df['text'].apply(lambda x: len(str(x).split()))
            fig = px.histogram(df, x='word_length')
            # Remove the title from the figure to prevent duplication with the Streamlit subheader
            fig.update_layout(showlegend=False, title=None)
            st.plotly_chart(fig, use_container_width=True)
            
    # Function to generate and display word clouds for each label
    def generate_word_clouds(df):
        st.subheader("Word Clouds")
        unique_labels = df['label'].unique()
        selected_label = st.selectbox("Choose a label for the word cloud", unique_labels)
        text = " ".join(review for review in df[df['label'] == selected_label]['text'] if isinstance(review, str))
        if text:
            wordcloud = WordCloud(background_color='white').generate(text)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.write(f"No text data available for word cloud for Label {selected_label}.")

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
            <a href="#data-preview">Data Preview</a>
            <a href="#sentiment-distribution">Sentiment Distribution</a>
            <a href="#word-length">Word Length</a>
            <a href="#word-clouds">Word Clouds</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Section 1: Dataset
    #st.markdown("<h2 id='dataset-used'>Dataset</h2>", unsafe_allow_html=True)
    # Add content for the dataset used section here...
    dataset_names = ["Sentiment 140", "Depressed and Non-Depressed Tweets", "Reddit Depression & SuicideWatch", "Reddit Depression Only", "Reddit Depression Only 2", "Suicide Text"]
    selected_dataset = st.selectbox("Select a dataset to view:", dataset_names)

    # Section 2: Overview
    if selected_dataset == "Sentiment 140":
        st.markdown(
            "This is the Sentiment 140 dataset, which contains 1,600,000 tweets extracted using the Twitter API."
            "In this analysis, we are only using tweets labeled as '0'. The label '0' represents normal, non-depressed,"
            "or negative sentiment texts. These texts are crucial for our model to identify and differentiate"
            "normal/negative sentiment from those indicative of depression. The aim is to train the model to effectively"
            "distinguish between normal/negative sentiment texts and potentially depressed ones."
        )

    elif selected_dataset == "Depressed and Non-Depressed Tweets":
        st.write("This is a dataset containing data from users on Twitter who are depressed and users who are not, extracted using the Twitter API. [Dataset Link](https://www.kaggle.com/datasets/hyunkic/twitter-depression-dataset/data)")

    elif selected_dataset == "Reddit Depression & SuicideWatch":
        st.write("This dataset includes posts from Reddit's 'depression' and 'SuicideWatch' forums.")

    elif selected_dataset == "Reddit Depression Only":
        st.write("This dataset includes posts from Reddit's 'depression' forum.")

    elif selected_dataset == "Reddit Depression Only 2":
        st.write("This dataset includes additional posts from Reddit's 'depression' forum.")

    elif selected_dataset == "Suicide Text":
        st.write("The dataset is a collection of posts from the 'SuicideWatch' and 'depression' subreddits of the Reddit platform. The posts are collected using the Pushshift API. [Dataset Link](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch)")
        # Add specific information about this dataset.

    # Section 3: Data Preview
    #st.markdown("<h2 id='data-preview'>Data Preview</h2>", unsafe_allow_html=True)
    with st.expander("Data Preview"):
        df = preprocess_and_load_data(selected_dataset)
        display_dataset_details(df)

    # Section 4: Sentiment distribution section
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
