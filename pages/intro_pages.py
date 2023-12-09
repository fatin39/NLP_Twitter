import streamlit as st
import pandas as pd
from visualisation.sentiment_distribution import sentiment_distribution
#from visualisation.sentiment_distribution import sentiment_distribution_app

# Define the correct file path to the CSV file
csv_file_path = "/Users/nurfatinaqilah/Documents/streamlit-test/data/sentiment_analysis.csv"

# Read the CSV file with the appropriate encoding
@st.cache_data  # Cache the data for better performance
def load_data():
    df = pd.read_csv(csv_file_path, encoding='latin1')  # Specify the encoding as 'latin1'
    return df

df = load_data()

def app():
    st.title('Introduction')
    st.markdown("This is the sentiment140 dataset. It contains around 1,600,000 tweets extracted using the twitter api . The tweets have been annotated (0 = negative, 4 = positive) and they can be used to detect sentiment .")

    # Left Column (Overview)
    left_column = st.container()
    with left_column:
        st.markdown('<div style="border: 1px solid #9AD8E1; border-radius: 10px; padding: 10px;">'
                    '<h2 style="color: #36b9cc;">Overview</h2>'
                    '<p><satrong>target</strong>: The polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)</p>'
                    '<p><strong>ids</strong>: The id of the tweet (2087)</p>'
                    '<p><strong>date</strong>: The date of the tweet (Sat May 16 23:58:44 UTC 2009)</p>'
                    '<p><strong>flag</strong>: The query (lyx). If there is no query, then this value is NO_QUERY.</p>'
                    '<p><strong>user</strong>: The user that tweeted (robotickilldozr)</p>'
                    '<p><strong>text</strong>: The text of the tweet (Lyx is cool)</p>'
                    '</div>', unsafe_allow_html=True)

    # Add space between columns
    st.markdown('')

    # Right Column (Summary and Boxes)
    right_column = st.container()
    with right_column:
        # Summary
        st.markdown('<div style="border: 1px solid #9AD8E1; border-radius: 10px; padding: 10px;">'
                    '<h2 style="color: #36b9cc;">Summary</h2>'
                    '<p>This section summarizes the key findings and insights from the analysis.</p>'
                    '</div>', unsafe_allow_html=True)

        # Add space between columns
        st.markdown('')

        # Create a horizontal layout for boxes
        col1, col2, col3 = st.columns([1, 1, 1])  # Create 3 columns with equal width
        with col1:
            # Box 1
            # Use the sentiment distribution app here
            sentiment_distribution_app(df)
            st.markdown('<div style="border: 1px solid #9AD8E1; border-radius: 10px; padding: 20px;">'
                        '<h3 style="color: #36b9cc;">Box 1</h3>'
                        '<p>This is another box with additional information. Make it longer if needed.</p>'
                        '</div>', unsafe_allow_html=True)

        with col2:
            # Box 2
            st.markdown('<div style="border: 1px solid #9AD8E1; border-radius: 10px; padding: 20px;">'
                        '<h3 style="color: #36b9cc;">Box 2</h3>'
                        '<p>This is a third box with more content. Make it longer if needed.</p>'
                        '</div>', unsafe_allow_html=True)

        with col3:
            # Box 3
            st.markdown('<div style="border: 1px solid #9AD8E1; border-radius: 10px; padding: 20px;">'
                        '<h3 style="color: #36b9cc;">Box 3</h3>'
                        '<p>This is a fourth box with even more information. Make it longer if needed.</p>'
                        '</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    app()