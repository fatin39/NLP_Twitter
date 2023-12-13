import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Set the Streamlit app configuration
st.set_page_config(layout='wide', initial_sidebar_state='expanded')

# Create a sidebar header
st.sidebar.header('Dashboard')

# Define sidebar options
selected_page = st.sidebar.selectbox("Select a page", ["Data Used", "Preprocessing", "Processed Data", "Twitter Depression Prediction"])

# Create a static header
st.sidebar.markdown(
    """
    # Sentiment Analysis Dashboard
    """
)

# Define the main content area
content = st.empty()

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
        <a href="#dataset-used">Dataset Used</a>
        <a href="#sentiment-distribution">Sentiment Distribution</a>
        <a href="#word-length">Word Length</a>
        <a href="#word-clouds">Word Clouds</a>
    </div>
    """,
    unsafe_allow_html=True,
)