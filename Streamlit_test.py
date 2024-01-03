import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import sys
import os

# Set the Streamlit app configuration
st.set_page_config(layout='wide', initial_sidebar_state='expanded')

no_sidebar_style = """
    <style>
        div[data-testid="stSidebarNav"] {display: none;}
    </style>
"""
st.markdown(no_sidebar_style, unsafe_allow_html=True)

# Create a static header
st.sidebar.markdown(
    """
    # Sentiment Analysis Dashboard
    """
)

# Define sidebar options
selected_page = st.sidebar.selectbox("Select a page", ["Data Used", "Preprocessing", "Processed Data", "Twitter Depression Prediction"])

# Define the main content area
content = st.empty()

# Add the 'pages' directory to the Python path
project_dir = '/Users/nurfatinaqilah/Documents/streamlit-test'
pages_dir = os.path.join(project_dir, 'pages')
sys.path.append(pages_dir)

# Import the app function from intro_pages.py
from dataset import app
if selected_page == "Data Used":
    app()

from twitter_interaction import app
if selected_page == "Twitter Depression Prediction":
    app()