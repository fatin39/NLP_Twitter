import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import sys
import os

# Set the Streamlit app configuration (call it only once at the beginning)
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
# In Streamlit_test.py, set the default page
selected_page = st.sidebar.selectbox("Select a page", ["Data Used", "Help and Insights", "Twitter Depression Prediction"], index=2)  # 4 corresponds to the index of "Twitter Depression Prediction"

# Define the main content area
content = st.empty()

# Add the 'pages' directory to the Python path
project_dir = '/Users/nurfatinaqilah/Documents/streamlit-test'
pages_dir = os.path.join(project_dir, 'pages')
sys.path.append(pages_dir)

# Import the app function from intro_pages.py
from dataset import dataset_app
if selected_page == "Data Used":
    dataset_app()

# from prediction_analysis import prediction_app
# if selected_page == "Analysis":
#     prediction_app()

# Integration of twitter_app
from twitter_interaction import twitter_app
if selected_page == "Twitter Depression Prediction":
   twitter_app()
   
from help_insights import help_insights_app
# Call the help_insights_app function when the new page is selected
if selected_page == "Help and Insights":
    help_insights_app()
