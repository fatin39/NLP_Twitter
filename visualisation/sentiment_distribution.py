import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def sentiment_distribution_app(df):
    st.title('Sentiment Distribution')
    
    # Calculate sentiment distribution
    sentiment_distribution = df['sentiment'].value_counts().reset_index()
    sentiment_distribution.columns = ['Sentiment', 'Count']

    # Create a bar chart using Matplotlib
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(sentiment_distribution['Sentiment'], sentiment_distribution['Count'], color=['red', 'green'])
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Count')
    ax.set_title('Sentiment Distribution')
    ax.set_xticks(sentiment_distribution['Sentiment'])
    ax.set_xticklabels(['Negative', 'Positive'])  # Replace with your own labels if needed

    # Display the chart
    st.pyplot(fig)
