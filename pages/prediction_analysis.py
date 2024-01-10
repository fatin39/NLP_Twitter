import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    data_path = '/Users/nurfatinaqilah/Documents/streamlit-test/dataset/predictions_severity.csv'
    return pd.read_csv(data_path)

def prediction_app():
    st.title("Text Severity Prediction Analysis")

    df = load_data()

    if st.checkbox('Show prediction data'):
        st.subheader('Prediction Data')
        st.write(df)

    # Display count plot for predictions
    if st.checkbox('Show severity distribution'):
        st.subheader('Severity Distribution')
        sns.countplot(data=df, x='severity_prediction')
        st.pyplot(plt)

    # Display specific text analysis
    if st.checkbox('Analyze Specific Text'):
        text_id = st.number_input('Enter text ID', min_value=0, max_value=len(df)-1, value=0)
        text_data = df.iloc[text_id]
        st.write('Processed Text:', text_data['processed_text'])
        st.write('Prediction:', text_data['prediction'])
        st.write('Severity:', text_data['severity_prediction'])
