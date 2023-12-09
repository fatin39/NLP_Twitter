# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt

# def app():
#     st.title('Data Analysis')

#     # Load processed data
#     df = pd.read_csv('data/processed_sentiment140.csv')

#     # Display Data (example)
#     if st.checkbox('Show processed data'):
#         st.write(df)

#     # Visualization (example)
#     st.subheader('Sentiment Distribution')
#     fig, ax = plt.subplots()
#     df['sentiment'].value_counts().plot(kind='bar', ax=ax)
#     st.pyplot(fig)
