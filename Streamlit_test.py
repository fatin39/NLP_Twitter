# import seaborn as sns
# import streamlit as st
# import pandas as pd
# import numpy
# import matplotlib.pyplot as plt
# import plost

# # hello world ! hi :) how r u? hiii

# st.set_page_config(layout='wide', initial_sidebar_state='expanded')

# with open('style.css') as f:
#     st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
# st.sidebar.header('Dashboard `version 2`')
# st.sidebar.subheader('Heat map parameter')
# time_hist_color = st.sidebar.selectbox('Color by', ('temp_min', 'temp_max')) 

# st.sidebar.subheader('Donut chart parameter')
# donut_theta = st.sidebar.selectbox('Select data', ('q2', 'q3'))

# st.sidebar.subheader('Line chart parameters')
# plot_data = st.sidebar.multiselect('Select data', ['temp_min', 'temp_max'], ['temp_min', 'temp_max'])
# plot_height = st.sidebar.slider('Specify plot height', 200, 500, 250)
# st.sidebar.markdown('''
# --- 
# Created with ❤️ by [Data Professor](https://youtube.com/dataprofessor/).
# ''')


# # Row A
# st.markdown('### Username: @farringt0n')
# col1, col2, col3 = st.columns(3)
# col1.metric("Followers", "113.9K", "0.0K")
# col2.metric("Tweets analysed", "248 Tweets", "-8%")
# col3.metric("Humidity", "86%", "4%")

# # Row B

# df = pd.read_csv("/Users/nurfatinaqilah/Documents/streamlit-test/data/user.csv", header=0)
# val_count  = df['label'].value_counts()
# fig = plt.figure(figsize=(10,5))
# sns.barplot(x=val_count.index, y=val_count.values, alpha=0.8)
# plt.title('Distribution of Sentiments')
# plt.ylabel('Number of tweets', fontsize=12)
# plt.xlabel('Sentiment', fontsize=12)

# c1, c2 = st.columns((7,3))
# with c1:
#     # Add figure in streamlit app
#     st.pyplot(fig)
    
# # with c2:
# #     st.markdown('### Donut chart')
# #     plost.donut_chart(
# #         data=stocks,
# #         theta=donut_theta,
# #         color='company',
# #         legend='bottom', 
# #         use_container_width=True)

# # Row C
# # st.markdown('### Line chart')
# # st.line_chart(seattle_weather, x = 'date', y = plot_data, height = plot_height)
