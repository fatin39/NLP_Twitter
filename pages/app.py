# import streamlit as st
# import pages.dataset as dataset
# import processing_data
# # import eda
# # import train_test
# # import user_input_prediction
# import twitter_interaction

# # Dictionary of pages
# PAGES = {
#     "Dataset": dataset,
#     "Preprocessing": processing_data,
#     "Twitter Data Retrieval": twitter_interaction,
#     # "EDA": eda,
#     # "Train and Test": train_test,
#     # "User Input and Prediction": user_input_prediction
# }
# def main():
#     st.sidebar.title("Sentiment Analysis Dashboard")
#     selection = st.sidebar.selectbox("Select Page", list(PAGES.keys()))
    
#     page = PAGES[selection]
#     page.app()  # Call the app function from the selected page module

# if __name__ == "__main__":
#     main()