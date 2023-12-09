import streamlit as st
from pages import intro_pages, data_analysis
from visualisation.sentiment_distribution import sentiment_distribution_app

with col1:
    # Display the sentiment bar chart in Box 1
    st.title('Box 1')
    sentiment_distribution_app(df)  # Call the function here
    # ...

PAGES = {
    "Introduction": intro_pages,
    "Data Analysis": data_analysis
}

def main():
    st.sidebar.title('Navigation')
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    page = PAGES[selection]

    with st.spinner(f"Loading {selection} ..."):
        page.app()

if __name__ == "__main__":
    main()
