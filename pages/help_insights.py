import streamlit as st

def help_insights_app():
    # Dropdown menu for user selection
    user_type = st.selectbox(
        "Who are you?",
        ["Individuals Concerned About Their Mental Health",
         "Mental Health Professionals",
         "Research Scholars and Academics"]
    )

    # Display information for individuals
    if user_type == "Individuals Concerned About Their Mental Health":
        
        # Add additional options for "Individuals Concerned About Their Mental Health"
        option = st.radio("Choose an option:",
                          ("Steps to Seek Mental Health Help in Malaysia (Private Path)",
                           "Understanding Symptoms",
                           "Hotline"))
        
        if option == "Steps to Seek Mental Health Help in Malaysia (Private Path)":
            st.markdown("### Steps to Seek Mental Health Help in Malaysia (Private Path)")
        
            # Display the Mentari Clinic logo
            # Replace '/path/to/mentari_logo.jpg' with the actual path to the logo image
            st.image("/Users/nurfatinaqilah/Documents/streamlit-test/images/mentari.png", caption="Mentari Clinic", width=700)

            st.image("/Users/nurfatinaqilah/Documents/streamlit-test/images/DASS_mentari.png.jpg", caption="DASS Test", width=700)

            st.markdown("""
                **1. Do the DASS Test**: Visit [Mentari's Self-Test Page](https://mentari.moh.gov.my/self-test/) and complete the DASS (Depression, Anxiety, and Stress Scale) test.
                
                **2. Download Test Result**: After completing the test, download your results.
                
                **3. Book an Appointment**: Book an appointment with the Klinik Mentari nearest to you.
                
                **4. Wait for Clinic's Call**: After booking, wait for the clinic to reach out to you for confirmation.
                
                **5. Attend the Appointment**: Go to the clinic for a series of tests (blood test, urine test) and a consultation session.
                
                **6. Referral for Professional Assistance**: If necessary, the doctor will provide a referral letter for further assistance from professionals such as psychiatrists or psychologists at a hospital.
            """)

            # Provide a link to the Mentari Clinic official website
            st.markdown("For more information, visit [Mentari Clinic's Official Website](https://mentari.moh.gov.my/malay/).")

            pass
        
        elif option == "Understanding Symptoms":
            st.markdown("### Symptoms")

            
            st.markdown("""
            
            **Psychological symptoms**

            The psychological symptoms of depression include:

            - continuous low mood or sadness
            - feeling hopeless and helpless
            - having low self-esteem
            - feeling tearful
            - feeling guilt-ridden
            - feeling irritable and intolerant of others
            - having no motivation or interest in things
            - finding it difficult to make decisions
            - not getting any enjoyment out of life
            - feeling anxious or worried
            - having suicidal thoughts or thoughts of harming yourself
            
            **Physical symptoms**
            
            The physical symptoms of depression include:

            - moving or speaking more slowly than usual
            - changes in appetite or weight (usually decreased, but sometimes increased)
            - constipation
            - unexplained aches and pains
            - lack of energy
            - low sex drive (loss of libido)
            - disturbed sleep – for example, finding it difficult to fall asleep at night or waking up very early in the morning

            **Social Symptoms**
            
            The social symptoms of depression include:

            - avoiding contact with friends and taking part in fewer social activities
            - neglecting your hobbies and interests
            - having difficulties in your home, work or family life
        """ 
            )
            pass
        
        elif option == "Hotline":
            st.markdown("If you need immediate assistance, please call this mental health crisis hotline: https://findahelpline.com/countries/my")
            pass
       
    elif user_type == "Mental Health Professionals":
        st.markdown("### Explanation for Mental Health Professionals")
        st.markdown("""
                    
                    
                    """)
        st.image("/Users/nurfatinaqilah/Documents/streamlit-test/images/Twitter Sentiment Analysis.drawio (1).png", caption="Methodology", width=700)

        st.markdown("""
        **Data Collected:**
        - From Kaggle and websites (Extracted by author)
        - Combined into a single dataset with labels aligned to 0 (normal), 1(depressed), 2(suicidal)  
                    """)
        st.image("/Users/nurfatinaqilah/Documents/streamlit-test/images/BERT MODEL.jpeg", caption="BERT Model", width=700)
        st.image("/Users/nurfatinaqilah/Documents/streamlit-test/images/distilBERT.png", caption="DistilBERT Model", width=700)

        st.markdown("**Deep Learning with BERT model (DistilBERT):**")
        st.markdown("""
        - BERT model differs from traditional machine learning models by using a deep learning approach to understand the context of words in text.
        - BERT’s key technical innovation is applying the bidirectional training of Transformer, a popular attention model, to language modelling. 
        - This is in contrast to previous efforts which looked at a text sequence either from left to right or combined left-to-right and right-to-left training. Instead, the Transformer processes the whole sentence at once and can thus capture more complex relationships between words.

        - DistilBERT, a distilled version of BERT, is a smaller, faster, cheaper, and lighter version of BERT.
        - Designed for practical applications where computational resources are limited while still preserving 95% of BERT's performances.
        """)

        st.markdown("**Conclusion:**")
        st.markdown("""
        This Final Year Project focused on detecting depression using Twitter data. 
        
        The key contribution is the development of a prediction model that leverages user profiles and tweets to predict depression, laying the foundation for future research in the field of depression detection and prevention.
        """)

        st.markdown("**Future Work:**")
        st.markdown("""
        - Implement a more granular analysis of depression levels.
        - Explore various deep learning models beyond BERT.
        - Expand the system to analyze multimedia data from social media.
        """)
        
        
    elif user_type == "Research Scholars and Academics":
        st.markdown("### Explanation for Research Scholars and Academics")
        st.markdown("""
                    
                    
                    """)
        st.image("/Users/nurfatinaqilah/Documents/streamlit-test/images/Twitter Sentiment Analysis.drawio (1).png", caption="Methodology", width=700)

        st.markdown("""
        **Data Collected:**
        - From Kaggle and websites (Extracted by author)
        - Combined into a single dataset with labels aligned to 0 (normal), 1(depressed), 2(suicidal)  
                    """)
        st.image("/Users/nurfatinaqilah/Documents/streamlit-test/images/BERT MODEL.jpeg", caption="BERT Model", width=700)
        st.image("/Users/nurfatinaqilah/Documents/streamlit-test/images/distilBERT.png", caption="DistilBERT Model", width=700)

        st.markdown("**Deep Learning with BERT model (DistilBERT):**")
        st.markdown("""
        - BERT model differs from traditional machine learning models by using a deep learning approach to understand the context of words in text.
        - BERT’s key technical innovation is applying the bidirectional training of Transformer, a popular attention model, to language modelling. 
        - This is in contrast to previous efforts which looked at a text sequence either from left to right or combined left-to-right and right-to-left training. Instead, the Transformer processes the whole sentence at once and can thus capture more complex relationships between words.

        - DistilBERT, a distilled version of BERT, is a smaller, faster, cheaper, and lighter version of BERT.
        - Designed for practical applications where computational resources are limited while still preserving 95% of BERT's performances.
        """)

        st.markdown("**Conclusion:**")
        st.markdown("""
        This Final Year Project focused on detecting depression using Twitter data. 
        
        The key contribution is the development of a prediction model that leverages user profiles and tweets to predict depression, laying the foundation for future research in the field of depression detection and prevention.
        """)

        st.markdown("**Future Work:**")
        st.markdown("""
        - Implement a more granular analysis of depression levels.
        - Explore various deep learning models beyond BERT.
        - Expand the system to analyze multimedia data from social media.
        """)