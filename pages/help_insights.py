import streamlit as st

def help_insights_app():
    # Dropdown menu for user selection
    user_type = st.selectbox(
        "Who are you?",
        ["Individuals Concerned About Their Mental Health",
         "Mental Health Professionals",
         "Parents and Guardians",
         "Research Scholars and Academics"]
    )

    # Display information for individuals
    if user_type == "Individuals Concerned About Their Mental Health":
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

    elif user_type == "Mental Health Professionals":
        st.markdown("### Resources for Mental Health Professionals")
        # Add guides, tutorials, and case studies here...
        # Example:
        st.markdown("*Guide on Integrating System in Therapy Sessions*")
        st.markdown("*Case Studies on Effective Use in Clinical Practice*")

    elif user_type == "Parents and Guardians":
        st.markdown("### Resources for Parents and Guardians")
        # Add educational materials, guidance on monitoring, and support resources here...
        # Example:
        st.markdown("*Understanding Adolescent Mental Health: A Comprehensive Guide*")
        st.markdown("*Tips for Monitoring Your Child's Online Activity Safely*")

    elif user_type == "Research Scholars and Academics":
        st.markdown("### Resources for Research Scholars and Academics")
        # Add information on data analysis tools, ethical considerations, and research collaboration opportunities here...
        # Example:
        st.markdown("*Utilizing Social Media Data in Mental Health Research: Tools and Ethics*")
        st.markdown("*Collaboration Opportunities in Mental Health Research Projects*")

