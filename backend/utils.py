# Utility functions for session state management

def init_session_state():
    """Initialize Streamlit session state variables for the application
    
    This function sets up the necessary session state variables:
    - messages: List to store chat conversation history
    - image: Currently uploaded image for analysis
    - text_context: Text document for question answering
    """
    import streamlit as st
    
    # Initialize chat message history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize image storage for visual Q&A
    if "image" not in st.session_state:
        st.session_state.image = None
    
    # Initialize text context for document Q&A
    if "text_context" not in st.session_state:
        st.session_state.text_context = ""
