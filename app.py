import streamlit as st
from src.model import GDPRAssistant
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="GDPR Assistant",
    page_icon="📚"
)

@st.cache_resource
def get_assistant():
    try:
        return GDPRAssistant()
    except Exception as e:
        logger.error(f"Error initializing assistant: {e}")
        raise e

def main():
    st.title("GDPR Q&A Assistant")
    
    # Add a sidebar with information
    st.sidebar.title("About")
    st.sidebar.info("This is a GDPR Q&A assistant that helps answer questions about GDPR regulations.")
    
    # Initialize assistant
    try:
        assistant = get_assistant()
        
        # Create the main interface
        question = st.text_input("Ask your GDPR-related question:")
        
        if question:
            try:
                with st.spinner("Finding answer..."):
                    response = assistant.ask(question)
                    st.write("Answer:", response)
            except Exception as e:
                st.error(f"Error processing question: {e}")
                logger.error(f"Error processing question: {e}")
                
    except Exception as e:
        st.error("Failed to initialize the assistant. Please check if the data file exists and is accessible.")
        logger.error(f"Initialization error: {e}")

if __name__ == "__main__":
    main()