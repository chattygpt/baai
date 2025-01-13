import streamlit as st
from pathlib import Path
import sys
import os
import pandas as pd
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from agents import MasterAgent
from utils.setup import setup_project
from config import MODEL_NAME

# Initialize
setup_project()
master_agent = MasterAgent()

def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file and return path."""
    try:
        if not uploaded_file.name.endswith('.csv'):
            raise ValueError("Only CSV files are supported")
            
        file_path = Path("uploads") / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        return str(file_path)
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def main():
    st.set_page_config(
        page_title="SQL Analysis Assistant",
        page_icon="🤖",
        layout="wide"
    )

    # Initialize session state for conversation history
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    st.title("SQL Analysis Assistant")
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File upload section
        st.subheader("1. Upload Data (Optional)")
        uploaded_file = st.file_uploader(
            "Upload a CSV file",
            type=["csv"]
        )
        
        file_path = None
        if uploaded_file:
            with st.spinner("Saving file..."):
                file_path = save_uploaded_file(uploaded_file)
                if file_path:
                    st.success(f"File saved: {uploaded_file.name}")

        # Query section
        st.subheader("2. Enter Your Question")
        query = st.text_area(
            "What would you like to know about the data?",
            height=100,
            help="Enter your question in plain English. For example: 'What are the top 5 selling products?'"
        )

        # Analysis button
        if st.button("Analyze", type="primary"):
            if not query:
                st.warning("Please enter a question.")
                return
                
            with st.spinner("Analyzing..."):
                try:
                    result = master_agent.run(query=query, file_path=file_path)
                    
                    if result.get("status") == "success":
                        # Add to session state history
                        st.session_state.conversation_history.append({
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'query': query,
                            'response': result.get("response", ""),
                            'reasoning': result.get("reasoning", "Analysis steps not available")
                        })
                        
                        st.markdown("### Results")
                        st.markdown(result.get("response", ""))
                        
                        with st.expander("View Analysis Steps"):
                            st.markdown(result.get("reasoning", "Analysis steps not available"))
                    else:
                        st.error(result.get("error", "An unknown error occurred"))
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")

    with col2:
        # Conversation History
        st.subheader("Conversation History")
        if st.session_state.conversation_history:
            for i, entry in enumerate(reversed(st.session_state.conversation_history)):
                st.markdown(f"### {entry['timestamp']}")
                st.markdown(f"**Q:** {entry['query']}")
                st.markdown(f"**A:** {entry['response']}")
                if entry.get('reasoning'):
                    if st.button(f"Show Analysis #{i}", key=f"analysis_{i}"):
                        st.markdown("**Analysis Steps:**")
                        st.markdown(entry['reasoning'])
                st.markdown("---")
        else:
            st.info("No conversation history yet")

        # Clear History button
        if st.button("Clear History"):
            st.session_state.conversation_history = []
            st.experimental_rerun()

if __name__ == "__main__":
    main() 