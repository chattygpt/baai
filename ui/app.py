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

from agents import run_analysis
from utils.setup import setup_project
from config import MODEL_NAME, DEBUG_MODE

# Initialize
setup_project()

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

    # Initialize session state
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = True
    if 'thread_id' not in st.session_state:
        st.session_state.thread_id = None
    if 'file_id' not in st.session_state:
        st.session_state.file_id = None

    st.title("SQL Analysis Assistant")
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Add debug mode toggle and clear button in sidebar
        st.sidebar.title("Settings")
        col_debug, col_clear = st.sidebar.columns(2)
        
        with col_debug:
            debug_enabled = st.checkbox("Enable Debug Mode", value=st.session_state.debug_mode)
            st.session_state.debug_mode = debug_enabled
            
        with col_clear:
            if st.button("New Analysis"):
                st.session_state.thread_id = None
                st.session_state.file_id = None
                st.session_state.conversation_history = []
                st.experimental_rerun()

        # File upload section
        st.subheader("1. Upload Data (Optional)")
        uploaded_file = st.file_uploader(
            "Upload a CSV file",
            type=["csv"]
        )
        
        file_path = None
        if uploaded_file:
            with st.spinner("Processing file..."):
                try:
                    file_path = save_uploaded_file(uploaded_file)
                    if file_path:
                        # Initialize with file
                        result = run_analysis(
                            query="Initialize data analysis",
                            file_path=file_path, 
                            debug_mode=st.session_state.debug_mode,
                            initialize=True
                        )
                        
                        if result.get("status") == "success":
                            st.session_state.thread_id = result.get("thread_id")
                            st.session_state.file_id = result.get("file_id")
                            st.success(f"File processed: {uploaded_file.name}")
                        else:
                            st.error(f"Failed to initialize analysis: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")

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
                    result = run_analysis(
                        query=query, 
                        file_path=file_path, 
                        debug_mode=st.session_state.debug_mode,
                        initialize=False,
                        thread_id=st.session_state.thread_id,
                        file_id=st.session_state.file_id
                    )
                    
                    if result.get("status") == "success":
                        # Show the response
                        response = result.get('response', {})
                        if isinstance(response, dict):
                            # Show final answer prominently
                            st.markdown("### Answer")
                            st.success(response.get('final_answer', ''))
                            
                            # Show analysis details
                            with st.expander("Analysis Details", expanded=True):
                                # Show steps
                                if response.get('steps'):
                                    st.markdown("**Steps Taken:**")
                                    for i, step in enumerate(response['steps'], 1):
                                        st.markdown(f"{i}. {step}")
                                
                                # Show results
                                if response.get('results'):
                                    st.markdown("\n**Findings:**")
                                    for res in response['results']:
                                        st.markdown(f"• {res}")
                            
                            # Show code only in debug mode
                            if st.session_state.debug_mode:
                                if response.get('code'):
                                    with st.expander("Code Used", expanded=False):
                                        st.code(response['code'], language='python')
                                        
                                if result.get('debug_output'):
                                    with st.expander("Debug Output", expanded=False):
                                        st.text(result['debug_output'])
                        else:
                            st.success(str(response))
                        
                        # Add to conversation history
                        st.session_state.conversation_history.append({
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'query': query,
                            'response': response.get('final_answer', '') if isinstance(response, dict) else str(response)
                        })
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
                st.markdown(f"**Question:** {entry['query']}")
                st.markdown(f"**Answer:** {entry['response']}")
                st.markdown("---")
        else:
            st.info("No conversation history yet")

        # Clear History button
        if st.button("Clear History"):
            st.session_state.conversation_history = []
            st.experimental_rerun()

if __name__ == "__main__":
    main() 