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
from utils.setup import setup_project, debug
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
        page_title="AI Biz Analyst",
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
    if 'user_prompt' not in st.session_state:
        st.session_state.user_prompt = ""
    
    # Initialize button clicked state if not exists
    if 'analyze_clicked' not in st.session_state:
        st.session_state.analyze_clicked = False

    st.title("AI Biz Analyst")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Data Analysis", "Thread Instructions", "Conversation History"])
    
    with tab1:
        # Create two columns for layout - main content and debug pane
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Add debug mode toggle and clear button in sidebar
            st.sidebar.title("Settings")
            col_debug, col_clear = st.sidebar.columns(2)
            
            with col_debug:
                debug_enabled = st.checkbox("Enable Debug Mode", value=st.session_state.debug_mode)
                st.session_state.debug_mode = debug_enabled
                
            with col_clear:
                if st.button("New Analysis", key="new_analysis"):
                    # Clear session state
                    if 'current_file_id' in st.session_state:
                        del st.session_state.current_file_id
                    # Clear other relevant session state variables
                    st.session_state.thread_id = None
                    st.session_state.file_id = None
                    st.session_state.conversation_history = []
                    st.rerun()

            # File upload section
            st.subheader("Upload Data")
            
            file_path = None  # Initialize file_path
            
            # Only show file uploader if no file is currently loaded
            if 'current_file_id' not in st.session_state:
                uploaded_file = st.file_uploader(
                    "Upload a CSV file (required)",
                    type=["csv"]
                )
                
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
                                    initialize=True,
                                    user_prompt=st.session_state.user_prompt
                                )
                                
                                if result.get("status") == "success":
                                    st.session_state.thread_id = result.get("thread_id")
                                    st.session_state.file_id = result.get("file_id")
                                    st.session_state.current_file_id = result.get("file_id")
                                    st.session_state.current_file_path = file_path  # Store file path
                                    # Store initialization result
                                    st.session_state.init_result = result
                                    st.success(f"File processed: {uploaded_file.name}")
                                else:
                                    st.error(f"Failed to initialize analysis: {result.get('error', 'Unknown error')}")
                        except Exception as e:
                            st.error(f"Error processing file: {str(e)}")
            else:
                # Use stored file path
                file_path = st.session_state.get('current_file_path')
                filename = Path(file_path).name if file_path else "unknown file"
                st.info(f"Currently using: {filename}\nClick 'New Analysis' to upload a different file.")

            # Query section
            st.subheader("Enter Your Question")
            query = st.text_area(
                "What would you like to know about the data?",
                height=100,
                key="query_input",
                help="Enter your question in plain English. For example: 'What are the top 5 selling products?'"
            )

            # Analysis button
            analyze_button = st.button("Analyze", type="primary", key="analyze_button")
            
            # Check if data is loaded before allowing analysis
            if analyze_button:
                if not file_path:
                    st.error("Please upload a data file before analyzing.")
                    return
                st.session_state.analyze_clicked = True
                # Clear previous debug output
                if hasattr(st.session_state, 'init_result'):
                    del st.session_state.init_result
                if hasattr(st.session_state, 'current_result'):
                    del st.session_state.current_result
                st.rerun()

            # Show answer in the left pane
            if st.session_state.analyze_clicked:
                st.session_state.analyze_clicked = False
                
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
                            file_id=st.session_state.file_id,
                            user_prompt=st.session_state.user_prompt
                        )
                        
                        if result and result.get("status") == "success":
                            # Show the response
                            response = result.get('response', {})
                            if isinstance(response, dict):
                                # Show final answer prominently in left pane
                                st.markdown("### Answer")
                                st.success(response.get('final_answer', ''))
                            else:
                                st.success(str(response))

                            # Store result in session state for right pane to use
                            st.session_state.current_result = result
                            
                            # Add to conversation history
                            st.session_state.conversation_history.append({
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'query': query,
                                'response': response.get('final_answer', '') if isinstance(response, dict) else str(response)
                            })
                        elif result:  # Handle error case when result exists
                            error_msg = result.get("error", "An unknown error occurred")
                            st.error(error_msg)
                        else:  # Handle case when result is None
                            st.error("Analysis failed: No response received")
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")

        # Debug and Results pane on the right
        with col2:
            st.markdown("### Debug Output")
            if st.session_state.debug_mode:
                # Show initialization analysis if available
                if hasattr(st.session_state, 'init_result'):
                    init_result = st.session_state.init_result
                    
                    if init_result.get('response') and isinstance(init_result['response'], dict):
                        with st.expander("Analysis Details", expanded=False):
                            if init_result['response'].get('steps'):
                                st.markdown("**Steps Taken:**")
                                for i, step in enumerate(init_result['response']['steps'], 1):
                                    st.markdown(f"{i}. {step}")
                            if init_result['response'].get('results'):
                                st.markdown("\n**Findings:**")
                                for res in init_result['response']['results']:
                                    st.markdown(f"• {res}")
                        
                        with st.expander("Code Used", expanded=False):
                            if init_result['response'].get('code'):
                                st.code(init_result['response']['code'], language='python')
                    
                    if init_result.get('debug_output'):
                        with st.expander("Full Analysis Log", expanded=True):
                            st.text_area("Debug Log", value=init_result['debug_output'], height=400, label_visibility="collapsed")
                
                # Show query analysis if available
                if hasattr(st.session_state, 'current_result'):
                    result = st.session_state.current_result
                    if result and result.get("status") == "success":
                        response = result.get('response', {})
                        if isinstance(response, dict):
                            with st.expander("Analysis Details", expanded=False):
                                if response.get('steps'):
                                    st.markdown("**Steps Taken:**")
                                    for i, step in enumerate(response['steps'], 1):
                                        st.markdown(f"{i}. {step}")
                                if response.get('results'):
                                    st.markdown("\n**Findings:**")
                                    for res in response['results']:
                                        st.markdown(f"• {res}")
                            
                            with st.expander("Code Used", expanded=False):
                                if response.get('code'):
                                    st.code(response['code'], language='python')
                        
                        if result.get('debug_output'):
                            with st.expander("Full Analysis Log", expanded=True):
                                st.text_area("Debug Log", value=result['debug_output'], height=400, label_visibility="collapsed")
                
                if not hasattr(st.session_state, 'init_result') and not hasattr(st.session_state, 'current_result'):
                    st.info("No debug output available yet.")
            else:
                st.info("Enable debug mode to see output here.")

    with tab2:
        st.header("Thread Instructions")
        user_prompt = st.text_area(
            "Enter custom instructions to be prepended to each analysis:",
            value=st.session_state.user_prompt,
            height=200,
            help="These instructions will be added before each analysis query."
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save Instructions", key="save_instructions"):
                st.session_state.user_prompt = user_prompt
                st.success("Instructions saved!")
        with col2:
            if st.button("Clear Instructions", key="clear_instructions"):
                st.session_state.user_prompt = ""
                st.success("Instructions cleared!")

    with tab3:
        st.header("Conversation History")
        if st.session_state.conversation_history:
            for i, entry in enumerate(reversed(st.session_state.conversation_history)):
                st.markdown(f"### {entry['timestamp']}")
                st.markdown(f"**Question:** {entry['query']}")
                st.markdown(f"**Answer:** {entry['response']}")
                st.markdown("---")
        else:
            st.info("No conversation history yet")

        # Clear History button
        if st.button("Clear History", key="clear_history_tab"):
            st.session_state.conversation_history = []
            st.rerun()

if __name__ == "__main__":
    main() 