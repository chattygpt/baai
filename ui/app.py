import streamlit as st
from pathlib import Path
import sys
import os
import pandas as pd
from datetime import datetime, timedelta
import time

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from agents import run_analysis
from utils.setup import setup_project, debug
from config import MODEL_NAME, DEBUG_MODE
from utils.vector_store import initialize_vector_store

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

def initialize_vector_search():
    """Initialize the vector search with proper error handling."""
    try:
        if 'vector_store_initialized' not in st.session_state:
            with st.spinner("Initializing document search..."):
                # Check if docs directory exists and has files
                docs_dir = Path("docs")
                if not docs_dir.exists() or not any(docs_dir.iterdir()):
                    st.warning("No documents found in the 'docs' directory. Vector search will be disabled.")
                    st.session_state.vector_store_initialized = False
                    return False
                
                # Initialize vector store
                initialize_vector_store()
                st.session_state.vector_store_initialized = True
                return True
        return st.session_state.vector_store_initialized
    except Exception as e:
        st.error(f"Error initializing document search: {str(e)}")
        st.session_state.vector_store_initialized = False
        return False

def kill_timed_out_threads():
    """Kill threads that have been running for too long."""
    try:
        if hasattr(st.session_state, 'thread_start_time'):
            current_time = datetime.now()
            thread_age = current_time - st.session_state.thread_start_time
            
            # If thread is older than 5 minutes, kill it
            if thread_age > timedelta(minutes=5):
                st.session_state.thread_id = None
                if hasattr(st.session_state, 'thread_start_time'):
                    del st.session_state.thread_start_time
                return True
        return False
    except Exception as e:
        st.error(f"Error killing thread: {str(e)}")
        return False

def main():
    st.set_page_config(
        page_title="AI Biz Analyst",
        page_icon="🤖",
        layout="wide"
    )

    # Initialize vector store with error handling
    vector_search_available = initialize_vector_search()

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

    # Maximum number of conversations to maintain in scroll
    MAX_CONVERSATIONS = 50

    st.title("AI Biz Analyst")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Data Analysis", "Thread Instructions", "Debug Output"])
    
    with tab1:
        # Create single column layout for main content
        with st.container():
            # Add debug mode toggle and clear button in sidebar
            st.sidebar.title("Settings")
            
            # Debug mode toggle
            debug_enabled = st.sidebar.checkbox("Enable Debug Mode", value=st.session_state.debug_mode)
            st.session_state.debug_mode = debug_enabled
            
            # New Analysis button
            if st.sidebar.button("New Analysis", key="new_analysis"):
                # Clear session state
                if 'current_file_id' in st.session_state:
                    del st.session_state.current_file_id
                # Clear thread ID to force new conversation
                st.session_state.thread_id = None
                st.session_state.file_id = None
                # Clear conversation history
                st.session_state.conversation_history = []
                # Clear file upload flag
                if 'file_uploaded' in st.session_state:
                    del st.session_state.file_uploaded
                # Clear any previous results
                if 'init_result' in st.session_state:
                    del st.session_state.init_result
                if 'current_result' in st.session_state:
                    del st.session_state.current_result
                st.rerun()
            
            # Kill Thread button
            if st.sidebar.button("Kill Thread", key="kill_thread"):
                if kill_timed_out_threads():
                    st.sidebar.success("Timed out thread killed successfully")
                    time.sleep(1)  # Give user time to see the message
                    st.rerun()
                else:
                    st.sidebar.info("No timed out threads to kill")

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

            # Query input at the top
            st.subheader("Ask a Question")
            query = st.text_area(
                "What would you like to know about the data?",
                height=100,
                key="query_input",
                help="Enter your question in plain English. For example: 'What are the top 5 selling products?'"
            )

            # Analysis button
            analyze_button = st.button("Analyze", type="primary", key="analyze_button")
            
            st.markdown("---")
            
            # Display conversation history with icons and chat-like bubbles in reverse order
            for entry in reversed(st.session_state.conversation_history):
                # User message with icon
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: start; margin-bottom: 1rem;">
                        <div style="background-color: #E8E8E8; padding: 0.5rem 1rem; border-radius: 15px; margin-left: 0.5rem; max-width: 80%;">
                            <p style="margin: 0;"><strong>🧑 You:</strong> {entry['query']}</p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # AI response with icon
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: start; margin-bottom: 1rem;">
                        <div style="background-color: #F0F7FF; padding: 0.5rem 1rem; border-radius: 15px; margin-left: 0.5rem; max-width: 80%;">
                            <p style="margin: 0;"><strong>🤖 AI:</strong></p>
                            <p style="margin: 0.5rem 0;">{entry['final_answer']}</p>
                            {"<hr style='margin: 0.5rem 0;'>" if entry['results'] else ""}
                            {"<br>".join(f"• {res}" for res in entry['results']) if entry['results'] else ""}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Check if data is loaded before allowing analysis
            if analyze_button:
                if not file_path:
                    st.error("Please upload a data file before analyzing.")
                    return
                st.session_state.analyze_clicked = True
                # Clear only current result, keep initialization result
                if hasattr(st.session_state, 'current_result'):
                    del st.session_state.current_result
                st.rerun()

            # Process analysis
            if st.session_state.analyze_clicked:
                st.session_state.analyze_clicked = False
                
                if not query:
                    st.warning("Please enter a question.")
                    return

                # Check for timed out threads before starting new analysis
                kill_timed_out_threads()

                with st.spinner("Analyzing..."):
                    try:
                        # Record thread start time
                        st.session_state.thread_start_time = datetime.now()
                        
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
                            response = result.get('response', {})
                            if isinstance(response, dict):
                                # Add to conversation history
                                st.session_state.conversation_history.append({
                                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    'query': query,
                                    'results': response.get('results', []) if isinstance(response, dict) else [],
                                    'final_answer': response.get('final_answer', '') if isinstance(response, dict) else str(response)
                                })
                                # Maintain only the most recent conversations
                                if len(st.session_state.conversation_history) > MAX_CONVERSATIONS:
                                    st.session_state.conversation_history = st.session_state.conversation_history[-MAX_CONVERSATIONS:]
                                
                                # Store result in session state for debug output
                                st.session_state.current_result = result
                                
                                st.rerun()  # Refresh to show new message in chat
                            else:
                                st.error("Unexpected response format")
                            
                        elif result:  # Handle error case when result exists
                            error_msg = result.get("error", "An unknown error occurred")
                            st.error(error_msg)
                        else:  # Handle case when result is None
                            st.error("Analysis failed: No response received")
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")

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
        st.header("Debug Output")
        if st.session_state.debug_mode:
            # Show initialization analysis if available
            if hasattr(st.session_state, 'init_result') and st.session_state.init_result:
                init_result = st.session_state.init_result
                st.markdown("### Initialization Analysis")
                
                if init_result.get('response') and isinstance(init_result['response'], dict):
                    with st.expander("Analysis Details", expanded=True):
                        if init_result['response'].get('steps'):
                            st.markdown("**Steps Taken:**")
                            for i, step in enumerate(init_result['response']['steps'], 1):
                                st.markdown(f"{i}. {step}")
                        if init_result['response'].get('results'):
                            st.markdown("\n**Findings:**")
                            for res in init_result['response']['results']:
                                st.markdown(f"• {res}")
                    
                    with st.expander("Code Used", expanded=True):
                        if init_result['response'].get('code'):
                            st.code(init_result['response']['code'], language='python')
                
                if init_result.get('debug_output'):
                    with st.expander("Full Analysis Log", expanded=True):
                        st.text_area("Debug Log", value=init_result['debug_output'], height=400, label_visibility="collapsed")
            
            # Show query analysis if available
            if hasattr(st.session_state, 'current_result') and st.session_state.current_result:
                result = st.session_state.current_result
                if result and result.get("status") == "success":
                    st.markdown("### Query Analysis")
                    response = result.get('response', {})
                    if isinstance(response, dict):
                        with st.expander("Analysis Details", expanded=True):
                            if response.get('steps'):
                                st.markdown("**Steps Taken:**")
                                for i, step in enumerate(response['steps'], 1):
                                    st.markdown(f"{i}. {step}")
                            if response.get('results'):
                                st.markdown("\n**Findings:**")
                                for res in response['results']:
                                    st.markdown(f"• {res}")
                        
                        with st.expander("Code Used", expanded=True):
                            if response.get('code'):
                                st.code(response['code'], language='python')
                    
                    if result.get('debug_output'):
                        with st.expander("Full Analysis Log", expanded=True):
                            st.text_area("Debug Log", value=result['debug_output'], height=400, label_visibility="collapsed")
            
            if (not hasattr(st.session_state, 'init_result') or not st.session_state.init_result) and \
               (not hasattr(st.session_state, 'current_result') or not st.session_state.current_result):
                st.info("No debug output available yet. Run an analysis to see debug information.")
        else:
            st.info("Enable debug mode in Settings to see output here.")

if __name__ == "__main__":
    main() 