import streamlit as st
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from agents import MasterAgent  # Import from agents package
from utils.logger import get_logger
from utils.setup import setup_project
from config import MODEL_NAME

# Initialize
logger = get_logger(__name__)
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
        logger.debug(f"File saved: {file_path}")
        return str(file_path)
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        st.error(f"Error saving file: {e}")
        return None

def main():
    st.set_page_config(
        page_title="SQL Analysis Assistant",
        page_icon="🤖",
        layout="wide"
    )

    st.title("SQL Analysis Assistant")
    
    # File upload section
    st.subheader("1. Upload Data (Optional)")
    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"]  # Only allow CSV files
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
                    st.markdown("### Results")
                    st.markdown(result.get("response", ""))
                    
                    if result.get("sql_query"):
                        with st.expander("View SQL Query"):
                            st.code(result["sql_query"], language="sql")
                else:
                    st.error(result.get("error", "An unknown error occurred"))
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")

    # Debug information in expander
    with st.expander("Debug Information"):
        st.write({
            "File Path": file_path,
            "Project Root": str(project_root),
            "Debug Mode": os.getenv("DEBUG_MODE", "true"),
            "Model": MODEL_NAME
        })

if __name__ == "__main__":
    main() 