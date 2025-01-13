from utils.setup import setup_project
import streamlit.cli as stcli
import sys

if __name__ == "__main__":
    # Initialize project
    setup_project()
    
    # Run Streamlit
    sys.argv = ["streamlit", "run", "ui/app.py"]
    stcli.main() 