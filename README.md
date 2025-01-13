# Business Analysis AI (BAAI)

An intelligent Business Analysis system powered by AI that helps analyze documents and create presentations.

## Requirements
- Python 3.10 or higher
- OpenAI API key

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd baai
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/MacOS
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Copy .env.example to .env
   - Add your OpenAI API key to .env

5. Start the services:
   ```bash
   # Terminal 1: Start API
   uvicorn api.main:app --reload
   
   # Terminal 2: Start UI
   streamlit run ui/app.py
   ```

## Troubleshooting

1. If you get ChromaDB errors:
   ```bash
   rm -rf data/chroma  # Clear ChromaDB cache
   ```

2. If files aren't uploading:
   - Check uploads directory permissions
   - Ensure file size is under 100MB

3. If analysis fails:
   - Verify OpenAI API key is valid
   - Check if files are readable
   - Try with smaller documents first

## Project Structure
