from typing import Dict, Any, Optional
from openai import OpenAI
from openai.types.beta import Assistant, Thread
from openai.types.beta.threads import Run
from io import StringIO
import pandas as pd
import json
import time
from config import DEBUG_MODE, OPENAI_API_KEY

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def get_assistant() -> Assistant:
    """Get or create an OpenAI Assistant."""
    try:
        _assistant = client.beta.assistants.create(
            name="Data Analyst",
            description="Python data analysis expert using pandas",
            model="gpt-4o-mini",
            tools=[{"type": "code_interpreter"}],
            instructions="""You are a Python data analysis expert using pandas.
            
            You MUST return your response in this exact JSON format:
            {
                "code": "your complete python code here",
                "steps": ["step 1", "step 2", "etc"],
                "results": ["result 1", "result 2", "etc"],
                "final_answer": "your final answer here"
            }

            Important:
            1. Always use valid JSON format
            2. Include all fields even if empty
            3. Escape any special characters in strings
            4. Format numbers to 2 decimal places
            5. Handle missing data appropriately
            
            Example response:
            {
                "code": "import pandas as pd\\ndf_filtered = df[df['year'] == 2020]\\ntotal = df_filtered['sales'].sum()",
                "steps": ["Loaded data", "Filtered for 2020", "Calculated total"],
                "results": ["Found 500 rows", "Total sales: $1,234.56"],
                "final_answer": "The total sales for 2020 were $1,234.56"
            }"""
        )

        if DEBUG_MODE:
            print(f"Created Assistant: {_assistant.id}")

        return _assistant

    except Exception as e:
        if DEBUG_MODE:
            print(f"Error creating assistant: {str(e)}")
        raise

def get_df_info(df: pd.DataFrame) -> str:
    """Get DataFrame info as string."""
    buffer = StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()

def analyze_data(df: pd.DataFrame, query: str, debug_mode: bool = True, 
                thread_id: Optional[str] = None, file_id: Optional[str] = None,
                initialize: bool = False) -> Dict[str, Any]:
    # Analyze data using OpenAI Assistant
    #
    # Args:
    #   df: DataFrame to analyze
    #   query: User's question
    #   debug_mode: Whether to show debug output
    #   thread_id: Existing thread ID to reuse
    #   file_id: Existing file ID to reuse
    #   initialize: Whether this is an initialization call
    
    # Track resources for cleanup
    file = None
    thread: Optional[Thread] = None
    debug_output = []
    new_file = False
    
    def debug(msg: str):
        # Helper to handle debug output
        if debug_mode:
            debug_output.append(msg)
            if 'st' in globals():
                st.text(msg)
            print(msg)
    
    try:
        # Validate input data
        if df is None or df.empty:
            return {
                'status': 'error',
                'error': "No data provided or empty DataFrame"
            }

        debug("\n=== Analysis Start ===")
        
        # STEP 1: File Management
        # Create new file if:
        # - This is initialization, or
        # - We don't have a file ID yet
        if initialize or not file_id:
            # Convert DataFrame to CSV and upload
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            file_content = csv_buffer.getvalue().encode('utf-8')

            file = client.files.create(
                file=file_content,
                purpose='assistants'
            )
            file_id = file.id
            new_file = True  # Mark as new for cleanup
            debug(f"File uploaded: {file_id}")

            # STEP 2: Thread Creation
            # Create new thread for the conversation
            thread = client.beta.threads.create()
            thread_id = thread.id
            debug(f"Thread created: {thread_id}")

            # STEP 3: Initial Message
            # Add first message with file attachment
            message = client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=[{
                    "type": "text",
                    "text": "CSV file uploaded for analysis"
                }],
                attachments=[{
                    "file_id": file_id,
                    "tools": [{"type": "code_interpreter"}]
                }]
            )
            debug(f"Initial message created: {message.id}")

            # If this is initialization, return success
            if initialize:
                return {
                    'status': 'success',
                    'response': {
                        'code': '',
                        'steps': [],
                        'results': [],
                        'final_answer': "File processed successfully"
                    },
                    'thread_id': thread_id,
                    'file_id': file_id,
                    'debug_output': '\n'.join(debug_output) if debug_output else None
                }
        else:
            # Use existing thread for follow-up questions
            thread = client.beta.threads.retrieve(thread_id)
            debug(f"Using existing thread: {thread_id}")

        # STEP 4: Query Processing
        # Add the user's question as a new message
        message = client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=[{
                "type": "text",
                "text": f"""Analyze this data to answer: {query}

Return your response in this exact JSON format:
{{
    "code": "your python code",
    "steps": ["step 1", "step 2", "etc"],
    "results": ["result 1", "result 2", "etc"],
    "final_answer": "your final answer"
}}"""
            }]
        )
        debug(f"Query message created: {message.id}")

        # STEP 5: Run Analysis
        # Start the assistant's analysis
        run: Run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=get_assistant().id,
            instructions="Analyze the data and show your work step by step."
        )
        debug(f"\nAnalysis started: {run.id}")

        # STEP 6: Wait for Completion
        # Monitor the run with timeout
        timeout = 300  # 5 minutes
        start_time = time.time()
        
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError("Analysis timed out after 5 minutes")

            run_status = client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run.id
            )
            
            # Debug output for run status
            if DEBUG_MODE:
                debug(f"Current status: {run_status.status}")
                # Monitor code execution
                if hasattr(run_status, 'required_action') and run_status.required_action:
                    tool_calls = run_status.required_action.submit_tool_outputs.tool_calls
                    for tool_call in tool_calls:
                        if tool_call.type == 'code_interpreter':
                            debug("\n=== Code Execution ===")
                            debug("Code:")
                            debug(tool_call.code_interpreter.input)
                            # Show output
                            outputs = client.beta.threads.runs.steps.list(
                                thread_id=thread_id,
                                run_id=run.id
                            )
                            for output in outputs.data:
                                if hasattr(output, 'step_details') and output.step_details.type == 'tool_calls':
                                    debug("\nOutput:")
                                    debug(str(output.step_details.tool_calls[0].code_interpreter.outputs))
                            debug("=== End Execution ===\n")

            # Check run completion
            if run_status.status == 'completed':
                break
            elif run_status.status in ['failed', 'cancelled', 'expired']:
                raise Exception(f"Analysis failed with status: {run_status.status}")

            time.sleep(2)

        # STEP 7: Process Response
        # Get the assistant's response
        messages = client.beta.threads.messages.list(
            thread_id=thread_id,
            order="desc",
            limit=1
        )

        if not messages.data:
            raise Exception("No response received")

        response_text = messages.data[0].content[0].text.value
        
        try:
            # Parse JSON response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                response_json = json.loads(json_str)
            else:
                raise json.JSONDecodeError("No JSON found", response_text, 0)
            
            # Validate response format
            required_fields = ['code', 'steps', 'results', 'final_answer']
            missing_fields = [f for f in required_fields if f not in response_json]
            
            if missing_fields:
                raise ValueError(f"Response missing required fields: {missing_fields}")
                
            debug("\n=== Final Analysis ===")
            debug(json.dumps(response_json, indent=2))
            debug("===================\n")

            # Return successful response
            return {
                'status': 'success',
                'response': response_json,
                'thread_id': thread_id,
                'file_id': file_id,
                'debug_output': '\n'.join(debug_output) if debug_output else None
            }
            
        except (json.JSONDecodeError, ValueError) as e:
            debug(f"\nWarning: Invalid response format: {str(e)}")
            return {
                'status': 'error',
                'error': f"Error parsing response: {response_text}"
            }

    except Exception as e:
        error_msg = f"Analysis error: {str(e)}"
        debug(f"\nError: {error_msg}")
        return {
            'status': 'error',
            'error': error_msg
        }

    finally:
        # Resource Cleanup
        try:
            # Only clean up if initialization failed
            if initialize and not thread_id:
                if file:
                    client.files.delete(file.id)
                    debug(f"File deleted: {file.id}")
                if thread:
                    client.beta.threads.delete(thread.id)
                    debug(f"Thread deleted: {thread.id}")
            # For regular queries or successful initialization, keep resources
            debug("=== Cleanup Complete ===\n")

        except Exception as e:
            debug(f"Cleanup error: {str(e)}")

# Export the main function
__all__ = ['analyze_data'] 