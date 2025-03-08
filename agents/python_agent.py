from typing import Dict, Any, Optional
import json
import time
import pandas as pd
from io import StringIO
import streamlit as st
from openai import OpenAI
from openai.types.beta import Assistant
from openai.types.beta.threads import Run
from config import DEBUG_MODE, OPENAI_API_KEY
from utils.setup import setup_project, debug

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def get_assistant() -> Assistant:
    """Get or create an OpenAI Assistant."""
    try:
        # Get assistant from session state or create new
        if 'openai_assistant' not in st.session_state:
            assistant = client.beta.assistants.create(
                model="gpt-4o-mini",
                name="StructuredOutputAssistant",
                tools=[{"type": "code_interpreter"}],
                instructions="""You are a Python data analysis expert. When analyzing data:
    1. You must write code to answer the given query based on conversation history and any uploaded files. 
    2. You must then run the code and generate output.
    3. Then verify that the output is correct.
    4. If the output is not correct, you must write a new code to answer the query.
    5. ALWAYS refer to previous messages in the thread for context before writing code.
    6. If a question directly or indirectly refers to previous or last analysis, use that information
    7. Build upon previous analyses when relevant
    8. Return ONLY a JSON object in this exact format:
                {
                    "code": "your complete python code here",
                    "steps": ["step 1", "step 2", "etc"],
                    "results": ["result 1", "result 2", "etc"],
                    "final_answer": "your final answer here"
                }""",
                response_format={"type": "json_object"}
            )
            st.session_state.openai_assistant = assistant
            debug(f"Created Assistant: {assistant.id}")
        
        return st.session_state.openai_assistant

    except Exception as e:
        debug(f"Assistant Error Details:")
        debug(f"- Error type: {type(e).__name__}")
        debug(f"- Error message: {str(e)}")
        raise

def get_df_info(df: pd.DataFrame) -> str:
    """Get DataFrame info as string."""
    buffer = StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()

def analyze_data(query: str, file_id: Optional[str] = None, thread_id: Optional[str] = None, user_prompt: Optional[str] = None) -> Dict[str, Any]:
    """Run analysis on uploaded data."""
    debug_output = []
    
    try:
        debug("\n=== Analysis Start ===", debug_output)
        if user_prompt:
            debug(f"User prompt: {user_prompt}", debug_output)

        # STEP 1: File Management
        # Store file_id in session state if not already present
        if file_id and 'current_file_id' not in st.session_state:
            st.session_state.current_file_id = file_id
        elif 'current_file_id' in st.session_state:
            file_id = st.session_state.current_file_id
        
        # STEP 2: Thread Management
        # Create new thread only if none exists
        if not thread_id:
            thread = client.beta.threads.create()
            thread_id = thread.id
            debug(f"New thread created: {thread_id}", debug_output)
        else:
            debug(f"Using existing thread: {thread_id}", debug_output)

        # STEP 4: Create Messages
        # First message: Upload file only if not already uploaded in this session
        if file_id and not st.session_state.get('file_uploaded', False):
            # Verify file exists and is readable
            try:
                file = client.files.retrieve(file_id)
                debug(f"File verified: {file.filename}, size: {file.bytes}, purpose: {file.purpose}", debug_output)
            except Exception as e:
                debug(f"Error verifying file: {str(e)}", debug_output)
                raise

            file_message = client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=[{
                    "type": "text",
                    "text": "Here is the CSV file for analysis. Please confirm you can read it by showing the first few rows."
                }],
                attachments=[{
                    "file_id": file_id,
                    "tools": [{"type": "code_interpreter"}]
                }]
            )
            debug(f"File upload message created: {file_message.id}", debug_output)
            st.session_state.file_uploaded = True  # Mark file as uploaded
        elif file_id:
            debug("File already uploaded in this session, skipping upload", debug_output)

        # Get conversation history and summarize it
        history = client.beta.threads.messages.list(
            thread_id=thread_id,
            order="asc"  # Get messages in chronological order
        )
        
        debug(f"Conversation History:\n{history}\n", debug_output)
        # Format conversation history more cleanly
        conversation_text = []
        for msg in history.data:
            role = "user" if msg.role == "user" else "assistant"
            # Clean the content text
            content = msg.content[0].text.value if msg.content else "No content"
            content = content.replace('\\', '/').replace('"', "'")  # Replace backslashes with forward slashes
            conversation_text.append(f"{role}: {content}")

        debug(f"Conversation History:\n{' '.join(conversation_text)}\n", debug_output)

        # Create summary request with cleaner formatting
        summary_request = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system", 
                    "content": """You are a helpful assistant. Summarize the conversation history and combine it with the new query to create a new query.

EXAMPLE 1:
- Conversation history: 
  Question: Which region had the highest revenue? 
  Answer: California had the highest revenue.
- New query: 
  Question: What were the top products?
- Generated query: 
  Question: Which are the top selling products in California?

EXAMPLE 2:
- Conversation history:
  Question: Which demograhic group purchases cars most frequently? 
  Answer: 25-34 years old
- New query:
  Question: What is their average purchase amount?
- Generated query:
  Question: What is the average purchase amount for the 25-34 year old demographic?

Generated queries should be concise and easily convertible to Python code."""
                },
                {
                    "role": "user",
                    "content": """Conversation history:
{}

New query:
{}

""".format(
                        '\n'.join(conversation_text),
                        query.replace('\\', '/').replace('"', "'")  # Clean the query text too
                    )
                }
            ]
        }
        
        # Get summary
        summary_response = client.chat.completions.create(**summary_request)
        summary = summary_response.choices[0].message.content
        debug(f"Conversation Summary: {summary}", debug_output)

        # Clean up summary text to avoid escape character issues
        clean_summary = summary#.replace('\\', '').replace('"', "'")

        # Second message: Send the query with context reference and summary
        message_text = clean_summary
        if user_prompt:
            message_text = f"{user_prompt}\n\n{message_text}"
            
        query_message = client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=[{
                "type": "text",
                "text": "Consider the uploaded file and analyze: " + message_text
            }]
        )
        debug(f"Query message created: {query_message.id}", debug_output)

        # STEP 5: Run Analysis
        run: Run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=get_assistant().id,
            instructions="Provide your results as a JSON object.",
            response_format={
                'type': 'json_schema',
                'json_schema': {
                    'name': 'analysis_response',
                    'schema': {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "The complete Python code used for analysis"
                            },
                            "steps": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of steps taken in the analysis"
                            },
                            "results": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of results found during analysis"
                            },
                            "final_answer": {
                                "type": "string",
                                "description": "The clear, concise final answer to the query"
                            }
                        },
                        "required": ["code", "steps", "results", "final_answer"]
                    }
                }
            }
        )
        debug(f"\nAnalysis started: {run.id}", debug_output)

        # STEP 6: Wait for Completion
        timeout = 300  # 5 minutes
        start_time = time.time()
        max_attempts = 15
        initial_wait = 4  # Initial wait time in seconds
        subsequent_wait = 2  # Subsequent wait time in seconds
        attempt = 0
        
        while attempt < max_attempts:
            if time.time() - start_time > timeout:
                raise TimeoutError("Analysis timed out after 5 minutes")

            run_status = client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run.id
            )
            
            debug(f"Current status: {run_status.status}", debug_output)
            
            if run_status.status == 'completed':
                # Get the latest message which contains the response
                messages = client.beta.threads.messages.list(
                    thread_id=thread_id,
                    order="desc",
                    limit=1
                )
                
                if not messages.data:
                    raise ValueError("No response message found")
                
                response_message = messages.data[0]
                response_text = response_message.content[0].text.value
                
                try:
                    # Parse JSON response
                    response_text = response_text.strip()
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    
                    if json_start >= 0 and json_end > json_start:
                        json_str = response_text[json_start:json_end].strip()
                        debug(f"\nExtracted JSON:\n{json_str}\n", debug_output)
                        response_json = json.loads(json_str)
                        
                        # Handle nested response in properties
                        if "properties" in response_json:
                            response_json = response_json["properties"]
                        
                        # Validate response format
                        required_fields = ['code', 'steps', 'results', 'final_answer']
                        missing_fields = [f for f in required_fields if f not in response_json]
                        
                        if missing_fields:
                            raise ValueError(f"Response missing required fields: {missing_fields}")
                            
                        debug("\n=== Final Analysis ===", debug_output)
                        debug(json.dumps(response_json, indent=2), debug_output)
                        debug("===================\n", debug_output)

                        # Return successful response
                        return {
                            'status': 'success',
                            'response': response_json,
                            'thread_id': thread_id,
                            'file_id': file_id,
                            'debug_output': '\n'.join(debug_output)
                        }
                    else:
                        raise ValueError("No JSON found in response")
                        
                except (json.JSONDecodeError, ValueError) as e:
                    debug(f"Error parsing response: {str(e)}", debug_output)
                    return {
                        'status': 'error',
                        'error': str(e),
                        'debug_output': '\n'.join(debug_output)
                    }
                
                break
            elif run_status.status in ['failed', 'cancelled', 'expired']:
                debug(f"Attempt {attempt + 1}/{max_attempts} failed with status: {run_status.status}", debug_output)
                if attempt == max_attempts - 1:
                    error_details = f"Final status: {run_status.status}"
                    if hasattr(run_status, 'last_error'):
                        error_details += f", Error: {run_status.last_error}"
                    return {
                        'status': 'error',
                        'error': f"Analysis failed after {max_attempts} attempts. {error_details}",
                        'debug_output': '\n'.join(debug_output)
                    }
                attempt += 1
                wait_time = initial_wait if attempt == 0 else subsequent_wait
                time.sleep(wait_time)
                continue
            elif run_status.status in ['queued', 'in_progress']:
                debug(f"Attempt {attempt + 1}/{max_attempts}: Status {run_status.status}", debug_output)
                if attempt == max_attempts - 1:
                    # Cancel the run before returning error
                    try:
                        client.beta.threads.runs.cancel(
                            thread_id=thread_id,
                            run_id=run.id
                        )
                        debug(f"Cancelled run {run.id} after max attempts", debug_output)
                    except Exception as e:
                        debug(f"Error cancelling run: {str(e)}", debug_output)
                    return {
                        'status': 'error',
                        'error': f"Analysis timed out after {max_attempts} attempts. Last status: {run_status.status}",
                        'debug_output': '\n'.join(debug_output)
                    }
                wait_time = initial_wait if attempt == 0 else subsequent_wait
                time.sleep(wait_time)
                attempt += 1
            else:
                # Unknown status
                debug(f"Unexpected status '{run_status.status}' on attempt {attempt + 1}/{max_attempts}", debug_output)
                if attempt == max_attempts - 1:
                    # Cancel the run before returning error
                    try:
                        client.beta.threads.runs.cancel(
                            thread_id=thread_id,
                            run_id=run.id
                        )
                        debug(f"Cancelled run {run.id} after unexpected status", debug_output)
                    except Exception as e:
                        debug(f"Error cancelling run: {str(e)}", debug_output)
                    return {
                        'status': 'error',
                        'error': f"Analysis failed with unexpected status: {run_status.status}",
                        'debug_output': '\n'.join(debug_output)
                    }

    except Exception as e:
        error_msg = f"Analysis error: {str(e)}"
        debug("\nError Details:", debug_output)
        debug(f"- Error type: {type(e).__name__}", debug_output)
        debug(f"- Error message: {str(e)}", debug_output)
        debug(f"- Error location: {e.__traceback__.tb_frame.f_code.co_name}", debug_output)
        debug(f"- Line number: {e.__traceback__.tb_lineno}", debug_output)
        return {
            'status': 'error',
            'error': error_msg,
            'debug_output': '\n'.join(debug_output)
        }

    finally:
        # Resource Cleanup
        try:
            debug("=== Analysis Complete ===\n", debug_output)
        except Exception as e:
            debug(f"Cleanup error: {str(e)}", debug_output)
        
        # Always return debug output even if we hit finally
        if not 'debug_output' in locals():
            return {
                'status': 'error',
                'error': 'Analysis terminated unexpectedly',
                'debug_output': '\n'.join(debug_output)
            }

# Export the main function
__all__ = ['analyze_data'] 