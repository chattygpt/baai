import pandas as pd
import sys
from pathlib import Path
import time
import os
import json
from datetime import datetime

# Add project root to path first
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# Load environment variables from .env file
env_path = project_root / '.env'
load_dotenv(env_path)

# Verify API key is loaded
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")

from openai import OpenAI
from agents import run_analysis
from utils.setup import setup_project, debug

# Initialize OpenAI client with API key from environment
client = OpenAI(api_key=api_key)

def create_test_assistant():
    """Create an OpenAI Assistant for testing."""
    return client.beta.assistants.create(
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

def compare_answers(client, actual_answer, expected_answer):
    """Compare actual and expected answers using GPT-4-mini to determine semantic equivalence"""
    prompt = f"""Compare these two answers and determine if they are semantically equivalent (True) or different (False).
    Only respond with 'True' or 'False'.
    
    Expected Answer: {expected_answer}
    Actual Answer: {actual_answer}
    
    Are they semantically equivalent (True/False)?"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using 4.0-mini model
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        result = response.choices[0].message.content.strip()
        return result == "True"
    except Exception as e:
        print(f"Error comparing answers: {e}")
        return False

def run_test_questions():
    # Initialize project
    setup_project()
    
    # Create test assistant and store its ID
    assistant = create_test_assistant()
    assistant_id = assistant.id
    print(f"Created test assistant with ID: {assistant_id}")
    
    # Create a mock session state for compatibility
    class MockSessionState:
        def __init__(self):
            self._data = {
                'openai_assistant': assistant,
                'file_uploaded': False,
                'debug_mode': True,
                'current_file_id': None,
                'conversation_history': [],
                'current_thread_id': None,
                'debug_output': []
            }
            
        def __getattr__(self, name):
            return self._data.get(name)
            
        def get(self, key, default=None):
            return self._data.get(key, default)
            
        def __setattr__(self, name, value):
            if name == '_data':
                super().__setattr__(name, value)
            else:
                self._data[name] = value
                
        def __contains__(self, key):
            return key in self._data
                
        def write(self, msg):
            """Mock Streamlit's write function for debug output"""
            if self._data.get('debug_mode'):
                print(f"[DEBUG] {msg}")
                self._data['debug_output'].append(msg)
                
        def __iter__(self):
            return iter(self._data)
            
        def __len__(self):
            return len(self._data)
            
        def keys(self):
            return self._data.keys()
            
        def values(self):
            return self._data.values()
            
        def items(self):
            return self._data.items()
            
    # Set up mock Streamlit session state
    import streamlit as st
    mock_state = MockSessionState()
    sys.modules['streamlit'].session_state = mock_state
    
    # Also mock st.write for debug output
    sys.modules['streamlit'].write = mock_state.write
    
    print("\n=== Test Run Starting ===")
    print(f"Debug mode: {mock_state.debug_mode}")
    
    # Initialize file paths
    file_path = Path(project_root) / 'uploads' / 'mock_data.csv'
    if not file_path.exists():
        print(f"Error: Could not find {file_path}")
        return
        
    questions_file = Path(project_root) / 'uploads' / '100qs_part2.csv'
    if not questions_file.exists():
        print(f"Error: Could not find {questions_file}")
        return
        
    # Load questions
    try:
        # Try UTF-8 with BOM first
        questions_df = pd.read_csv(questions_file, encoding='utf-8-sig')
    except UnicodeDecodeError:
        try:
            # Try latin1 as fallback
            questions_df = pd.read_csv(questions_file, encoding='latin1')
        except Exception as e:
            print(f"Error reading questions file with latin1 encoding: {e}")
            try:
                # Try cp1252 as last resort (common for Windows files)
                questions_df = pd.read_csv(questions_file, encoding='cp1252')
            except Exception as e:
                print(f"Failed to read questions file with any encoding: {e}")
                return

    total_questions = len(questions_df)
    print(f"Successfully loaded {total_questions} questions from {questions_file}")
    
    # Check for required columns
    required_columns = ['Question']
    golden_answer_column = None
    for possible_name in ['Answer', 'Golden Answer', 'GoldenAnswer', 'Golden_Answer', 'Expected Answer', 'ExpectedAnswer']:
        if possible_name in questions_df.columns:
            golden_answer_column = possible_name
            break
    
    if golden_answer_column is None:
        print("Warning: No golden answer column found. Test status will not be computed.")
    else:
        print(f"Using '{golden_answer_column}' as the golden answer column")
    
    print(f"Found columns: {', '.join(questions_df.columns)}")
    
    # Prepare results dataframe - use all questions
    test_df = questions_df.copy()
    test_df['ai_answer'] = ''
    test_df['status'] = ''
    test_df['error'] = ''
    test_df['test_status'] = False  # Add test status column
    test_df['run_id'] = datetime.now().strftime('%Y%m%d_%H%M%S')
    test_df['thread_id'] = ''
    test_df['file_id'] = ''
    test_df['timestamp'] = ''
    test_df['duration_seconds'] = 0.0
    test_df['debug_output'] = ''
    test_df['code'] = ''
    test_df['steps'] = ''
    test_df['results'] = ''
    test_df['conversation_history'] = ''
    test_df['raw_response'] = ''
    
    # Initialize analysis with mock data
    print("\nInitializing analysis...")
    init_result = run_analysis(
        query="Initialize data analysis",
        file_path=str(file_path),
        debug_mode=True,
        initialize=True
    )
    
    if init_result.get('status') != 'success':
        print(f"Failed to initialize: {init_result.get('error', 'Unknown error')}")
        return
        
    # Store initialization results in mock session state
    thread_id = init_result.get('thread_id')
    file_id = init_result.get('file_id')
    mock_state.current_thread_id = thread_id
    mock_state.current_file_id = file_id
    
    print(f"Initialized with thread_id: {thread_id}, file_id: {file_id}")
    
    # Run each question
    start_time = time.time()
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def cleanup_active_runs(thread_id):
        """Cancel any active runs on the thread"""
        try:
            runs = client.beta.threads.runs.list(thread_id=thread_id)
            for run in runs.data:
                if run.status in ['queued', 'in_progress', 'requires_action']:
                    print(f"Canceling active run {run.id}")
                    client.beta.threads.runs.cancel(thread_id=thread_id, run_id=run.id)
                    time.sleep(1)  # Give the API time to process the cancellation
        except Exception as e:
            print(f"Error cleaning up runs: {str(e)}")
    
    for idx, row in test_df.iterrows():
        question = row['Question']
        question_start_time = time.time()
        print(f"\nProcessing question {idx + 1}/{total_questions}: {question}")
        
        try:
            # Clean up any active runs before starting new one
            cleanup_active_runs(thread_id)
            
            result = run_analysis(
                query=question,
                file_path=str(file_path),
                debug_mode=True,
                initialize=False,
                thread_id=thread_id,
                file_id=file_id
            )
            
            # Update basic info
            test_df.at[idx, 'run_id'] = run_id
            test_df.at[idx, 'thread_id'] = thread_id
            test_df.at[idx, 'file_id'] = file_id
            test_df.at[idx, 'timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            test_df.at[idx, 'duration_seconds'] = time.time() - question_start_time
            test_df.at[idx, 'debug_output'] = '\n'.join(result.get('debug_output', '').split('\n')) if result.get('debug_output') else ''
            test_df.at[idx, 'conversation_history'] = str(mock_state.conversation_history)
            test_df.at[idx, 'raw_response'] = str(result)
            
            if result and result.get('status') == 'success':
                response = result.get('response', {})
                if isinstance(response, dict):
                    ai_answer = response.get('final_answer', '')
                    test_df.at[idx, 'ai_answer'] = ai_answer
                    test_df.at[idx, 'code'] = response.get('code', '')
                    test_df.at[idx, 'steps'] = str(response.get('steps', []))
                    test_df.at[idx, 'results'] = str(response.get('results', []))
                    test_df.at[idx, 'status'] = 'success'
                    
                    # Compare with golden answer
                    if golden_answer_column:
                        try:
                            golden_answer = row[golden_answer_column]
                            if pd.isna(golden_answer):
                                print(f"Warning: Golden answer is missing for question {idx + 1}")
                                test_df.at[idx, 'test_status'] = False
                            else:
                                test_status = compare_answers(client, ai_answer, str(golden_answer))
                                test_df.at[idx, 'test_status'] = test_status
                                print(f"Answer comparison result: {'PASS' if test_status else 'FAIL'}")
                        except Exception as e:
                            print(f"Error comparing answers: {str(e)}")
                            test_df.at[idx, 'test_status'] = False
                    else:
                        test_df.at[idx, 'test_status'] = False
                else:
                    test_df.at[idx, 'ai_answer'] = str(response)
                    test_df.at[idx, 'status'] = 'success'
                    test_df.at[idx, 'test_status'] = False
            else:
                test_df.at[idx, 'status'] = 'error'
                test_df.at[idx, 'error'] = result.get('error', 'Unknown error')
                
            # Save progress after each question
            if (idx + 1) % 10 == 0:
                progress_file = Path(project_root) / 'output' / 'test_results_progress.csv'
                try:
                    test_df.to_csv(progress_file, index=False, encoding='utf-8-sig')
                except UnicodeDecodeError:
                    test_df.to_csv(progress_file, index=False, encoding='latin1')
                elapsed_time = time.time() - start_time
                avg_time_per_question = elapsed_time / (idx + 1)
                remaining_questions = total_questions - (idx + 1)
                estimated_remaining_time = remaining_questions * avg_time_per_question
                
                print(f"\nProgress Update:")
                print(f"Completed: {idx + 1}/{total_questions} questions ({((idx + 1)/total_questions)*100:.1f}%)")
                print(f"Success rate: {(test_df['status'] == 'success').sum()/(idx + 1)*100:.1f}%")
                print(f"Elapsed time: {elapsed_time/60:.1f} minutes")
                print(f"Estimated remaining time: {estimated_remaining_time/60:.1f} minutes")
                print(f"Progress saved to: {progress_file}")
                
        except Exception as e:
            print(f"Error processing question: {str(e)}")
            test_df.at[idx, 'status'] = 'error'
            test_df.at[idx, 'error'] = str(e)
        
        # Sleep between questions to avoid rate limiting
        time.sleep(2)
    
    # Save final results
    output_file = Path(project_root) / 'output' / 'test_results_final.csv'
    output_file.parent.mkdir(exist_ok=True)
    try:
        test_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    except UnicodeDecodeError:
        test_df.to_csv(output_file, index=False, encoding='latin1')
    
    # Calculate total time
    total_time = time.time() - start_time
    
    print(f"\nTest complete. Results saved to {output_file}")
    print(f"\nFinal Summary:")
    print(f"Total questions processed: {total_questions}")
    success_count = (test_df['status'] == 'success').sum()
    print(f"Successful: {success_count} ({(success_count/total_questions)*100:.1f}%)")
    print(f"Failed: {total_questions - success_count} ({((total_questions-success_count)/total_questions)*100:.1f}%)")
    
    # Add test status statistics
    test_pass_count = (test_df['test_status'] == True).sum()
    print(f"\nTest Results:")
    print(f"Passed: {test_pass_count} ({(test_pass_count/total_questions)*100:.1f}%)")
    print(f"Failed: {total_questions - test_pass_count} ({((total_questions-test_pass_count)/total_questions)*100:.1f}%)")
    
    print(f"\nTiming:")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average time per question: {total_time/total_questions:.1f} seconds")

if __name__ == "__main__":
    run_test_questions() 