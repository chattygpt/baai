from typing import Dict, Any
from langchain.agents import tool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import pandas as pd
from io import StringIO
from config import DEBUG_MODE

class PythonAgent:
    def __init__(self, llm):
        self.llm = llm
        self.df = None
        self.df_info = None
        self.setup_agent()

    def _get_df_info(self, df):
        """Capture DataFrame info output."""
        buffer = StringIO()
        df.info(buf=buffer)
        return buffer.getvalue()

    def setup_agent(self):
        """Setup the Python agent with tools and prompt."""
        # Define tools
        @tool
        def python_repl(code: str) -> str:
            """Execute Python code and return the result."""
            try:
                # Make DataFrame available to the code
                df = self.df.copy()
                df.columns = df.columns.str.lower()
                
                # Get DataFrame info if not already captured
                if self.df_info is None:
                    self.df_info = self._get_df_info(df)
                
                # Create namespace with DataFrame info
                local_vars = {
                    'df': df,
                    'pd': pd,
                    'df_info': self.df_info,
                    'columns': list(df.columns),
                    'dtypes': df.dtypes.to_dict()
                }
                
                # Execute code
                if '\n' in code:
                    lines = code.strip().split('\n')
                    result = None
                    for line in lines:
                        if line.strip():
                            exec(line, globals(), local_vars)
                            # Store last assignment result
                            if '=' in line:
                                var_name = line.split('=')[0].strip()
                                if var_name in local_vars:
                                    result = local_vars[var_name]
                    return str(result) if result is not None else "Code executed successfully"
                else:
                    result = eval(code, globals(), local_vars)
                    return str(result)
            except Exception as e:
                return f"Error: {str(e)}"

        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Python data analysis expert. You help users analyze data using Python and pandas.
            The data is available in a pandas DataFrame named 'df'.
            The DataFrame columns are automatically converted to lowercase.
            
            DataFrame Information:
            {df_info}
            
            You can access DataFrame information using:
            - df: The DataFrame itself
            - columns: List of column names
            - dtypes: Dictionary of column data types
            
            Important rules:
            1. Calculate step by step
            2. Use proper pandas operations
            3. Show your work clearly
            4. Return the final numerical result
            
            Example query: "What is the sales growth for SKU_123 in Aug compared to Jul?"
            Example steps:
            1. Calculate July sales:
               july_sales = df[(df['sku'] == '123') & (df['date'].dt.month == 7)]['quantity'].sum()
            
            2. Calculate August sales:
               aug_sales = df[(df['sku'] == '123') & (df['date'].dt.month == 8)]['quantity'].sum()
            
            3. Calculate growth:
               growth = ((aug_sales - july_sales) / july_sales) * 100
               round(growth, 2)  # Return this number
            """),
            ("human", "{input}"),
            ("assistant", "I'll help you calculate that."),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Create the agent
        self.agent = create_openai_functions_agent(
            llm=self.llm,
            prompt=prompt,
            tools=[python_repl]
        )
        
        # Create the executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=[python_repl],
            verbose=True,
            handle_parsing_errors=True
        )

    def run(self, query: str) -> str:
        """Run the agent with a query."""
        try:
            # Get DataFrame info if not already captured
            if self.df_info is None and self.df is not None:
                self.df_info = self._get_df_info(self.df)
            
            result = self.agent_executor.invoke({
                "input": query,
                "df_info": self.df_info or "DataFrame not loaded"
            })
            return str(result.get("output", "")) if isinstance(result, dict) else str(result)
        except Exception as e:
            return f"Error processing query: {str(e)}" 