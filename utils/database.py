import sqlite3
from pathlib import Path
from typing import Optional
import pandas as pd
from utils.logger import get_logger

logger = get_logger(__name__)

class DatabaseManager:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or "data/analysis.db"
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.logger = logger

    def create_connection(self) -> sqlite3.Connection:
        """Create a database connection."""
        try:
            conn = sqlite3.connect(self.db_path)
            self.logger.debug(f"Connected to database: {self.db_path}")
            return conn
        except Exception as e:
            self.logger.error(f"Error connecting to database: {e}")
            raise

    def create_table_from_df(self, df: pd.DataFrame, table_name: str) -> None:
        """Create a table from a pandas DataFrame."""
        try:
            conn = self.create_connection()
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            self.logger.debug(f"Created table '{table_name}' with {len(df)} rows")
        except Exception as e:
            self.logger.error(f"Error creating table: {e}")
            raise
        finally:
            conn.close()

    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute a SQL query and return results as DataFrame."""
        try:
            conn = self.create_connection()
            result = pd.read_sql_query(query, conn)
            self.logger.debug(f"Executed query: {query}")
            return result
        except Exception as e:
            self.logger.error(f"Error executing query: {e}")
            raise
        finally:
            conn.close()

    def get_schema(self) -> str:
        """Get the database schema as a string."""
        try:
            conn = self.create_connection()
            cursor = conn.cursor()
            
            schema_parts = []
            for table in cursor.execute("SELECT name FROM sqlite_master WHERE type='table';"):
                table_name = table[0]
                columns = cursor.execute(f"PRAGMA table_info({table_name});").fetchall()
                
                schema_parts.append(f"Table: {table_name}")
                for col in columns:
                    schema_parts.append(f"  - {col[1]} ({col[2]})")
                schema_parts.append("")
            
            return "\n".join(schema_parts)
        except Exception as e:
            self.logger.error(f"Error getting schema: {e}")
            raise
        finally:
            conn.close() 