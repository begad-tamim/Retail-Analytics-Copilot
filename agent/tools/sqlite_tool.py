"""
SQLite tool for querying the Northwind database.
Provides schema inspection and SQL execution with error handling.
"""
import sqlite3
from typing import Dict, List, Optional, Any


class SQLiteTool:
    """Wrapper for SQLite database operations."""
    
    def __init__(self, db_path: str = "data/northwind.sqlite"):
        """Initialize connection to SQLite database.
        
        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    
    def get_schema_summary(self) -> str:
        """Get a summary of the database schema.
        
        Returns:
            String containing table names and their columns.
        """
        cursor = self.conn.cursor()
        
        # Get all tables
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' 
            ORDER BY name
        """)
        tables = cursor.fetchall()
        
        schema_parts = []
        for table in tables:
            table_name = table[0]
            
            # Get column information for each table
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            column_info = []
            for col in columns:
                col_name = col[1]
                col_type = col[2]
                column_info.append(f"  - {col_name} ({col_type})")
            
            schema_parts.append(f"Table: {table_name}")
            schema_parts.extend(column_info)
            schema_parts.append("")  # Empty line between tables
        
        return "\n".join(schema_parts)
    
    def execute_sql(self, sql: str) -> Dict[str, Any]:
        """Execute a SQL query and return results.
        
        Args:
            sql: SQL query string to execute.
            
        Returns:
            Dictionary with keys:
                - columns: List of column names
                - rows: List of dictionaries (one per row)
                - error: Error message if execution failed (None on success)
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql)
            
            # For SELECT queries, fetch results
            if sql.strip().upper().startswith('SELECT'):
                columns = [description[0] for description in cursor.description]
                rows = []
                for row in cursor.fetchall():
                    row_dict = {}
                    for idx, col in enumerate(columns):
                        row_dict[col] = row[idx]
                    rows.append(row_dict)
                
                return {
                    "columns": columns,
                    "rows": rows,
                    "error": None
                }
            else:
                # For non-SELECT queries (INSERT, UPDATE, DELETE)
                self.conn.commit()
                return {
                    "columns": [],
                    "rows": [],
                    "error": None
                }
        
        except Exception as e:
            return {
                "columns": [],
                "rows": [],
                "error": str(e)
            }
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
    
    def __del__(self):
        """Ensure connection is closed when object is destroyed."""
        self.close()


# Global instance
_sqlite_tool = None


def get_sqlite_tool(db_path: str = "data/northwind.sqlite") -> SQLiteTool:
    """Get or create a singleton SQLite tool instance.
    
    Args:
        db_path: Path to the SQLite database file.
        
    Returns:
        SQLiteTool instance.
    """
    global _sqlite_tool
    if _sqlite_tool is None:
        _sqlite_tool = SQLiteTool(db_path)
    return _sqlite_tool
