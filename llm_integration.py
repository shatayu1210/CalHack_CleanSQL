#!/usr/bin/env python3
import json
import os
from typing import Any, Dict, List, Optional

import anthropic
from dotenv import load_dotenv

load_dotenv()

class AnthropicSQLGenerator:
    def __init__(self):
        self.client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
    
    def generate_schema_sql(self, profile: Dict[str, Any]) -> str:
        """Generate SQL schema creation from data profile"""
        
        prompt = f"""
        Based on this CSV data profile, generate a simple SQL schema for DuckDB.
        
        Dataset Info:
        - File: {profile['dataset']['filename']}
        - Rows: {profile['dataset']['row_count']}
        - Columns: {profile['dataset']['column_count']}
        
        Column Details:
        {json.dumps(profile['columns'], indent=2)}
        
        Generate ONLY a CREATE TABLE statement with appropriate data types.
        Do NOT include indexes, constraints, or ALTER TABLE statements.
        Use simple DuckDB-compatible syntax.
        
        Return only the CREATE TABLE statement, no explanations.
        """
        
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            # Clean the response to remove markdown formatting
            sql_text = response.content[0].text
            # Remove ```sql and ``` markers
            sql_text = sql_text.replace('```sql', '').replace('```', '').strip()
            return sql_text
        except Exception as e:
            print(f"Error generating schema: {e}")
            return self._fallback_schema(profile)
    
    def generate_query_sql(self, question: str, profile: Dict[str, Any]) -> str:
        """Generate SQL query from natural language question"""
        
        columns_info = []
        for col in profile['columns']:
            col_info = f"- {col['name']}: {col['duckdb_type']} ({col['semantic_type']})"
            if 'examples' in col:
                col_info += f" - Examples: {col['examples'][:3]}"
            columns_info.append(col_info)
        
        prompt = f"""
        Generate a SQL query for DuckDB based on this question: "{question}"
        
        Available columns:
        {chr(10).join(columns_info)}
        
        Dataset: {profile['dataset']['filename']} ({profile['dataset']['row_count']} rows)
        
        Return only the SQL query, no explanations. Use proper DuckDB syntax.
        """
        
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            # Clean the response to remove markdown formatting
            sql_text = response.content[0].text.strip()
            sql_text = sql_text.replace('```sql', '').replace('```', '').strip()
            return sql_text
        except Exception as e:
            print(f"Error generating query: {e}")
            return None
    
    def format_response(self, question: str, sql_result: Any, profile: Dict[str, Any]) -> str:
        """Format SQL results into human-friendly response"""
        
        prompt = f"""
        Format this SQL query result into a natural, human-friendly response.
        
        Question: "{question}"
        Dataset: {profile['dataset']['filename']}
        
        SQL Result: {json.dumps(sql_result, indent=2)}
        
        Provide a clear, conversational answer that directly addresses the question.
        Include relevant numbers and insights.
        """
        
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            print(f"Error formatting response: {e}")
            return f"Query result: {sql_result}"
    
    def _fallback_schema(self, profile: Dict[str, Any]) -> str:
        """Fallback schema generation if LLM fails"""
        table_name = profile['dataset']['filename'].replace('.csv', '').replace('-', '_')
        
        sql_parts = [f"CREATE TABLE {table_name} ("]
        
        for col in profile['columns']:
            col_name = col['name'].replace(' ', '_').replace('-', '_')
            sql_parts.append(f"    {col_name} {col['duckdb_type']},")
        
        sql_parts[-1] = sql_parts[-1].rstrip(',')  # Remove last comma
        sql_parts.append(");")
        
        return '\n'.join(sql_parts)

class DataAssistant:
    def __init__(self):
        self.sql_generator = AnthropicSQLGenerator()
        self.duckdb_connection = None
    
    def setup_database(self, profile: Dict[str, Any], csv_path: str):
        """Setup DuckDB database with generated schema"""
        import duckdb

        # Generate schema
        schema_sql = self.sql_generator.generate_schema_sql(profile)
        print("Generated Schema:")
        print(schema_sql)
        
        # Connect to DuckDB
        self.duckdb_connection = duckdb.connect()
        
        # Create table
        self.duckdb_connection.execute(schema_sql)
        
        # Load data
        table_name = profile['dataset']['filename'].replace('.csv', '').replace('-', '_')
        load_sql = f"INSERT INTO {table_name} SELECT * FROM read_csv_auto('{csv_path}')"
        self.duckdb_connection.execute(load_sql)
        
        print(f"✅ Data loaded into DuckDB table: {table_name}")
    
    def ask_question(self, question: str, profile: Dict[str, Any]) -> str:
        """Process natural language question and return formatted response"""
        
        if not self.duckdb_connection:
            return "Database not initialized. Please upload a CSV first."
        
        # Generate SQL query
        sql_query = self.sql_generator.generate_query_sql(question, profile)
        if not sql_query:
            return "Sorry, I couldn't generate a query for that question."
        
        print(f"Generated SQL: {sql_query}")
        
        try:
            # Execute query
            result = self.duckdb_connection.execute(sql_query).fetchall()
            
            # Format response
            formatted_response = self.sql_generator.format_response(question, result, profile)
            return formatted_response
            
        except Exception as e:
            return f"Error executing query: {e}"
    
    def close(self):
        """Close database connection"""
        if self.duckdb_connection:
            self.duckdb_connection.close()
