#!/usr/bin/env python3
import json
import os
import re
from typing import Any, Dict, List, Optional

import pandas as pd

import anthropic
from dotenv import load_dotenv

load_dotenv()

class AnthropicSQLGenerator:
    def __init__(self):
        self.client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )

    @staticmethod
    def _normalize_sql_output(raw_text: str) -> str:
        """Extract a usable SQL statement from LLM output."""
        if not raw_text:
            return raw_text

        text = raw_text.strip()

        # Prefer content inside fenced code blocks if present
        code_blocks = re.findall(r"```(?:sql)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
        if code_blocks:
            text = code_blocks[0].strip()

        # Strip leading explanatory sentences before the actual SQL
        sql_match = re.search(
            r"(?is)\b(WITH|SELECT|INSERT|UPDATE|DELETE)\b.*", text
        )
        if sql_match:
            text = sql_match.group(0).strip()

        # Remove trailing explanations after the SQL statement
        end_match = re.search(r";[^;]*$", text, flags=re.DOTALL)
        if end_match and end_match.start() != -1:
            tail = text[end_match.start():]
            if not re.search(r";\s*\Z", tail):
                text = text[: end_match.end()].strip()

        return text.strip()
    
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
    
    @staticmethod
    def _extract_json_payload(text: str) -> Optional[Dict[str, Any]]:
        """Find and parse the first JSON object in a text blob."""
        if not text:
            return None

        text = text.strip()
        json_matches = re.findall(r"\{.*\}", text, flags=re.DOTALL)
        for candidate in json_matches:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        try:
            return json.loads(text)
        except Exception:
            return None

    def generate_query_sql(self, question: str, profile: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate SQL query (and metadata) from natural language question."""
        
        columns_info = []
        for col in profile['columns']:
            col_info = f"- {col['name']}: {col['duckdb_type']} ({col['semantic_type']})"
            if 'examples' in col:
                col_info += f" - Examples: {col['examples'][:3]}"
            columns_info.append(col_info)

        dataset = profile.get("dataset", {})
        table_name = dataset.get("table_name") or dataset.get("filename", "uploaded_data").replace(".csv", "").replace("-", "_")
        
        prompt = f"""
        You are a data analysis assistant working with DuckDB.

        Dataset information:
        - Table name: {table_name}
        - Row count: {profile['dataset']['row_count']}
        - Columns:
        {chr(10).join(columns_info)}

        Task:
        - Understand the user question: "{question}"
        - Produce a valid DuckDB SELECT query that answers the question.
        - Provide a concise insight explaining what the query is expected to show.
        - Mention any assumptions or warnings only if necessary.

        Response format:
        {{
          "sql_query": "...",      // single DuckDB-compatible SELECT statement targeting {table_name}
          "insights": "...",       // short natural-language summary of the expected result
          "notes": "...",          // optional notes or assumptions (use "" if none)
          "follow_up_questions": [] // optional list of strings with follow-up ideas, empty list if none
        }}

        Requirements:
        - Return ONLY the JSON object (no markdown, no explanation outside JSON).
        - The SQL must reference only the table {table_name}.
        - Do NOT include read_csv_auto, create table, drop table, or comments in the SQL.
        """
        
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            raw_text = response.content[0].text.strip()
            payload = self._extract_json_payload(raw_text)
            if not payload:
                print("Failed to parse JSON from LLM response; falling back to normalization.")
                sql_text = self._normalize_sql_output(raw_text)
                return {
                    "sql_query": sql_text,
                    "insights": "",
                    "notes": "",
                    "follow_up_questions": [],
                }
            if "sql_query" in payload and isinstance(payload["sql_query"], str):
                payload["sql_query"] = self._normalize_sql_output(payload["sql_query"])
            payload["insights"] = payload.get("insights") or ""
            payload["notes"] = payload.get("notes") or ""
            follow_ups = payload.get("follow_up_questions")
            if isinstance(follow_ups, list):
                payload["follow_up_questions"] = follow_ups
            elif follow_ups is None or follow_ups == "":
                payload["follow_up_questions"] = []
            else:
                payload["follow_up_questions"] = [str(follow_ups)]
            return payload
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
        self.table_name: Optional[str] = None
    
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
        profile.setdefault("dataset", {})
        profile["dataset"]["table_name"] = table_name
        self.table_name = table_name
        load_sql = f"INSERT INTO {table_name} SELECT * FROM read_csv_auto('{csv_path}')"
        self.duckdb_connection.execute(load_sql)
        
        print(f"✅ Data loaded into DuckDB table: {table_name}")
    
    def ask_question(self, question: str, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Process natural language question and return structured response."""

        if not self.duckdb_connection:
            return {"error": "Database not initialized. Please upload a CSV first."}

        # Generate SQL query
        generation = self.sql_generator.generate_query_sql(question, profile)
        if not generation:
            return {"error": "Sorry, I couldn't generate a query for that question."}

        if isinstance(generation, dict):
            sql_query = generation.get("sql_query")
            insights = generation.get("insights", "")
            notes = generation.get("notes", "")
            follow_ups = generation.get("follow_up_questions", [])
        else:
            sql_query = generation
            insights = ""
            notes = ""
            follow_ups = []

        if not sql_query:
            return {"error": "The assistant did not provide a SQL query to execute."}

        print(f"Generated SQL: {sql_query}")

        if "read_csv_auto" in sql_query.lower() and self.table_name:
            sql_query = re.sub(
                r"read_csv_auto\s*\([^)]*\)",
                self.table_name,
                sql_query,
                flags=re.IGNORECASE,
            )
            print(f"Rewritten SQL to use table name: {sql_query}")

        try:
            # Execute query
            cursor = self.duckdb_connection.execute(sql_query)
            try:
                result_df = cursor.fetch_df()
            except Exception:
                rows = cursor.fetchall()
                col_names = [desc[0] for desc in cursor.description] if hasattr(cursor, "description") else []
                result_df = pd.DataFrame(rows, columns=col_names or None)

            # Limit the number of rows returned to avoid overwhelming the UI
            MAX_ROWS = 200
            if len(result_df.index) > MAX_ROWS:
                display_df = result_df.head(MAX_ROWS)
            else:
                display_df = result_df

            result_records = display_df.to_dict(orient="records")
            result_columns = list(display_df.columns)

            answer_text = insights
            if not answer_text:
                answer_text = self.sql_generator.format_response(
                    question, result_df.to_dict(orient="records"), profile
                )

            return {
                "answer": answer_text,
                "sql": sql_query,
                "columns": result_columns,
                "rows": result_records,
                "truncated": len(result_df.index) > MAX_ROWS,
                "notes": notes,
                "follow_up_questions": follow_ups,
            }

        except Exception as e:
            return {
                "error": f"Error executing query: {e}",
                "sql": sql_query,
                "notes": notes,
                "follow_up_questions": follow_ups,
            }
    
    def close(self):
        """Close database connection"""
        if self.duckdb_connection:
            self.duckdb_connection.close()
