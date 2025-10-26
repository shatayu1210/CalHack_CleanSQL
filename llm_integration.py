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
        for col in profile.get("columns", []):
            name = col.get("name", "")
            dtype = col.get("duckdb_type", "UNKNOWN")
            semantic = col.get("semantic_type", "unknown")
            parts = [f"- {name}: {dtype}", f"semantic={semantic}"]
            null_ratio = col.get("null_ratio")
            if null_ratio is not None:
                parts.append(f"null_pct≈{round(float(null_ratio) * 100, 1)}%")
            parse_fail = col.get("parse_fail_pct")
            if parse_fail is not None:
                parts.append(f"parse_fail≈{round(float(parse_fail), 1)}%")
            if "examples" in col and col["examples"]:
                examples = col["examples"][:3]
                parts.append(f"examples={examples}")
            columns_info.append("  " + ", ".join(parts))

        dataset = profile.get("dataset", {})
        table_name = dataset.get("table_name") or dataset.get("filename", "uploaded_data").replace(".csv", "").replace("-", "_")
        profile_notes = profile.get("notes") or []
        notes_section = ""
        if profile_notes:
            notes_section = "\nProfile notes:\n" + "\n".join(f"- {note}" for note in profile_notes)

        rules_text = """
Default cleaning rules:
  (R1) Numeric 'study_hours' -> impute with SUBJECT median.
  (R2) Categorical 'grade_level' -> impute with GLOBAL mode.
  (R3) Categorical 'gender' -> impute with literal 'U'.
  (R4) Treat empty strings as NULL; trim and lower-case where grouping needs it.
  (R5) Parse 'exam_date' using ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%m/%d/%Y %H:%M', '%m/%d/%Y'].
  (R6) Apply imputations only to columns actually referenced in the query.
"""
        
        prompt = f"""
You are CleanSQL's analytics assistant working with DuckDB.

Dataset:
  - Table: {table_name}
  - Rows: {profile['dataset']['row_count']}
  - Columns:
{chr(10).join(columns_info) if columns_info else '  (none listed)'}
{notes_section}

{rules_text}

Task:
  1. Understand the user question: "{question}"
  2. Produce TWO DuckDB SQL statements:
     a. "raw_query": A straightforward SELECT that assumes clean data (no CTEs, no imputations).
     b. "robust_query": A production-ready DuckDB SQL using CTEs that applies the default rules above.
  3. Summarize any data-quality handling in a short clause for "data_quality_note".
  4. List optional follow-up questions (if any).

Formatting requirements:
{{
  "raw_query": "...",              // plain SELECT referencing only {table_name}
  "robust_query": "...",           // DuckDB SQL with CTEs + imputations
  "data_quality_note": "...",      // one sentence, mention imputations/cleaning actually used
  "notes": "...",                  // optional assumptions (empty string if none)
  "follow_up_questions": []        // optional string list, [] if none
}}

Rules:
  - Return ONLY the JSON object (no markdown fences or prose outside JSON).
  - Both queries must reference ONLY the table {table_name}.
  - Use TIMESTAMP literals for time comparisons where helpful.
  - Do not include INSERT/CREATE/DROP statements.
  - Ensure the robust query's final SELECT provides answer-ready columns.
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
                    "raw_query": sql_text,
                    "robust_query": sql_text,
                    "insights": "",
                    "notes": "",
                    "data_quality_note": "",
                    "follow_up_questions": [],
                }
            if "raw_query" in payload and isinstance(payload["raw_query"], str):
                payload["raw_query"] = self._normalize_sql_output(payload["raw_query"])
            if "robust_query" in payload and isinstance(payload["robust_query"], str):
                payload["robust_query"] = self._normalize_sql_output(payload["robust_query"])
            if "sql_query" in payload and not payload.get("robust_query"):
                payload["robust_query"] = self._normalize_sql_output(payload["sql_query"])
            if not payload.get("raw_query") and payload.get("sql_query"):
                payload["raw_query"] = self._normalize_sql_output(payload["sql_query"])
            payload["insights"] = payload.get("insights") or ""
            payload["notes"] = payload.get("notes") or ""
            payload["data_quality_note"] = payload.get("data_quality_note") or ""
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
You are rewriting analytics results for an end user.

Question: "{question}"
Dataset: {profile['dataset']['filename']}
SQL Result: {json.dumps(sql_result, indent=2)}

Respond with a single concise sentence that directly answers the question, includes the key numbers from the result, and avoids extra commentary.
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

    @staticmethod
    def _sanitize_sql(sql_text: Optional[str]) -> Optional[str]:
        """Trim stray markdown/quote wrappers that break DuckDB parsing."""
        if not sql_text or not isinstance(sql_text, str):
            return sql_text

        cleaned = sql_text.strip()
        if not cleaned:
            return cleaned

        wrappers = ('"""', "'''", "```", '"', "'")
        for wrapper in wrappers:
            if cleaned.startswith(wrapper) and cleaned.endswith(wrapper):
                cleaned = cleaned[len(wrapper) : -len(wrapper)].strip()
                break

        # Handle dangling trailing quote without a matching opener
        if cleaned.endswith('"') and cleaned.count('"') % 2 == 1:
            cleaned = cleaned[:-1].rstrip()
        if cleaned.endswith("'") and cleaned.count("'") % 2 == 1:
            cleaned = cleaned[:-1].rstrip()

        return cleaned
    
    def setup_database(self, profile: Dict[str, Any], csv_path: str):
        """Setup DuckDB database with generated schema"""
        import duckdb

        self.duckdb_connection = duckdb.connect()

        raw_table_name = profile.get("dataset", {}).get("filename", "uploaded_data")
        base_name = (
            raw_table_name.replace(".csv", "")
            .replace(".xlsx", "")
            .replace(".xls", "")
        )
        sanitized = re.sub(r"\W+", "_", base_name).strip("_")
        if not sanitized:
            sanitized = "uploaded_data"
        if sanitized[0].isdigit():
            sanitized = f"t_{sanitized}"
        table_name = sanitized.lower()

        profile.setdefault("dataset", {})
        profile["dataset"]["table_name"] = table_name
        self.table_name = table_name

        create_sql = f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT * FROM read_csv_auto(?, header=True)
        """
        print("Loading dataset into DuckDB with inferred schema...")
        self.duckdb_connection.execute(create_sql, [csv_path])
        
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
            raw_sql = generation.get("raw_query")
            robust_sql = generation.get("robust_query") or raw_sql
            insights = generation.get("insights", "")
            notes = generation.get("notes", "")
            follow_ups = generation.get("follow_up_questions", [])
            data_note = generation.get("data_quality_note", "")
        else:
            raw_sql = generation
            robust_sql = generation
            insights = ""
            notes = ""
            follow_ups = []
            data_note = ""

        if not raw_sql and not robust_sql:
            return {"error": "The assistant did not provide a SQL query to execute."}

        cleaned_queries = {}
        for key, sql_text in (("raw_sql", raw_sql), ("robust_sql", robust_sql)):
            if sql_text and self.table_name and "read_csv_auto" in sql_text.lower():
                cleaned_queries[key] = re.sub(
                    r"read_csv_auto\s*\([^)]*\)",
                    self.table_name,
                    sql_text,
                    flags=re.IGNORECASE,
                )
            else:
                cleaned_queries[key] = sql_text

        raw_sql = cleaned_queries.get("raw_sql")
        robust_sql = cleaned_queries.get("robust_sql")

        raw_sql = self._sanitize_sql(raw_sql)
        robust_sql = self._sanitize_sql(robust_sql)

        sql_to_execute = robust_sql or raw_sql

        print(f"Generated raw SQL: {raw_sql}")
        if robust_sql and robust_sql != raw_sql:
            print(f"Generated robust SQL: {robust_sql}")

        try:
            # Execute query
            cursor = self.duckdb_connection.execute(sql_to_execute)
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

            answer_sentence = insights
            if not answer_sentence:
                answer_sentence = self.sql_generator.format_response(
                    question, result_df.to_dict(orient="records"), profile
                )
            answer_sentence = (answer_sentence or "").strip()
            if answer_sentence:
                answer_sentence = " ".join(answer_sentence.split())

            formatted_data_note = (data_note or "").strip()
            if formatted_data_note:
                note_line = formatted_data_note
            else:
                note_line = "No additional data-quality adjustments were applied."
            final_answer = f"Question: {question}"
            final_answer += f"\nAnswer: {answer_sentence}" if answer_sentence else "\nAnswer: See robust query results below."
            final_answer += f"\nData note: {note_line}"

            return {
                "answer": final_answer,
                "sql": sql_to_execute,
                "raw_sql": raw_sql,
                "robust_sql": robust_sql or raw_sql,
                "columns": result_columns,
                "rows": result_records,
                "truncated": len(result_df.index) > MAX_ROWS,
                "notes": notes,
                "data_quality_note": note_line,
                "follow_up_questions": follow_ups,
            }

        except Exception as e:
            return {
                "error": f"Error executing query: {e}",
                "sql": sql_to_execute,
                "raw_sql": raw_sql,
                "robust_sql": robust_sql or raw_sql,
                "notes": notes,
                "data_quality_note": data_note or "No additional data-quality adjustments were applied.",
                "follow_up_questions": follow_ups,
            }
    
    def close(self):
        """Close database connection"""
        if self.duckdb_connection:
            self.duckdb_connection.close()
