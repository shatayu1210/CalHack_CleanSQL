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

    # --- Begin User-Specified RAG Imputation Pipeline ---
    def choose_policy(dtype, nullp, profile, df):
        """Minimal policy chooser (works today)"""
        if dtype == "numeric":
            if nullp <= 0.01: return {"method":"drop_rows"}
            if nullp <= 0.15:
                key = pick_stable_key(df)
                return {"method":"median", "by":key}
            return {"method":"iterative", "max_iter":15, "by":None, "fallback":"median"}
        if dtype in ("categorical","boolean"):
            key = pick_stable_key(df)
            if nullp <= 0.10: return {"method":"mode", "by":key}
            return {"method":"mode", "by":key, "unknown":"Unknown"}
        if dtype == "date":
            return {"method":"ffill_bfill"}  # or "median_date"
        return {"method":"drop_rows"}

    def pick_stable_key(df):
        """Heuristic: choose 1–2 categorical columns with low cardinality (≤10), good association, enough support (≥30 rows)."""
        cats = [c for c in df.columns if df[c].dtype == 'object' and df[c].nunique() <= 10]
        for c in cats:
            if df[c].value_counts().min() >= 30:
                return c
        return None

    def apply_policies(df, policies, cols):
        """Apply imputations only to columns actually used; return (df_imputed, report)."""
        import numpy as np
        dfq = df.copy()
        report = {}
        for c in cols:
            pol = policies[c]
            null_mask = dfq[c].isnull()
            n_missing = null_mask.sum()
            method = pol.get("method")
            if method == "drop_rows":
                dfq = dfq[~null_mask]
                report[c] = {"imputed":0, "method":"drop_rows"}
            elif method == "median":
                by = pol.get("by")
                if by:
                    medians = dfq.groupby(by)[c].transform('median')
                    dfq.loc[null_mask, c] = medians[null_mask]
                else:
                    med = dfq[c].median()
                    dfq.loc[null_mask, c] = med
                report[c] = {"imputed":int(n_missing), "method":"median", "by":by}
            elif method == "iterative":
                # Placeholder for IterativeImputer or KNN
                dfq.loc[null_mask, c] = dfq[c].median()
                report[c] = {"imputed":int(n_missing), "method":"iterative->median"}
            elif method == "mode":
                by = pol.get("by")
                if by:
                    modes = dfq.groupby(by)[c].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown")
                    dfq.loc[null_mask, c] = modes[null_mask]
                else:
                    mode = dfq[c].mode().iloc[0] if not dfq[c].mode().empty else "Unknown"
                    dfq.loc[null_mask, c] = mode
                report[c] = {"imputed":int(n_missing), "method":"mode", "by":by}
            elif method == "ffill_bfill":
                dfq[c] = dfq[c].fillna(method='ffill').fillna(method='bfill')
                report[c] = {"imputed":int(n_missing), "method":"ffill_bfill"}
            else:
                report[c] = {"imputed":int(n_missing), "method":method}
        return dfq, report
    # --- End User-Specified RAG Imputation Pipeline ---

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

        # Replaced with user-specified robust RAG imputation policy:
        rules_text = '''
Default RAG rules for missing data (robust, works on any dataset):
- LLM should NOT compute numbers from raw tables.
- Parse NL → QuerySpec (columns, filters, group-bys, metric).
- Pull profiles from Weaviate (per column): type (numeric/categorical/boolean/date), min/max/mean/std, allowed_values, null%, duplicate%, etc.
- Pick a policy per column at query time, compute answer in Python (read-only), return number + short imputation report for transparency.
- Policy matrix (see code for details) enforced in Python, not SQL.
'''
        # The actual imputation logic is now Python-side, see choose_policy and apply_policies.
        
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
     b. "robust_query": A DuckDB SQL that creates CTE(s) for all columns referenced in the raw_query, applying the default imputation policy for each column (see rules above). The final SELECT must use these imputed columns and never simply drop rows with missing values unless the policy says so. Example: for G1 with 12% null, use a CTE that imputes missing G1 with group-median, and select from that. For categorical columns like sex with <10% null, impute with mode. Only drop rows if policy is drop_rows. The robust query must always use the imputed columns for all aggregations and filters.
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
  - Ensure the robust query's final SELECT provides answer-ready columns and always uses the imputed columns per policy.
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
        

class DataAssistant:
    def __init__(self):
        self.sql_generator = AnthropicSQLGenerator()
        # Initialize DuckDB in normal in-memory mode (read_only not supported for :memory:)
        import duckdb
        self.duckdb_connection = duckdb.connect(database=':memory:')
        self.table_name: Optional[str] = None
        self.semantic_search = self._semantic_search_weaviate

    def _semantic_search_weaviate(self, question):
        """Returns list of (column, score) by semantic similarity using Weaviate vector search."""
        try:
            from profiler import W_CLIENT, embed_text
            if W_CLIENT is None or embed_text is None:
                return []
            vec = embed_text(question)
            if vec is None:
                return []
            # Query Weaviate for most similar CleanSQLColumn objects
            res = W_CLIENT.query.get(
                "CleanSQLColumn",
                ["column"]
            ).with_near_vector({"vector": vec, "certainty": 0.0}).with_limit(10).with_additional(["certainty"]).do()
            hits = res.get("data", {}).get("Get", {}).get("CleanSQLColumn", [])
            return [(h["column"], h["_additional"].get("certainty", 0)) for h in hits if "column" in h and "_additional" in h]
        except Exception:
            return []

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
            SELECT * FROM parquet_scan(?)
        """
        print("Loading dataset into DuckDB from Parquet (fast/compact)...")
        self.duckdb_connection.execute(create_sql, [csv_path])
        
        print(f"✅ Data loaded into DuckDB table: {table_name}")
    
    def ask_question(self, question: str, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Process natural language question and return structured response."""

        if not self.duckdb_connection:
            return {"error": "Database not initialized. Please upload a CSV first."}

        # --- Heuristic for mutation intent: warn and block ---
        mutation_keywords = ["edit", "update", "remove", "delete", "get rid", "drop", "alter", "truncate"]
        q_lower = question.strip().lower()
        if any(word in q_lower for word in mutation_keywords):
            return {"error": "Sorry! I'm an append-safe assistant and operate in read-only mode! I'm happy to assist you in analysing data :)"}

        # (Removed LLM-guided RAG column selection logic)
        # Optionally, keep semantic search for other logic if needed

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

        # --- Robustly check if raw_sql is valid SQL ---
        import re
        raw_sql_str = (raw_sql or '').strip()
        # Block any mutation queries in raw_sql
        mutation_sql_keywords = ["update", "delete", "insert", "drop", "alter", "truncate"]
        if any(kw in raw_sql_str.lower() for kw in mutation_sql_keywords):
            return {"error": "Sorry! I'm an append-safe assistant and operate in read-only mode! I'm happy to assist you in analysing data :)"}
        # Check for JSON, JSON keys, empty, or not starting with SELECT/WITH
        if (
            not raw_sql_str or
            raw_sql_str.startswith('{') or
            raw_sql_str.startswith('"') or
            'robust_query' in raw_sql_str or
            'raw_query' in raw_sql_str or
            raw_sql_str.count('{') > 0 or
            raw_sql_str.count('}') > 0 or
            not re.match(r'^(select|with)\b', raw_sql_str, re.IGNORECASE)
        ):
            return {"error": "Please elaborate further on what you need precisely."}

        # --- Check if raw SQL is too generic (no aggregation, columns, or conditions) ---
        import sqlparse
        parsed = sqlparse.parse(raw_sql or "")
        tokens = [token.value.lower() for stmt in parsed for token in stmt.flatten() if token.value]
        columns = [c['name'].lower() for c in profile.get('columns',[])]
        aggs = ["avg", "average", "mean", "sum", "count", "min", "max", "median", "mode", "group by", "having"]
        has_col = any(col in " ".join(tokens) for col in columns)
        has_agg = any(agg in " ".join(tokens) for agg in aggs)
        has_condition = any(tok in ["where", "and", "or", "=", ">", "<"] for tok in tokens)
        if not (has_col or has_agg or has_condition):
            return {"error": "Please elaborate further on what you need precisely."}

        # --- Guardrail: check all columns in SQL exist in dataset ---
        import sqlparse
        parsed = sqlparse.parse(raw_sql or "")
        referenced_cols = set()
        for stmt in parsed:
            for token in stmt.flatten():
                if token.ttype is None and token.value and token.value.strip().isidentifier():
                    referenced_cols.add(token.value.strip())
        dataset_cols = set([c['name'] for c in profile.get('columns',[])])
        # Only error if any referenced col is not in dataset
        if referenced_cols and not referenced_cols.issubset(dataset_cols):
            file_name = profile.get('dataset',{}).get('filename','your file')
            col_names = [c['name'] for c in profile.get('columns',[])]
            return {"error": f"Sorry, I couldn't find all columns referenced in your query for '{file_name}'. Please try again. (Preview columns: {', '.join(col_names)})"}

        cleaned_queries = {}
        import re as _re
        for key, sql_text in (("raw_sql", raw_sql), ("robust_sql", robust_sql)):
            # Fix MODE() with missing argument: replace MODE() with MODE(col) for COALESCE usage
            if sql_text and "COALESCE" in sql_text and "MODE()" in sql_text:
                # Find all COALESCE(x, MODE()) and replace with COALESCE(x, MODE(x) OVER ())
                def _mode_fix(m):
                    col = m.group(1)
                    return f"COALESCE({col}, MODE({col}) OVER ())"
                sql_text = _re.sub(r"COALESCE\((\w+),\s*MODE\(\)\)", _mode_fix, sql_text)
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

            # If no results, show a friendly message
            if display_df.empty:
                return {
                    "answer": "Sorry! No results found for your query. Please try again!",
                    "sql": sql_to_execute,
                    "raw_sql": raw_sql,
                    "robust_sql": robust_sql or raw_sql,
                    "columns": result_columns,
                    "rows": [],
                    "truncated": False,
                    "notes": notes,
                    "data_quality_note": data_note,
                    "follow_up_questions": follow_ups,
                }

            answer_sentence = insights
            if not answer_sentence:
                answer_sentence = self.sql_generator.format_response(
                    question, result_df.to_dict(orient="records"), profile
                )
            answer_sentence = (answer_sentence or "").strip()
            if answer_sentence:
                answer_sentence = " ".join(answer_sentence.split())

            # --- Custom imputation reporting ---
            # Assume: policies/report are available from imputation pipeline
            # For demo, stub with empty dicts if not present
            policies = locals().get('policies', {})
            impute_report = locals().get('impute_report', {})
            col_nulls = {}
            for col in result_columns:
                # Find null % and count from profile
                col_prof = next((c for c in profile.get('columns', []) if c['name']==col), None)
                null_pct = col_prof['null_ratio']*100 if col_prof and 'null_ratio' in col_prof else None
                null_count = col_prof['null_count'] if col_prof and 'null_count' in col_prof else None
                if col in impute_report:
                    m = impute_report[col].get('method','')
                    n = impute_report[col].get('imputed',0)
                    # Suppress data note if null_pct is None or exactly 0
                    if null_pct is None or null_pct == 0:
                        continue
                    null_str = f"~{null_pct:.1f}% null" if null_pct is not None else ""
                    count_str = f"or {null_count} occurrences" if null_count is not None else ""
                    if n > 0:
                        col_nulls[col] = f"{null_str} {count_str} were imputed with {m}".strip()
            # Data Quality Note formatting
            if display_df.empty or (result_records and all(v is None for row in result_records for v in row.values())):
                note_line = ""
            elif len(col_nulls) == 1:
                note_line = next(iter(col_nulls.values()))
            elif len(col_nulls) > 1:
                note_line = "Data Quality Notes:\n" + "\n".join([f"{i+1}. {v}" for i,v in enumerate(col_nulls.values())])
            else:
                note_line = (data_note or "No additional data-quality adjustments were applied.").strip()
            # Remove any LLM-assumed text about value existence if result is empty
            if display_df.empty and answer_sentence and "Assumed" in answer_sentence:
                answer_sentence = ""
            # If answer_sentence indicates no data, suppress Data note and assumptions entirely
            suppress_data_note = False
            suppress_assumptions = False
            if answer_sentence and (
                answer_sentence.strip().lower().startswith("there is no available data") or
                answer_sentence.strip().lower().startswith("the dataset does not contain any") or
                answer_sentence.strip().lower().startswith("no data available")
            ):
                suppress_data_note = True
                suppress_assumptions = True
            final_answer = f"Question: {question}"
            final_answer += f"\nAnswer: {answer_sentence}" if answer_sentence else "\nAnswer: See robust query results below."
            if note_line and not suppress_data_note:
                final_answer += f"\nData note: {note_line}"
            if notes and not suppress_assumptions and str(notes).strip():
                final_answer += f"\n{notes.strip()}"

            return {
                "answer": final_answer,
                "sql": sql_to_execute,
                "raw_sql": raw_sql,
                "robust_sql": robust_sql or raw_sql,
                "columns": result_columns,
                "rows": result_records,
                "truncated": len(result_df.index) > MAX_ROWS,
                "notes": notes,
                "nothing_to_show": suppress_data_note,
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
