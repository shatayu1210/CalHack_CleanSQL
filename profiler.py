#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import duckdb

try:
    import orjson as fastjson
    def dumps(obj):
        return fastjson.dumps(obj, option=fastjson.OPT_INDENT_2).decode()
except Exception:
    def dumps(obj):
        return json.dumps(obj, indent=2, ensure_ascii=False)

W_CLIENT = None
EMBED = None


# Set up connectivity to a local or remote Weaviate instance and, if requested,
# load a lightweight embedding model. This makes the profiler capable of pushing
# profiles into a RAG store with vectors, without forcing you to install heavy
# ML dependencies unless you actually want embeddings.
def init_weaviate(url: Optional[str], api_key: Optional[str], embed_model: Optional[str]):
    global W_CLIENT, EMBED
    if not url:
        return
    try:
        import weaviate
        if api_key:
            W_CLIENT = weaviate.Client(url=url, auth_client_secret=weaviate.AuthApiKey(api_key))
        else:
            W_CLIENT = weaviate.Client(url=url)
    except Exception as e:
        print(f"[warn] Weaviate init failed: {e}", file=sys.stderr)
        W_CLIENT = None
    if embed_model:
        try:
            from sentence_transformers import SentenceTransformer
            EMBED = SentenceTransformer(embed_model)
        except Exception as e:
            print(f"[warn] embedding model load failed: {e}", file=sys.stderr)
            EMBED = None


# Create the collection (aka class) in Weaviate on the fly if it doesn't exist.
# We keep the schema minimal and pragmatic so you can start searching right away
# and iterate later without ceremony.
def ensure_class(class_name: str):
    if not W_CLIENT or not class_name:
        return
    try:
        schema = W_CLIENT.schema.get()
        names = {c.get("class") for c in schema.get("classes", [])}
        if class_name in names:
            return
    except Exception:
        pass
    try:
        class_obj = {
            "class": class_name,
            "description": "CleanSQL column profiles for hybrid search",
            "vectorizer": "text2vec-transformers",
            "properties": [
                {"name": "dataset", "dataType": ["text"]},
                {"name": "column", "dataType": ["text"]},
                {"name": "duckdb_type", "dataType": ["text"]},
                {"name": "semantic_type", "dataType": ["text"]},
                {"name": "null_ratio", "dataType": ["number"]},
                {"name": "distinct_count", "dataType": ["number"]},
                {"name": "search_text", "dataType": ["text"]},
                {"name": "profile_json", "dataType": ["text"]}
            ]
        }
        W_CLIENT.schema.create_class(class_obj)
    except Exception as e:
        print(f"[warn] Weaviate class create failed: {e}", file=sys.stderr)


# Turn a short piece of text into a vector using the local model if available.
# If no model is loaded, we simply skip vectors so the flow remains optional
# and never blocks your basic profiling.
def embed_text(text: str) -> Optional[List[float]]:
    if EMBED is None:
        return None
    try:
        v = EMBED.encode([text], normalize_embeddings=True)
        return v[0].tolist()
    except Exception as e:
        print(f"[warn] embedding failed: {e}", file=sys.stderr)
        return None


# Quick helper to fetch the file size. Not all filesystems behave the same,
# so we guard with a try/except and return -1 if anything looks off.
def size_bytes(path: str) -> int:
    try:
        return os.path.getsize(path)
    except Exception:
        return -1


# Read a CSV with DuckDB and compute a compact but useful profile. The goal is
# speed and clarity: types, nulls, distincts, numeric stats with outlier bounds,
# categorical top values, and basic datetime ranges. No data is modified—this
# is purely descriptive so you can build robust plans later.
def profile_csv(csv_path: str,
                sample_rows: Optional[int],
                topk: int,
                approx_distinct: bool,
                date_try_cast_for_varchar: bool) -> Dict[str, Any]:
    con = duckdb.connect()
    con.execute("PRAGMA threads=" + str(os.cpu_count() or 8))
    auto_opts = []
    if sample_rows:
        auto_opts.append(f"SAMPLE_SIZE={sample_rows}")
    src_sql = f"SELECT * FROM read_csv_auto('{csv_path}', {', '.join(auto_opts)})" if auto_opts else f"SELECT * FROM read_csv_auto('{csv_path}')"
    con.execute(f"CREATE VIEW v AS {src_sql}")
    schema_rows = con.execute("PRAGMA table_info('v')").fetchall()
    columns = [{"name": r[1], "duckdb_type": r[2]} for r in schema_rows]
    row_count = con.execute("SELECT count(*) FROM v").fetchone()[0]
    col_count = len(columns)
    dup_rows = 0
    try:
        hash_expr = "hash(" + ", ".join([f'"{c["name"]}"' for c in columns]) + ")"
        dup_rows = con.execute(f"""
            WITH h AS (
              SELECT {hash_expr} AS h
              FROM v
            )
            SELECT COALESCE(SUM(cnt - 1), 0)
            FROM (SELECT h, COUNT(*) AS cnt FROM h GROUP BY h) t
            WHERE cnt > 1
        """).fetchone()[0]
    except Exception:
        dup_rows = None
    profiles: List[Dict[str, Any]] = []
    for col in columns:
        name = col["name"]
        dtype = col["duckdb_type"].upper()
        nulls, non_nulls = con.execute(f"""
            SELECT SUM(CASE WHEN "{name}" IS NULL THEN 1 ELSE 0 END),
                   SUM(CASE WHEN "{name}" IS NOT NULL THEN 1 ELSE 0 END)
            FROM v
        """).fetchone()
        null_ratio = (nulls or 0) / row_count if row_count else 0.0
        if approx_distinct:
            distinct_count = con.execute(f'SELECT approx_count_distinct("{name}") FROM v').fetchone()[0]
        else:
            try:
                distinct_count = con.execute(f'SELECT count(DISTINCT "{name}") FROM v').fetchone()[0]
            except Exception:
                distinct_count = con.execute(f'SELECT approx_count_distinct("{name}") FROM v').fetchone()[0]
        prof: Dict[str, Any] = {
            "name": name,
            "duckdb_type": dtype,
            "null_count": int(nulls or 0),
            "null_ratio": float(null_ratio),
            "distinct_count": int(distinct_count or 0),
        }
        lower = name.lower()
        semantic = None
        if any(k in lower for k in ["date", "time", "timestamp"]):
            semantic = "datetime"
        elif any(k in lower for k in ["id", "uuid", "case_id", "ticket"]):
            semantic = "identifier"
        elif dtype in ("BIGINT","INTEGER","SMALLINT","HUGEINT","UBIGINT","UINTEGER","USMALLINT","TINYINT","UTINYINT","DECIMAL","DOUBLE","REAL"):
            semantic = "numeric"
        elif dtype in ("DATE","TIMESTAMP","TIMESTAMP_TZ","TIME"):
            semantic = "datetime"
        elif dtype in ("BOOLEAN",):
            semantic = "boolean"
        else:
            if row_count and distinct_count <= min(50, max(10, row_count * 0.02)):
                semantic = "category"
            else:
                semantic = "text"
        prof["semantic_type"] = semantic
        if semantic == "numeric":
            stats = con.execute(f"""
                SELECT
                  MIN("{name}"),
                  MAX("{name}"),
                  AVG("{name}"),
                  STDDEV_SAMP("{name}"),
                  QUANTILE_CONT("{name}", 0.25),
                  QUANTILE_CONT("{name}", 0.5),
                  QUANTILE_CONT("{name}", 0.75),
                  SUM(CASE WHEN "{name}" = 0 THEN 1 ELSE 0 END),
                  SUM(CASE WHEN "{name}" < 0 THEN 1 ELSE 0 END)
                FROM v WHERE "{name}" IS NOT NULL
            """).fetchone()
            (vmin, vmax, mean, std, q1, med, q3, zeros, negatives) = stats
            iqr = (q3 - q1) if (q3 is not None and q1 is not None) else None
            low = (q1 - 1.5 * iqr) if iqr is not None else None
            high = (q3 + 1.5 * iqr) if iqr is not None else None
            outliers = None
            if low is not None and high is not None:
                outliers = con.execute(f"""
                  SELECT SUM(CASE WHEN "{name}" < ? OR "{name}" > ? THEN 1 ELSE 0 END)
                  FROM v WHERE "{name}" IS NOT NULL
                """, [low, high]).fetchone()[0]
            prof["numeric"] = {
                "min": vmin, "max": vmax, "mean": mean, "stddev": std,
                "q1": q1, "median": med, "q3": q3, "iqr": iqr,
                "outlier_low": low, "outlier_high": high, "outlier_count": int(outliers or 0),
                "zeros": int(zeros or 0), "negatives": int(negatives or 0),
            }
        if semantic == "datetime":
            try_expr = f'"{name}"'
            if dtype not in ("DATE","TIMESTAMP","TIMESTAMP_TZ","TIME") and date_try_cast_for_varchar:
                try_expr = f"try_cast(\"{name}\" as TIMESTAMP)"
            dmin, dmax, invalid = con.execute(f"""
                SELECT MIN({try_expr}), MAX({try_expr}),
                       SUM(CASE WHEN {try_expr} IS NULL AND "{name}" IS NOT NULL THEN 1 ELSE 0 END)
                FROM v
            """).fetchone()
            prof["datetime"] = {
                "min": str(dmin) if dmin is not None else None,
                "max": str(dmax) if dmax is not None else None,
                "invalid_parse": int(invalid or 0)
            }
        if semantic in ("text","category","identifier"):
            top_vals = con.execute(f"""
                SELECT "{name}" as val, COUNT(*) as c
                FROM v
                WHERE "{name}" IS NOT NULL
                GROUP BY 1
                ORDER BY c DESC
                LIMIT {topk}
            """).fetchall()
            total_non_null = non_nulls or 1
            topk_list = [{"value": r[0], "count": int(r[1]), "ratio": float((r[1] or 0)/total_non_null)} for r in top_vals]
            avg_len = None
            try:
                avg_len = con.execute(f"""
                    SELECT AVG(length("{name}")) FROM v WHERE "{name}" IS NOT NULL
                """).fetchone()[0]
            except Exception:
                pass
            prof["categorical"] = {"topk": topk_list, "avg_length": float(avg_len) if avg_len is not None else None}
        examples = con.execute(f'SELECT "{name}" FROM v WHERE "{name}" IS NOT NULL LIMIT 3').fetchall()
        prof["examples"] = [r[0] for r in examples]
        profiles.append(prof)
    dataset = {
        "path": os.path.abspath(csv_path),
        "filename": os.path.basename(csv_path),
        "size_bytes": size_bytes(csv_path),
        "row_count": int(row_count),
        "column_count": int(col_count),
        "duplicate_row_count": int(dup_rows) if dup_rows is not None else None,
        "created_at": int(time.time()),
    }
    return {
        "dataset": dataset,
        "columns": profiles,
        "duckdb_version": duckdb.__version__,
        "engine": "duckdb",
    }


# Persist the profiling results in a friendly layout: one dataset-level JSON
# for the overview, and one JSON per column. This makes it trivial to feed
# the artifacts into a RAG index or to diff profiles over time.
def save_artifacts(profile: Dict[str, Any], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "dataset_profile.json"), "w") as f:
        f.write(dumps(profile))
    cols_dir = os.path.join(out_dir, "columns")
    os.makedirs(cols_dir, exist_ok=True)
    dataset_name = profile["dataset"]["filename"]
    for col in profile["columns"]:
        obj = {"dataset": dataset_name, "column": col["name"], "profile": col}
        with open(os.path.join(cols_dir, f'{col["name"]}.json'), "w") as f:
            f.write(dumps(obj))


# Squeeze the most important facts about a column into a single string. This
# helps both keyword (BM25) and vector search latch onto the right columns
# during retrieval without overengineering the schema.
def build_search_text(dataset_name: str, col: Dict[str, Any]) -> str:
    parts = [
        f"dataset: {dataset_name}",
        f"column: {col['name']}",
        f"type: {col.get('duckdb_type')}",
        f"semantic: {col.get('semantic_type')}",
        f"null_ratio: {round(col.get('null_ratio',0.0),4)}",
        f"distinct: {col.get('distinct_count')}",
    ]
    if "numeric" in col:
        n = col["numeric"]
        parts += [f"min:{n.get('min')} max:{n.get('max')} mean:{n.get('mean')}",
                  f"q1:{n.get('q1')} median:{n.get('median')} q3:{n.get('q3')}",
                  f"outliers:{n.get('outlier_count')}"]
    if "categorical" in col:
        topk = col["categorical"].get("topk") or []
        parts.append("top_values: " + ", ".join([str(t['value']) for t in topk]))
    if "datetime" in col:
        d = col["datetime"]
        parts += [f"date_min:{d.get('min')} date_max:{d.get('max')} invalid:{d.get('invalid_parse')}"]
    return " | ".join([p for p in parts if p])


# Push each column profile into Weaviate as a small, searchable object. If an
# embedding model is present, we add vectors; if not, BM25 still works fine.
# Batching keeps things smooth on a laptop.
def ingest_weaviate(profile: Dict[str, Any], class_name: str):
    if not W_CLIENT or not class_name:
        return
    ensure_class(class_name)
    dataset_name = profile["dataset"]["filename"]
    try:
        W_CLIENT.batch.configure(batch_size=64, dynamic=True)
    except Exception:
        pass
    for col in profile["columns"]:
        search_text = build_search_text(dataset_name, col)
        vec = embed_text(search_text)
        props = {
            "dataset": dataset_name,
            "column": col["name"],
            "duckdb_type": col.get("duckdb_type"),
            "semantic_type": col.get("semantic_type"),
            "null_ratio": float(col.get("null_ratio",0.0)),
            "distinct_count": int(col.get("distinct_count",0)),
            "search_text": search_text,
            "profile_json": json.dumps(col, ensure_ascii=False),
        }
        uid = hashlib.md5(f"{dataset_name}:{col['name']}".encode()).hexdigest()
        try:
            W_CLIENT.batch.add_data_object(props, class_name=class_name, uuid=uid, vector=vec)
        except Exception as e:
            print(f"[warn] batch add failed: {e}", file=sys.stderr)
    try:
        W_CLIENT.batch.flush()
    except Exception:
        pass


# Command-line entrypoint. You point it at a CSV and an output folder, and it
# does the rest: profile with DuckDB, write JSON artifacts, and optionally
# ingest into Weaviate so you can query profiles right away.
def main():
    ap = argparse.ArgumentParser(description="CleanSQL CSV Profiler (DuckDB → JSON; optional Weaviate)")
    ap.add_argument("--csv", required=True, help="Path to CSV")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--topk", type=int, default=10, help="Top-K values for categorical/text")
    ap.add_argument("--sample-rows", type=int, default=None, help="DuckDB read_csv_auto SAMPLE_SIZE")
    ap.add_argument("--no-approx-distinct", action="store_true", help="Disable approx_count_distinct")
    ap.add_argument("--date-try-cast", action="store_true", help="Try casting varchar to TIMESTAMP for date profiling")
    ap.add_argument("--weaviate-url", type=str, default=None, help="Weaviate URL, e.g., http://localhost:8080")
    ap.add_argument("--weaviate-api-key", type=str, default=None, help="Weaviate API key")
    ap.add_argument("--weaviate-class", type=str, default="CleanSQLColumn", help="Weaviate class name")
    ap.add_argument("--embed-model", type=str, default=None, help="sentence-transformers model for vectors")
    args = ap.parse_args()

    init_weaviate(args.weaviate_url, args.weaviate_api_key, args.embed_model)

    prof = profile_csv(
        args.csv,
        sample_rows=args.sample_rows,
        topk=args.topk,
        approx_distinct=not args.no_approx_distinct,
        date_try_cast_for_varchar=args.date_try_cast,
    )
    save_artifacts(prof, args.out)

    if args.weaviate_url:
        ingest_weaviate(prof, args.weaviate_class)

    print(f"[ok] Profiled: {args.csv}")
    print(f"[ok] Output: {args.out}")
    if args.weaviate_url:
        print(f"[ok] Ingested to Weaviate class: {args.weaviate_class}")
    
    # NEW: Initialize LLM-powered data assistant
    try:
        from llm_integration import DataAssistant
        print("\n🤖 Initializing AI Data Assistant...")
        assistant = DataAssistant()
        assistant.setup_database(prof, args.csv)
        
        # Interactive Q&A loop
        print("\n💬 Ask questions about your data! (type 'quit' to exit)")
        print("💡 Example: 'What's the average age?' or 'Show me the top 5 cities'")
        
        while True:
            question = input("\n❓ Your question: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if question:
                response = assistant.ask_question(question, prof)
                print(f"\n🤖 Assistant: {response}")
        
        assistant.close()
        print("\n👋 Goodbye!")
        
    except ImportError:
        print("\n⚠️  LLM integration not available. Install anthropic and python-dotenv to enable AI features.")
    except Exception as e:
        print(f"\n⚠️  LLM integration failed: {e}")
        print("Continuing without AI features...")


if __name__ == "__main__":
    main()
