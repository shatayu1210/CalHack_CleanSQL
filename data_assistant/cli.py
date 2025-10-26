#!/usr/bin/env python3
"""Command-line interface for the AI Data Assistant."""

import argparse
import json
import os
import tempfile
from contextlib import suppress
from typing import Dict, List, Optional, Tuple

import pandas as pd

from llm_integration import DataAssistant
from profiler import profile_csv

TRUE_VALUES = {"true", "t", "yes", "y", "1"}
FALSE_VALUES = {"false", "f", "no", "n", "0"}
BOOLEAN_COLUMNS = {
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "romantic",
}
NUMERIC_COLUMNS = [
    "age",
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health",
    "absences",
    "G1",
    "G2",
    "G3",
]
INTEGER_COLUMNS = set(NUMERIC_COLUMNS)
DEFAULT_CATEGORY_FILL = "Unknown"
SUMMARY_PREVIEW_LIMIT = 4


def _format_numeric_value(value: object) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.2f}"
    return str(value)


def _coerce_boolean_series(series: pd.Series) -> Tuple[pd.Series, Optional[bool], int]:
    """Normalise boolean-like columns and return (series, fill_value, missing_count)."""
    normalized = []
    for value in series:
        if pd.isna(value):
            normalized.append(pd.NA)
            continue
        if isinstance(value, bool):
            normalized.append(value)
            continue
        if isinstance(value, (int, float)) and not pd.isna(value):
            if value == 1:
                normalized.append(True)
                continue
            if value == 0:
                normalized.append(False)
                continue
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                normalized.append(pd.NA)
                continue
            lowered = stripped.lower()
            if lowered in TRUE_VALUES:
                normalized.append(True)
                continue
            if lowered in FALSE_VALUES:
                normalized.append(False)
                continue
        normalized.append(pd.NA)

    normal_series = pd.Series(normalized, index=series.index, dtype="object")
    missing_count = int(normal_series.isna().sum())

    dropna = normal_series.dropna()
    if dropna.empty:
        fill_value = False
    else:
        fill_value = bool(dropna.mode().iloc[0])

    normal_series.loc[normal_series.isna()] = fill_value
    return normal_series.astype(bool), fill_value, missing_count


def _coerce_numeric_series(
    series: pd.Series, as_integer: bool
) -> Tuple[pd.Series, float, int]:
    """Coerce numeric column, fill with median (rounded for ints)."""
    cleaned = series.apply(lambda x: x.strip() if isinstance(x, str) else x)
    numeric = pd.to_numeric(cleaned, errors="coerce")
    missing_count = int(numeric.isna().sum())

    median_value = numeric.median(skipna=True)
    if pd.isna(median_value):
        median_value = 0.0

    if as_integer:
        fill_value = int(round(float(median_value)))
        reported_value = fill_value
    else:
        fill_value = float(median_value)
        reported_value = fill_value

    numeric = numeric.fillna(fill_value)

    if as_integer:
        numeric = numeric.round().astype(int)

    return numeric, reported_value, missing_count


def _coerce_categorical_series(series: pd.Series) -> Tuple[pd.Series, str, int]:
    """Standardise categorical column and fill with mode."""
    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.replace("", pd.NA)
    missing_count = int(cleaned.isna().sum())

    mode_values = cleaned.dropna().mode()
    if mode_values.empty:
        fill_value = DEFAULT_CATEGORY_FILL
    else:
        fill_value = str(mode_values.iloc[0])

    cleaned = cleaned.fillna(fill_value)
    return cleaned.astype(str), fill_value, missing_count


def _summarise_section(
    title: str,
    items: Dict[str, Dict[str, object]],
    value_formatter,
) -> Optional[str]:
    if not items:
        return None
    entries = sorted(items.items(), key=lambda kv: kv[0])
    formatted: List[str] = []
    for idx, (column, meta) in enumerate(entries):
        if idx >= SUMMARY_PREVIEW_LIMIT:
            remaining = len(entries) - SUMMARY_PREVIEW_LIMIT
            formatted.append(f"... (+{remaining} more)")
            break
        formatted.append(f"{column}={value_formatter(meta)}")
    return f"{title}: " + ", ".join(formatted)


def _format_report_lines(report: Dict[str, Dict[str, Dict[str, object]]]) -> List[str]:
    """Return human-readable summary lines for console output."""
    lines: List[str] = []

    numeric_line = _summarise_section(
        "Numeric medians",
        report.get("numeric", {}),
        lambda meta: f"{_format_numeric_value(meta['fill'])} (filled {meta['missing']} rows)",
    )
    if numeric_line:
        lines.append(numeric_line)

    boolean_line = _summarise_section(
        "Boolean mode",
        report.get("boolean", {}),
        lambda meta: f"{'True' if meta['fill'] else 'False'} (filled {meta['missing']} rows)",
    )
    if boolean_line:
        lines.append(boolean_line)

    categorical_line = _summarise_section(
        "Categorical mode",
        report.get("categorical", {}),
        lambda meta: f"{meta['fill']} (filled {meta['missing']} rows)",
    )
    if categorical_line:
        lines.append(categorical_line)

    return lines


def _prepare_dataset(csv_path: str) -> Tuple[str, Dict[str, Dict[str, Dict[str, object]]]]:
    """Load CSV, apply imputation rules, and persist to a temporary cleaned file."""
    df = pd.read_csv(csv_path)

    imputation_report: Dict[str, Dict[str, Dict[str, object]]] = {
        "numeric": {},
        "boolean": {},
        "categorical": {},
    }

    for column in BOOLEAN_COLUMNS:
        if column in df.columns:
            series, fill_value, missing = _coerce_boolean_series(df[column])
            df[column] = series
            if missing > 0:
                imputation_report["boolean"][column] = {
                    "fill": fill_value,
                    "missing": missing,
                }

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            series, fill_value, missing = _coerce_numeric_series(
                df[column], as_integer=column in INTEGER_COLUMNS
            )
            df[column] = series
            if missing > 0:
                imputation_report["numeric"][column] = {
                    "fill": fill_value,
                    "missing": missing,
                }

    for column in df.columns:
        if column in BOOLEAN_COLUMNS or column in NUMERIC_COLUMNS:
            continue
        series, fill_value, missing = _coerce_categorical_series(df[column])
        df[column] = series
        if missing > 0:
            imputation_report["categorical"][column] = {
                "fill": fill_value,
                "missing": missing,
            }

    temp_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline=""
    )
    try:
        df.to_csv(temp_file.name, index=False)
    finally:
        temp_file.close()

    return temp_file.name, imputation_report


def _assemble_profile_note(lines: List[str]) -> Optional[str]:
    if not lines:
        return None
    return "CLI imputations applied — " + "; ".join(lines)


def run(
    csv_path: str,
    profile_path: Optional[str] = None,
    interactive: bool = True,
    question: Optional[str] = None,
) -> dict:
    """Run the assistant setup and optional question."""
    if not csv_path:
        raise ValueError("CSV path is required.")

    print("🧼 Cleaning dataset and imputing missing values...")
    try:
        cleaned_csv_path, imputation_report = _prepare_dataset(csv_path)
    except Exception as exc:
        return {"error": str(exc)}
    summary_lines = _format_report_lines(imputation_report)
    for line in summary_lines:
        print(f"   • {line}")

    result: dict = {
        "imputation_report": imputation_report,
        "imputation_summary": summary_lines,
    }

    assistant: Optional[DataAssistant] = None
    try:
        if profile_path and os.path.exists(profile_path):
            with open(profile_path, "r", encoding="utf-8") as file:
                profile = json.load(file)
        else:
            print("📊 Creating data profile...")
            profile = profile_csv(cleaned_csv_path, None, 10, True, False)

        dataset_meta = profile.setdefault("dataset", {})
        dataset_meta["filename"] = os.path.basename(csv_path)
        dataset_meta["original_path"] = os.path.abspath(csv_path)
        dataset_meta["imputed_source"] = "cli_median_mode"

        profile_note = _assemble_profile_note(summary_lines)
        if profile_note:
            notes = profile.setdefault("notes", [])
            notes.append(profile_note)
            result["imputation_note"] = profile_note

        result["profile"] = profile

        print("🤖 Initializing AI Data Assistant...")
        assistant = DataAssistant()
        assistant.setup_database(profile, cleaned_csv_path)

        if interactive:
            print("\n💬 Ask questions about your data! (type 'quit' to exit)")
            print("💡 Example: 'What's the average age?' or 'Show me the top 5 cities'")

            while True:
                user_question = input("\n❓ Your question: ").strip()
                if user_question.lower() in ["quit", "exit", "q"]:
                    break

                if not user_question:
                    continue

                try:
                    response = assistant.ask_question(user_question, profile)
                except Exception as exc:
                    print(f"\n❌ {exc}")
                    continue
                if isinstance(response, dict):
                    if response.get("error"):
                        print(f"\n❌ {response['error']}")
                    else:
                        print(f"\n🤖 Assistant: {response.get('answer')}")
                        if response.get("sql"):
                            print(f"🧮 SQL: {response['sql']}")
                        if response.get("notes"):
                            print(f"📝 Notes: {response['notes']}")
                        follow_ups = response.get("follow_up_questions") or []
                        if follow_ups:
                            print("🔎 Follow-up questions:")
                            for item in follow_ups:
                                print(f"  - {item}")
                        preview_rows = response.get("rows", [])[:5]
                        if preview_rows:
                            print("📋 Preview:")
                            for row in preview_rows:
                                print(row)
                else:
                    print(f"\n🤖 Assistant: {response}")

            print("\n👋 Goodbye!")
        else:
            if question:
                try:
                    payload = assistant.ask_question(question, profile)
                except Exception as exc:
                    result["error"] = str(exc)
                    payload = None
                result["response"] = payload
                if isinstance(payload, dict) and payload.get("error"):
                    result["error"] = payload["error"]
                elif isinstance(payload, dict):
                    if payload.get("notes"):
                        result["notes"] = payload["notes"]
                    follow_ups = payload.get("follow_up_questions")
                    if follow_ups:
                        result["follow_up_questions"] = follow_ups
    except Exception as exc:
        result["error"] = str(exc)
    finally:
        if assistant is not None:
            assistant.close()
        if os.path.abspath(cleaned_csv_path) != os.path.abspath(csv_path):
            with suppress(FileNotFoundError):
                os.remove(cleaned_csv_path)

    return result


def main(
    csv_path: Optional[str] = None,
    profile_path: Optional[str] = None,
    interactive: bool = True,
    question: Optional[str] = None,
) -> dict:
    """Entry point for CLI or programmatic use."""
    if csv_path is None and interactive:
        parser = argparse.ArgumentParser(description="AI Data Assistant")
        parser.add_argument("--csv", required=True, help="Path to CSV file")
        parser.add_argument("--profile", help="Path to existing profile JSON")
        parser.add_argument("--question", help="Single question to answer and exit")

        args = parser.parse_args()
        csv_path = args.csv
        profile_path = args.profile
        question = args.question

    if csv_path is None:
        raise ValueError("csv_path must be provided when interactive=False.")

    return run(
        csv_path,
        profile_path,
        interactive=interactive and question is None,
        question=question,
    )


if __name__ == "__main__":
    main()
