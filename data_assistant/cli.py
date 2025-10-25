#!/usr/bin/env python3
"""Command-line interface for the AI Data Assistant."""

import argparse
import json
import os
from typing import Optional

from llm_integration import DataAssistant
from profiler import profile_csv


def run(
    csv_path: str,
    profile_path: Optional[str] = None,
    interactive: bool = True,
    question: Optional[str] = None,
) -> dict:
    """Run the assistant setup and optional question."""
    if not csv_path:
        raise ValueError("CSV path is required.")

    if profile_path and os.path.exists(profile_path):
        with open(profile_path, "r", encoding="utf-8") as file:
            profile = json.load(file)
    else:
        print("📊 Creating data profile...")
        profile = profile_csv(csv_path, None, 10, True, False)

    print("🤖 Initializing AI Data Assistant...")
    assistant = DataAssistant()
    assistant.setup_database(profile, csv_path)

    result: dict = {"profile": profile}

    if interactive:
        print("\n💬 Ask questions about your data! (type 'quit' to exit)")
        print("💡 Example: 'What's the average age?' or 'Show me the top 5 cities'")

        try:
            while True:
                question = input("\n❓ Your question: ").strip()
                if question.lower() in ["quit", "exit", "q"]:
                    break

                if question:
                    response = assistant.ask_question(question, profile)
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
        finally:
            assistant.close()
            print("\n👋 Goodbye!")
    else:
        try:
            if question:
                payload = assistant.ask_question(question, profile)
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
            assistant.close()

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
