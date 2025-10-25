#!/usr/bin/env python3
import argparse
import json
import os

from llm_integration import DataAssistant
from profiler import profile_csv


def main():
    parser = argparse.ArgumentParser(description="AI Data Assistant")
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--profile", help="Path to existing profile JSON")
    
    args = parser.parse_args()
    
    # Load or create profile
    if args.profile and os.path.exists(args.profile):
        with open(args.profile, 'r') as f:
            profile = json.load(f)
    else:
        print("📊 Creating data profile...")
        profile = profile_csv(args.csv, None, 10, True, False)
    
    # Initialize assistant
    print("🤖 Initializing AI Data Assistant...")
    assistant = DataAssistant()
    assistant.setup_database(profile, args.csv)
    
    # Interactive session
    print("\n💬 Ask questions about your data! (type 'quit' to exit)")
    print("💡 Example: 'What's the average age?' or 'Show me the top 5 cities'")
    
    while True:
        question = input("\n❓ Your question: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        if question:
            response = assistant.ask_question(question, profile)
            print(f"\n🤖 Assistant: {response}")
    
    assistant.close()
    print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()
