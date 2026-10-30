import argparse
import json
import os

from .agent import MockPolicy, ask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--history", default="[]", help="JSON array of {role,content}")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--mock", action="store_true", help="offline workflow fixture, not LLM evaluation")
    args = parser.parse_args()
    print(json.dumps(ask(args.question, json.loads(args.history), max_steps=args.max_steps, policy=MockPolicy() if args.mock or os.getenv("MOCK_MODEL") == "1" else None), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
