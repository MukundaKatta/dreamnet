"""CLI for dreamnet."""
import sys, json, argparse
from .core import Dreamnet

def main():
    parser = argparse.ArgumentParser(description="DreamNet — AI Hallucination as Dreaming. Research platform studying LLM hallucinations as analog to dreaming.")
    parser.add_argument("command", nargs="?", default="status", choices=["status", "run", "info"])
    parser.add_argument("--input", "-i", default="")
    args = parser.parse_args()
    instance = Dreamnet()
    if args.command == "status":
        print(json.dumps(instance.get_stats(), indent=2))
    elif args.command == "run":
        print(json.dumps(instance.search(input=args.input or "test"), indent=2, default=str))
    elif args.command == "info":
        print(f"dreamnet v0.1.0 — DreamNet — AI Hallucination as Dreaming. Research platform studying LLM hallucinations as analog to dreaming.")

if __name__ == "__main__":
    main()
