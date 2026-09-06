from __future__ import annotations
from core.graph import get_agent

def main():
    agent = get_agent(); print("Research Paper Answer Bot. Type 'exit' to quit.")
    while True:
        query = input("\nYou: ").strip()
        if query.lower() in {"exit", "quit"}: break
        if query:
            result = agent.ask(query, "cli")
            print("\nAssistant:", result.answer)
            for source in result.sources: print(f"[{source.number}] {source.title}, p. {source.page}")

if __name__ == "__main__": main()
