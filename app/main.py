import json
from pathlib import Path

from app.agents.classifier import classify_ticket

TICKETS_PATH = Path("data/tickets/sample_tickets.json")


def main():
    tickets = json.loads(TICKETS_PATH.read_text(encoding="utf-8"))
    correct = 0

    for t in tickets:
        result = classify_ticket(t["text"])
        expected = t.get("expected_category")
        ok = (expected == result.category)

        if ok:
            correct += 1

        print(f'{t["id"]} | expected={expected:<7} predicted={result.category:<7} '
              f'conf={result.confidence:.2f} ok={ok}')

    acc = correct / max(1, len(tickets))
    print(f"\nAccuracy: {correct}/{len(tickets)} = {acc:.2%}")


if __name__ == "__main__":
    main()
