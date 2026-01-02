import json
import re
from pathlib import Path

KB_PATH = Path("data/kb/articles.md")
OUT_PATH = Path("data/kb/chunks.json")

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def parse_chunks(md: str):
    lines = md.splitlines()
    category = None
    chunks = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        m_cat = re.match(r"^##\s+Category:\s*(.+)$", line)
        if m_cat:
            category = m_cat.group(1).strip()
            i += 1
            continue

        m_q = re.match(r"^###\s+Q:\s*(.+)$", line)
        if m_q:
            question = m_q.group(1).strip()
            block_lines = [lines[i]]
            i += 1
            while i < len(lines):
                nxt = lines[i].strip()
                if re.match(r"^(###\s+Q:|##\s+Category:)", nxt):
                    break
                block_lines.append(lines[i])
                i += 1

            content = normalize_ws("\n".join(block_lines))
            chunks.append({
                "category": category or "Unknown",
                "question": question,
                "content": content
            })
            continue

        i += 1

    return chunks

def main():
    md = KB_PATH.read_text(encoding="utf-8")
    chunks = parse_chunks(md)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(chunks)} chunks -> {OUT_PATH}")

if __name__ == "__main__":
    main()
