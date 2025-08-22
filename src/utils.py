import json
from pathlib import Path
from typing import Iterable, Dict, Any, List

def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]

def simple_word_chunks(text: str, max_words=600, overlap=120) -> list[str]:
    words = text.split()
    chunks = []
    step = max_words - overlap
    for i in range(0, max(1, len(words)), max(1, step)):
        chunk = " ".join(words[i:i+max_words])
        if chunk:
            chunks.append(chunk)
        if i + max_words >= len(words):
            break
    return chunks
