import argparse
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from utils import read_jsonl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="data/corpus.jsonl")
    ap.add_argument("--index", required=True, help="data/faiss.index")
    ap.add_argument("--meta", default="data/meta.pkl", help="id->meta map")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    rows = read_jsonl(args.corpus)
    texts = [r["text"] for r in rows]
    ids = [r["id"] for r in rows]
    metas = [{"source": r["source"], "page": r["page"]} for r in rows]

    print("[info] embedding…")
    model = SentenceTransformer(args.model)
    emb = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

    print("[info] building FAISS…")
    d = emb.shape[1]
    index = faiss.IndexFlatIP(d)  # cosine via normalized vectors
    index.add(emb.astype(np.float32))
    Path(args.index).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, args.index)

    with open(args.meta, "wb") as f:
        pickle.dump({"ids": ids, "metas": metas}, f)

    print(f"[ok] index saved to {args.index}; meta to {args.meta}; n={len(ids)}")

if __name__ == "__main__":
    main()
