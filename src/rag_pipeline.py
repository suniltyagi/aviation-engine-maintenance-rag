import argparse
import json
import pickle
import re
from pathlib import Path

# --------- FAST mode & CPU thread limits ---------
import os
FAST = os.environ.get("RAG_FAST", "0") == "1"
try:
    import torch
    if os.environ.get("OMP_NUM_THREADS") is None:
        os.environ["OMP_NUM_THREADS"] = "4"
    # keep PyTorch thread count sane on CPU
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
except Exception:
    pass
# -------------------------------------------------

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from utils import read_jsonl

# Optional HF generation pipeline
try:
    from transformers import pipeline as hf_pipeline
except Exception:
    hf_pipeline = None


# -------------------- FAISS / RETRIEVAL --------------------

def load_index(index_path: str):
    """Read a FAISS index from disk."""
    return faiss.read_index(index_path)


def search(query: str, encoder: SentenceTransformer, index, k: int = 6):
    """Encode the query and retrieve top-k from a FAISS index."""
    qv = encoder.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    qv = qv.astype(np.float32)
    D, I = index.search(qv, k)
    return I[0], D[0]


# -------------------- GENERIC UTILITIES --------------------

def _sentences(txt: str):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', txt.strip()) if s.strip()]


def _dedupe_sentences(text: str) -> str:
    sents = _sentences(text)
    seen, cleaned = set(), []
    for s in sents:
        key = re.sub(r'\s+', ' ', s).strip().lower()
        if key and key not in seen:
            cleaned.append(s.strip())
            seen.add(key)
    return ' '.join(cleaned)


def _extract_subject(q: str) -> str:
    """
    Heuristic subject extractor from common question forms.
    Works for generic questions; falls back gracefully.
    """
    original = q.strip()
    pats = [
        r'^\s*why\s+(?:is|are)\s+(.+?)(?:\s+(?:located|mounted|used|installed|required|needed|called)|\?)',
        r'^\s*what\s+(?:is|are)\s+(.+?)\?',
        r'^\s*how\s+(?:does|do)\s+(.+?)\?',
        r'^\s*where\s+is\s+(.+?)\?',
        r'^\s*which\s+(.+?)\?'
    ]
    for p in pats:
        m = re.search(p, original, flags=re.I)
        if m:
            sub = m.group(1).strip(' "\'.,;:()[]{}')
            return ' '.join(sub.split()[:12])
    # Fallbacks
    title_spans = re.findall(r'(?:[A-Z][a-z0-9\-\'"]*\s+){1,6}[A-Z][a-z0-9\-\'"]*', original)
    if title_spans:
        sub = max(title_spans, key=len).strip(' "\'.,;:()[]{}')
        return ' '.join(sub.split()[:12])
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-']*", original)
    stops = {"why","how","what","where","which","is","are","the","a","an","of","in","to","for","and","or","with","on","at","by","from","this","that","it"}
    content = [t for t in tokens if t.lower() not in stops]
    return ' '.join(content[:6]) if content else "The system"

# -------------------- GENERATION HELPERS --------------------

def _stop_token_ids(tokenizer):
    ids = set()
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        if isinstance(eos, list): ids.update(eos)
        else: ids.add(eos)
    try:
        if "<|im_end|>" in tokenizer.get_vocab():
            ids.add(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    except Exception:
        pass
    return [i for i in ids if i is not None]

def _filter_gen_kwargs(model, want: dict) -> dict:
    """Only keep generation kwargs the current backend accepts."""
    gc = getattr(model, "generation_config", None)
    allowed = set(dir(gc)) if gc else {
        "max_new_tokens","min_new_tokens","num_beams","do_sample",
        "no_repeat_ngram_size","repetition_penalty","eos_token_id","pad_token_id",
        "temperature","top_p","top_k","return_full_text"
    }
    return {k: v for k, v in want.items() if k in allowed and v is not None}

def _strip_chat_markers(text: str) -> str:
    for m in ("<|im_start|>","<|im_end|>"):
        text = text.replace(m, "")
    return text.strip()

_SUBJECT_DEFAULT = "The system"


def _prefer_subject_phrase(text: str, subject: str = _SUBJECT_DEFAULT) -> str:
    """Ensure the first sentence starts with a concrete subject phrase (not 'It/This')."""
    sents = _sentences(text)
    if not sents:
        return text
    s0 = re.sub(r'^\s*(It|This|The valve)\b', subject, sents[0], flags=re.I)
    sents[0] = s0
    return ' '.join(sents)


def _is_t5(model_id: str) -> bool:
    return "t5" in model_id.lower()


def _pick_task(model_id: str) -> str:
    return "text2text-generation" if _is_t5(model_id) else "text-generation"


def _truncate_by_tokens(prompt: str, tok: AutoTokenizer, max_in_tokens: int) -> str:
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=max_in_tokens)
    return tok.decode(enc.input_ids[0], skip_special_tokens=True)


def _numbers_in(text: str):
    # integers, decimals, ranges, percents
    return set(re.findall(r"\d+(?:\.\d+)?(?:\s*[-–]\s*\d+(?:\.\d+)?)?\s*%?", text))


def _answer_numbers_are_grounded(answer: str, evidence: str) -> bool:
    nums = {n.strip() for n in _numbers_in(answer)}
    if not nums:
        return True
    ev = evidence.replace("\n", " ")
    return all(n in ev for n in nums)


# -------------------- SUMMARISER (for fallback & pre-shrink) --------------------

def auto_summarize(context_text: str, question: str, max_sentences: int = 4) -> str:
    """Extractive summary biased by question keywords (generic, domain-agnostic)."""
    sents = _sentences(context_text)
    q_tokens = re.findall(r"[A-Za-z0-9\-]+", question.lower())
    q_keywords = {t for t in q_tokens if len(t) > 3}

    boost_terms = {"percent", "%", "thrust", "energy", "power", "propeller", "icing", "temperature", "mixture", "control", "valve", "air"}
    scored = []
    for s in sents:
        stoks = re.findall(r"[A-Za-z0-9\-]+", s.lower())
        overlap = len(q_keywords.intersection(stoks))
        is_caption = bool(re.search(r"figure\s+\d", s.lower()))
        boost = sum(1 for t in boost_terms if t in s.lower())
        score = overlap + boost - (1 if is_caption else 0)
        scored.append((score, s))

    picked = [s for sc, s in sorted(scored, key=lambda x: x[0], reverse=True) if sc > 0][:max_sentences]
    if not picked:
        picked = [s for s in sents if not re.search(r"figure\s+\d", s.lower())][:max_sentences]

    cleaned = []
    for s in picked:
        s = re.sub(r"\s*\(Source:.*?\)\s*", " ", s)
        s = re.sub(r"\s{2,}", " ", s).strip()
        cleaned.append(s)
    return " ".join(cleaned)


# -------------------- EVIDENCE SELECTION --------------------

def _select_evidence_for(context_text: str, question: str, focus_terms: tuple, topn: int = 4) -> str:
    """
    Pick evidence sentences biased toward focus terms (if any).
    Generic scoring: question keyword overlap + boosts for focus terms + numeric/% boosts.
    """
    sents = _sentences(context_text)
    q_tokens = re.findall(r"[A-Za-z0-9\-]+", question.lower())
    q_kw = {t for t in q_tokens if len(t) > 3}
    scored = []
    for s in sents:
        lower = s.lower()
        stoks = re.findall(r"[A-Za-z0-9\-]+", lower)
        overlap = len(q_kw.intersection(stoks))
        boost = 0
        if any(ft in lower for ft in focus_terms):
            boost += 2
        if any(x in lower for x in ("%", " percent", " per cent", "thrust", "energy", "power", "propeller", "torque")):
            boost += 1
        is_caption = bool(re.search(r"figure\s+\d", lower))
        score = overlap + boost - (1 if is_caption else 0)
        scored.append((score, s))
    kept = [s for sc, s in sorted(scored, key=lambda x: x[0], reverse=True) if sc > 0][:topn]
    return "\n".join(kept) if kept else "\n".join(sents[:topn])


def _build_simple_prompt(question: str, focused_context: str, is_t5: bool) -> str:
    instr = (
        "Use only the evidence below to answer in 1–2 sentences. "
        "Do not restate the question. If the evidence doesn’t contain the answer, say 'Not found in context.'"
    )
    body = (
        f"{instr}\n\n"
        f"EVIDENCE:\n{focused_context}\n\n"
        f"QUESTION: {question}\n"
        f"ANSWER:"
    )
    if is_t5:
        return body
    return "You are a helpful technical assistant.\n" + body


def _parse_focus(csv: str) -> tuple:
    # "percent, propeller, shaft" -> ("percent","propeller","shaft")
    if not csv:
        return ()
    return tuple(t.strip() for t in csv.split(",") if t.strip())


# -------------------- MAIN --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True, help="Question to answer")
    ap.add_argument("--k", type=int, default=6, help="Top-k passages to retrieve")
    ap.add_argument("--corpus", default="data/corpus.jsonl", help="Chunked corpus JSONL")
    ap.add_argument("--index", default="data/faiss.index", help="FAISS index path")
    ap.add_argument("--meta", default="data/meta.pkl", help="Pickle with per-chunk metadata")
    ap.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="Sentence-Transformers embedding model id")
    ap.add_argument("--gen_model", default="", help="HF model id for generation (optional)")
    ap.add_argument("--out", default="runs/preds.jsonl", help="Output JSONL path")
    # NEW: generic evidence control
    ap.add_argument("--focus_a", default="", help="Comma-separated terms to bias evidence (optional)")
    ap.add_argument("--focus_b", default="", help="Comma-separated terms to bias evidence (optional)")
    ap.add_argument("--evn", type=int, default=None, help="Evidence lines per pool; default is 3 (FAST) or 4")
    args = ap.parse_args()

    # Load data
    rows = read_jsonl(args.corpus)
    with open(args.meta, "rb") as f:
        meta = pickle.load(f)

    # Encoder + index
    encoder = SentenceTransformer(args.embed_model)
    index = load_index(args.index)

    # Retrieve
    idxs, _ = search(args.q, encoder, index, k=args.k)

    # Build human-readable context block
    contexts = []
    for rank, i in enumerate(idxs):
        r = rows[i]
        m = meta["metas"][i]
        src = m.get("source", "unknown")
        page = m.get("page", "?")
        contexts.append(f"[{rank+1}] {r['text']}\n(Source: {src} p.{page})")
    context_text = "\n\n".join(contexts)

    # ---- Generate (optional) or fallback ----
    if args.gen_model and hf_pipeline is not None:
        model_id = args.gen_model
        is_t5 = _is_t5(model_id)
        task = _pick_task(model_id)

        # ---- Evidence selection (generic; optional focus from CLI) ----
        evn = args.evn if args.evn is not None else (3 if FAST else 4)
        focus_a = _parse_focus(args.focus_a)
        focus_b = _parse_focus(args.focus_b)

        if focus_a or focus_b:
            ev_a = _select_evidence_for(context_text, args.q, focus_terms=focus_a, topn=evn)
            ev_b = _select_evidence_for(context_text, args.q, focus_terms=focus_b, topn=evn)
            focused_context = (ev_a + "\n" + ev_b).strip()
        else:
            # Fully generic: single pool; use twice the lines to compensate for not splitting
            focused_context = _select_evidence_for(context_text, args.q, focus_terms=(), topn=evn * 2)

        # Tokenizer & safe truncation
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        max_in = 512 if is_t5 else min(getattr(tok, "model_max_length", 2048), 2048)
        raw_prompt = _build_simple_prompt(args.q, focused_context, is_t5)
        prompt = _truncate_by_tokens(raw_prompt, tok, max_in_tokens=max_in)

        # Optional: Qwen chat template
        if "qwen" in model_id.lower() and hasattr(tok, "apply_chat_template"):
            msgs = [
                {"role": "system", "content": "Answer concisely using only the provided evidence."},
                {"role": "user", "content": prompt},
            ]
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

        # HF pipeline (CPU) + pad-token safety
        gen = hf_pipeline(task, model=model_id, device=-1, trust_remote_code=True, tokenizer=tok)

        # Ensure pad token is set
        try:
            if gen.model.config.pad_token_id is None:
                if tok.pad_token_id is not None:
                    gen.model.config.pad_token_id = tok.pad_token_id
                elif getattr(tok, "eos_token_id", None) is not None:
                    gen.model.config.pad_token_id = tok.eos_token_id
        except Exception:
            pass

        # Configure EOS to stop at </s> or <|im_end|> (for Qwen chat)
        try:
            stop_ids = _stop_token_ids(tok)
            if stop_ids:
                # some backends accept list; if not, we’ll collapse below
                gen.model.generation_config.eos_token_id = stop_ids
        except Exception:
            pass

        # ----------- Progress logs -----------
        print(f"[gen] model: {model_id}  (FAST={FAST})")
        try:
            print(f"[gen] prompt chars: {len(prompt)}")
        except Exception:
            pass
        print("[gen] starting generation...", flush=True)
        # ------------------------------------

        # FAST-aware decode settings (filtered to avoid warnings)
        raw_gen_kwargs = dict(
            max_new_tokens=80 if FAST else 140,
            min_new_tokens=16 if FAST else 32,
            num_beams=1 if FAST else 4,
            do_sample=False,
            no_repeat_ngram_size=6,
            repetition_penalty=1.2,
            # critical: prevents prompt echo from pipeline
            return_full_text=False,
            # early_stopping is often ignored; we omit to avoid the warning
        )
        gen_kwargs = _filter_gen_kwargs(gen.model, raw_gen_kwargs)

        # If eos_token_id list is rejected, collapse to first element on the fly
        try:
            out = gen(prompt, **gen_kwargs)
        except TypeError as e:
            if "eos_token_id" in str(e) and isinstance(getattr(gen.model.generation_config, "eos_token_id", None), list):
                gen.model.generation_config.eos_token_id = gen.model.generation_config.eos_token_id[0]
                out = gen(prompt, **gen_kwargs)
            else:
                raise
        print("[gen] done.", flush=True)

        # With return_full_text=False, HF returns only the continuation
        generated = (out[0].get("generated_text") or "").strip()

        # Safety trims (in case anything leaked)
        if "ANSWER:" in generated:
            try:
                generated = generated.split("ANSWER:", 1)[-1].strip()
            except Exception:
                pass
        generated = _strip_chat_markers(generated)

        answer = _dedupe_sentences(generated)
        if not answer or "EVIDENCE" in answer or answer.endswith(":"):
            # fallback: short extractive summary of the focused context
            answer = auto_summarize(focused_context, args.q, max_sentences=2)

    else:
        # Retrieval-only fallback
        answer = auto_summarize(context_text, args.q)

    # Write JSONL
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        rec = {"question": args.q, "answer": answer}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[ok] wrote {args.out}")


if __name__ == "__main__":
    main()
