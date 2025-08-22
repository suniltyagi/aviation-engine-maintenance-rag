import argparse
import json
from pathlib import Path

from rouge_score import rouge_scorer

def load_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]

def exact_match(pred, answers):
    pred_norm = pred.strip().lower()
    return max(1.0 if pred_norm == a.strip().lower() else 0.0 for a in answers)

def rouge_l(pred, answers):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return max(scorer.score(a, pred)["rougeL"].fmeasure for a in answers)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="runs/preds.jsonl")
    ap.add_argument("--gold", required=True, help="data/qna_eval.jsonl")
    args = ap.parse_args()

    preds = load_jsonl(args.pred)
    golds = load_jsonl(args.gold)
    if not preds or not golds:
        print("[error] empty preds or gold")
        return

    # naive 1:1 pairing for demo (extend to ids later)
    p = preds[0]
    g = golds[0]

    em = exact_match(p["answer"], g["answers"])
    rL = rouge_l(p["answer"], g["answers"])

    print(f"Exact Match: {em:.3f}")
    print(f"ROUGE-L    : {rL:.3f}")

    try:
        # Optional: simple RAGAS usage example
        from ragas.metrics import faithfulness, answer_relevancy
        from ragas import evaluate
        import pandas as pd
        df = pd.DataFrame([{
            "question": g.get("question", ""),
            "contexts": [""],  # You can store retrieved contexts and pass them here.
            "answer": p.get("answer", ""),
            "ground_truth": g.get("answers", [""])[0],
        }])
        res = evaluate(df, metrics=[faithfulness, answer_relevancy])
        print(res)
    except Exception as e:
        print(f"[info] RAGAS not executed: {e}")

if __name__ == "__main__":
    main()