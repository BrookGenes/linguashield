"""
eval.py - run one backend over the held-out test set and save its predictions.

Usage:
    python eval.py keyword     # old-school English keyword filter
    python eval.py base        # general LLM (needs BASE_* env vars)
    python eval.py mine        # your Adaption model via API (needs MINE_* env vars)
    python eval.py local       # your Adaption model, downloaded weights
"""
import sys
import time
import pandas as pd
from backends import get_backend

kind = sys.argv[1] if len(sys.argv) > 1 else "keyword"
backend = get_backend(kind)
df = pd.read_csv("data/test_set.csv")

preds, reasons = [], []
for i, row in df.iterrows():
    try:
        out = backend.classify(row["text"])
    except Exception as e:  # keep going; record the failure as 'unknown'
        out = {"label": "unknown", "reason": f"ERROR: {e}"}
    preds.append(out["label"])
    reasons.append(out["reason"])
    print(f"[{i+1}/{len(df)}] {row['id']}: true={row['label']} pred={out['label']}")
    time.sleep(0.05)

df["pred"] = preds
df["reason"] = reasons
df["correct"] = df["pred"] == df["label"]
path = f"results/{backend.name}.csv"
df.to_csv(path, index=False)
print(f"\nSaved {path}   overall accuracy = {df['correct'].mean():.1%}")
