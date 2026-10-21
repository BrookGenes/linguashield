"""
inspect_dataset.py - sanity-check the dataset Adaption generated BEFORE you train on it.

Usage:  python scripts/inspect_dataset.py data/adaption_dataset.csv
It auto-detects the message / label / language columns. Open the printed report
and also read 20-30 rows by hand.
"""
import re
import sys
import pandas as pd

path = sys.argv[1]
df = pd.read_json(path, lines=True) if path.endswith(".jsonl") else pd.read_csv(path)
print(f"Rows: {len(df)}\nColumns: {list(df.columns)}\n")

def find(*hints):
    for c in df.columns:
        if any(h in c.lower() for h in hints):
            return c

text_col = find("message", "text", "content", "input")
label_col = find("label", "class", "category")
lang_col = find("lang")
reason_col = find("reason", "explanation")
print(f"Detected -> text: {text_col}, label: {label_col}, language: {lang_col}, reason: {reason_col}\n")

if label_col:
    print("Label balance:\n", df[label_col].value_counts(), "\n")
if lang_col:
    print("Rows per language:\n", df[lang_col].value_counts(), "\n")
if text_col:
    t = df[text_col].astype(str)
    print(f"Exact duplicate messages: {t.duplicated().sum()}")
    real_urls = t.str.contains(r"https?://|www\.", regex=True)
    print(f"Messages containing REAL-looking URLs (should be 0, use [link]): {real_urls.sum()}")
    phones = t.str.contains(r"\+?\d[\d\s\-]{8,}\d", regex=True)
    print(f"Messages containing long phone-like numbers (review): {phones.sum()}")
    overlap = set(t.str.lower().str.strip()) & set(
        pd.read_csv("data/test_set.csv").text.str.lower().str.strip())
    print(f"Overlap with your held-out test set (MUST be 0): {len(overlap)}")
    print("\n--- 10 random rows to eyeball ---")
    print(df.sample(min(10, len(df)), random_state=1).to_string())
