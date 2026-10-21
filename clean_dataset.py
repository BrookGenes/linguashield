"""
clean_dataset.py - fix the problems in the dataset Adaption generated, for free.

Problems it fixes (all seen in the real export):
  1. Labels translated into local languages ("estafa", "ukukhwabanisa", "udanganyifu"...)
     -> converted back to exactly "scam" / "legitimate". Unknown words are DROPPED, never guessed.
  2. Wrong 'language' values (Yoruba rows tagged Swahili, etc.)
     -> language is taken from TARGET LANGUAGE CODE when present.
  3. Real brands (Zelle, DHL, Walmart...), real-looking URLs and e-mail addresses -> rows dropped.
  4. Near-duplicates (same text, different "Ref 1234" number) -> only the first kept.
  5. Any row identical to your held-out test set -> dropped (prevents cheating).
  6. Too many scams vs legitimate messages -> scams are down-sampled per language.

Usage:
    python scripts/clean_dataset.py data/adaption_dataset.csv
Works with .csv, .jsonl or .xlsx. Writes:
    data/clean_dataset.csv   (message,label,reason,language)  - for checking / training
    data/clean_train.jsonl   (prompt,completion)               - for AutoScientist / fine-tuning
"""
import json
import re
import sys
import unicodedata
import pandas as pd

MAX_SCAM_TO_LEGIT_RATIO = 1.5

SCAM_WORDS = {"scam", "estafa", "etan", "ukukhwabanisa", "ukukhwaba", "udanganyifu",
              "panloloko", "inkohliso", "itanjere", "itanje", "jibiti", "etenitenin",
              "phishing", "fraud"}
LEGIT_WORDS = {"legitimate", "legitimo", "lehitimo", "halali", "tito", "legit", "safe"}

LANG_BY_CODE = {"en": "English", "es": "Spanish", "sw": "Swahili", "fil": "Tagalog",
                "tl": "Tagalog", "yo": "Yoruba", "zu": "Zulu"}
LANG_BY_NAME = {"english": "English", "kiingereza": "English",
                "spanish": "Spanish", "espanol": "Spanish",
                "swahili": "Swahili", "kiswahili": "Swahili", "suajili": "Swahili",
                "tagalog": "Tagalog", "tagalo": "Tagalog", "filipino": "Tagalog",
                "yoruba": "Yoruba", "kiyoruba": "Yoruba",
                "zulu": "Zulu", "isizulu": "Zulu"}

REAL_BRANDS = ["zelle", "dhl", "walmart", "netflix", "m-pesa", "mpesa", "paypal",
               "fedex", "ups ", "usps", "venmo", "amazon prime"]


def strip_marks(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s).lower().strip())
    return "".join(c for c in s if not unicodedata.combining(c))


def norm_label(raw):
    w = strip_marks(raw)
    if w in SCAM_WORDS:
        return "scam"
    if w in LEGIT_WORDS:
        return "legitimate"
    return None


def norm_lang(code, name):
    c = strip_marks(code) if isinstance(code, str) else ""
    if c in LANG_BY_CODE:
        return LANG_BY_CODE[c]
    n = strip_marks(name) if isinstance(name, str) else ""
    return LANG_BY_NAME.get(n, "unknown")


def load(path):
    if path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    if path.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path)


def find_col(df, *hints, exclude=()):
    for c in df.columns:
        lc = c.lower()
        if any(h in lc for h in hints) and not any(x in lc for x in exclude):
            return c
    return None


def parse_json_blob(s):
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", str(s), re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


def main(path):
    df = load(path)
    p_col = find_col(df, "prompt", "message", "input", "text", exclude=("length",))
    c_col = find_col(df, "completion", "output", "response")
    code_col = find_col(df, "code")
    print(f"Loaded {len(df)} rows. prompt={p_col!r} completion={c_col!r} lang_code={code_col!r}")
    if not p_col or not c_col:
        sys.exit("Could not find prompt/completion columns. Columns are: %s" % list(df.columns))

    drops = {"bad_json": 0, "unknown_label": 0, "real_brand": 0, "url_or_email": 0,
             "duplicate": 0, "in_test_set": 0}
    test_texts = set()
    try:
        test_texts = set(pd.read_csv("data/test_set.csv").text.str.lower().str.strip())
    except FileNotFoundError:
        print("(data/test_set.csv not found - skipping test-set overlap check)")

    seen, rows = set(), []
    for _, r in df.iterrows():
        msg = str(r[p_col]).strip()
        blob = parse_json_blob(r[c_col])
        if not isinstance(blob, dict):
            drops["bad_json"] += 1
            continue
        label = norm_label(blob.get("label", ""))
        if label is None:
            drops["unknown_label"] += 1
            continue
        low = msg.lower()
        if any(b in low for b in REAL_BRANDS):
            drops["real_brand"] += 1
            continue
        if re.search(r"https?://|www\.|[\w.]+@[\w.]+\.\w+", msg):
            drops["url_or_email"] += 1
            continue
        if low in test_texts:
            drops["in_test_set"] += 1
            continue
        key = re.sub(r"\d+", "", re.sub(r"\s+", " ", low))
        if key in seen:
            drops["duplicate"] += 1
            continue
        seen.add(key)
        lang = norm_lang(r[code_col] if code_col else None, blob.get("language"))
        rows.append({"message": msg, "label": label,
                     "reason": str(blob.get("reason", "")).strip(), "language": lang})

    out = pd.DataFrame(rows)
    print("\nRows dropped:", drops, f"-> {len(out)} kept")

    # balance scams vs legitimate per language
    parts = []
    for lang, g in out.groupby("language"):
        legit, scam = g[g.label == "legitimate"], g[g.label == "scam"]
        cap = max(int(len(legit) * MAX_SCAM_TO_LEGIT_RATIO), 1) if len(legit) else len(scam)
        parts += [legit, scam.sample(min(len(scam), cap), random_state=1)]
    bal = pd.concat(parts).sample(frac=1, random_state=1).reset_index(drop=True)

    print("\nBefore balancing:\n", out.groupby(["language", "label"]).size().unstack(fill_value=0))
    print("\nAfter balancing (this is what you train on):\n",
          bal.groupby(["language", "label"]).size().unstack(fill_value=0))
    n_legit = (bal.label == "legitimate").sum()
    print(f"\nTotal training rows: {len(bal)}   legitimate: {n_legit}")
    if n_legit < 300:
        print("WARNING: very few legitimate examples. Run a second, legitimate-only dataset in "
              "Adaption (see the guide) before training, or the model will cry 'scam' at everything.")
    unk = (bal.language == "unknown").sum()
    if unk:
        print(f"Note: {unk} rows have an unknown language (kept, but excluded from per-language checks).")

    bal.to_csv("data/clean_dataset.csv", index=False)
    with open("data/clean_train.jsonl", "w", encoding="utf-8") as f:
        for r in bal.itertuples():
            f.write(json.dumps({
                "prompt": r.message,
                "completion": json.dumps({"label": r.label, "reason": r.reason},
                                         ensure_ascii=False)}, ensure_ascii=False) + "\n")
    print("\nSaved data/clean_dataset.csv and data/clean_train.jsonl")


if __name__ == "__main__":
    main(sys.argv[1])
