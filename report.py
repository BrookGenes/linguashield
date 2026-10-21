"""
report.py - combine everything in results/*.csv into a summary table + the gap chart.
Run it after you have run eval.py for 2+ backends.
"""
import glob
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LANG_NAMES = {"en": "English", "es": "Spanish", "sw": "Swahili", "tl": "Tagalog",
              "yo": "Yoruba", "zu": "Zulu"}
ORDER = ["keyword_filter", "base_model", "adaption_model"]
LABELS = {"keyword_filter": "Keyword filter", "base_model": "Base LLM",
          "adaption_model": "LinguaShield (Adaption)"}

rows = []
for path in glob.glob("results/*.csv"):
    if os.path.basename(path) == "summary.csv":
        continue
    name = os.path.basename(path)[:-4]
    d = pd.read_csv(path)
    for lang, g in d.groupby("language"):
        scams, legit = g[g.label == "scam"], g[g.label == "legitimate"]
        rows.append({
            "model": name, "language": lang,
            "accuracy": g.correct.mean(),
            "scam_detection_rate": (scams.pred == "scam").mean(),      # recall on scams
            "false_alarm_rate": (legit.pred == "scam").mean(),         # legit flagged as scam
        })
summary = pd.DataFrame(rows)
summary.to_csv("results/summary.csv", index=False)
print(summary.round(2).to_string(index=False))

models = [m for m in ORDER if m in set(summary.model)]
langs = list(summary.language.unique())
fig, ax = plt.subplots(figsize=(9, 5))
width = 0.8 / max(len(models), 1)
for i, m in enumerate(models):
    vals = [summary[(summary.model == m) & (summary.language == l)].scam_detection_rate.mean()
            for l in langs]
    ax.bar([x + i * width for x in range(len(langs))], [v * 100 for v in vals],
           width, label=LABELS[m])
ax.set_xticks([x + width * (len(models) - 1) / 2 for x in range(len(langs))])
ax.set_xticklabels([LANG_NAMES.get(l, l) for l in langs])
ax.set_ylabel("Scams caught (%)")
ax.set_ylim(0, 105)
ax.set_title("The language security gap: scam detection rate by language")
ax.legend()
plt.tight_layout()
plt.savefig("results/gap_chart.png", dpi=150)
print("\nSaved results/summary.csv and results/gap_chart.png")
