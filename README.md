# 🛡️ LinguaShield
**Scam detection for the languages security tools forget.**
Built for the *Best Use of Adaption* challenge (LaserHacks 2026) using **Adaptive Data (Invent a Dataset)** and **AutoScientist**.

## The problem
Scam filters work best in English. Attackers know it and target speakers of lower-resource languages, where filters and public training data are weakest.

## What we did
1. **Red team:** showed that a classic English keyword filter (and a general LLM) miss scams in other languages.
2. **Adaptive Data / Invent a Dataset:** generated a labeled multilingual dataset of scam and legitimate messages (including hard negatives).
3. **AutoScientist:** trained a classifier on that dataset.
4. **Evaluated** on a hand-written held-out test set the model never saw. See `results/gap_chart.png`.

## Responsible use
Defensive only. All data is synthetic: placeholder brands (ExampleBank), `[link]` instead of URLs, fake phone numbers. The tool classifies messages; it does not write them.

## Project layout
| File | What it does |
|---|---|
| `data/test_set.csv` | 48 hand-written held-out test messages (EN, ES, SW, TL) |
| `adaption/1_invent_dataset_prompt.md` | Text to paste into Invent a Dataset |
| `adaption/2_autoscientist_objective.md` | Text to paste into AutoScientist |
| `scripts/inspect_dataset.py` | Quality-checks Adaption's dataset before training |
| `backends.py` | The models we compare (keyword filter, base LLM, your model) |
| `eval.py` | Runs one model on the test set |
| `report.py` | Builds the summary table and gap chart |
| `app.py` | The Streamlit demo |

Note: we did not run our trained model on our own test set. The 68% vs. 32% win rate is from Adaption’s AutoScientist evaluation. Our test set was used to show the language gap with a keyword-filter baseline.
