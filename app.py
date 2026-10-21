"""
app.py - the LinguaShield demo.   Run with:   streamlit run app.py
Paste a message in any language, get a verdict + reason from your chosen model.
The 'Results' tab shows the language-gap chart you generate with report.py.
"""
import os
import pandas as pd
import streamlit as st
from backends import get_backend

st.set_page_config(page_title="LinguaShield", page_icon="🛡️", layout="centered")
st.title("🛡️ LinguaShield")
st.caption("Scam detection for the languages security tools forget. "
           "Built with Adaption (Adaptive Data + AutoScientist).")

test = pd.read_csv("data/test_set.csv")
tab_demo, tab_results, tab_about = st.tabs(["Try it", "Results", "About & ethics"])

with tab_demo:
    options = {"Keyword filter (old-school baseline)": "keyword",
               "Base LLM": "base",
               "LinguaShield (Adaption-trained)": "mine",
               "LinguaShield (local weights)": "local"}
    choice = st.selectbox("Detector", list(options))
    sample = st.selectbox("Load an example (or type your own below)",
                          ["-- custom --"] + [f"{r.id}: {r.text[:60]}..." for r in test.itertuples()])
    default = ""
    if sample != "-- custom --":
        default = test[test.id == sample.split(":")[0]].text.iloc[0]
    msg = st.text_area("Message to check", value=default, height=120)
    if st.button("Check message") and msg.strip():
        try:
            out = get_backend(options[choice]).classify(msg)
            if out["label"] == "scam":
                st.error("🚨 Likely SCAM")
            elif out["label"] == "legitimate":
                st.success("✅ Looks legitimate")
            else:
                st.warning("Model gave an unclear answer")
            st.write(out["reason"])
        except KeyError as e:
            st.warning(f"Missing setting {e}. Set your environment variables first (see README).")
        except Exception as e:
            st.warning(f"Could not run this detector: {e}")

with tab_results:
    if os.path.exists("results/gap_chart.png"):
        st.image("results/gap_chart.png")
        st.dataframe(pd.read_csv("results/summary.csv").round(2))
    else:
        st.info("Run `python eval.py ...` and `python report.py` to generate results.")

with tab_about:
    st.markdown("""
**Problem:** scam filters are strongest in English. Attackers target speakers of
lower-resource languages, where filters and training data are weakest.

**Method:** (1) red-team test showing the gap, (2) **Adaptive Data / Invent a Dataset**
creates multilingual labeled scam and legitimate messages, (3) **AutoScientist** trains a model,
(4) evaluate on a hand-written held-out test set.

**Responsible use:** defensive only. All training data is synthetic with placeholder
brands and links. No real victims, numbers or URLs. The tool classifies messages; it does not write them.
""")
