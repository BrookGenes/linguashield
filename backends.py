"""
backends.py - every "model" LinguaShield can test lives here.

A backend is any object with:   classify(text: str) -> dict
returning {"label": "scam" | "legitimate", "reason": "<one sentence>"}

Backends included:
  keyword   - a classic English-keyword spam filter (the "old-school" baseline)
  openai    - any OpenAI-compatible chat API (OpenAI, Together, a server hosting
              your Adaption model, etc.)  -> use for the BASE model and/or YOURS
  hf_local  - a model you downloaded (e.g. weights from AutoScientist) run locally
"""
import json
import os
import re
import urllib.request

SYSTEM_PROMPT = (
    "You are a security assistant that protects people from scam and phishing messages "
    "in any language. Classify the message the user sends as either \"scam\" or "
    "\"legitimate\". Reply with ONLY a JSON object: "
    "{\"label\": \"scam\" or \"legitimate\", \"reason\": \"one short sentence in English\"}."
)


import unicodedata

_SCAM = {"scam", "estafa", "etan", "ukukhwabanisa", "ukukhwaba", "udanganyifu", "panloloko",
         "inkohliso", "itanjere", "itanje", "jibiti", "etenitenin", "phishing", "fraud"}
_LEGIT = {"legitimate", "legitimo", "lehitimo", "halali", "tito", "legit", "safe", "ham"}


def _plain(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s).lower().strip())
    return "".join(c for c in s if not unicodedata.combining(c))


def _label_from_word(word: str) -> str:
    w = _plain(word)
    if w in _SCAM or "scam" in w or "phish" in w:
        return "scam"
    if w in _LEGIT or "legit" in w:
        return "legitimate"
    return "unknown"


def parse_output(raw: str) -> dict:
    """Turn a model's raw text into {'label', 'reason'}. Understands translated labels too."""
    raw = (raw or "").strip()
    match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            return {"label": _label_from_word(data.get("label", "")),
                    "reason": str(data.get("reason", ""))}
        except json.JSONDecodeError:
            pass
    m = re.search(r'"label"\s*:\s*"([^"]+)"', raw)
    if m:
        return {"label": _label_from_word(m.group(1)), "reason": raw[:200]}
    low = _plain(raw)
    if "legitimate" in low and "scam" not in low:
        return {"label": "legitimate", "reason": raw[:200]}
    if "scam" in low or "phishing" in low:
        return {"label": "scam", "reason": raw[:200]}
    return {"label": "unknown", "reason": raw[:200]}


class KeywordFilter:
    """English-only keyword filter: the 'before' picture of the language security gap."""
    name = "keyword_filter"
    KEYWORDS = [
        "verify your", "account has been locked", "claim now", "you won", "congratulations",
        "redelivery fee", "work from home", "earn $", "your phone has a virus",
        "give the technician", "send your id", "bank number", "send $", "urgently",
        "click here", "password",
    ]

    def classify(self, text: str) -> dict:
        low = text.lower()
        hits = [k for k in self.KEYWORDS if k in low]
        if hits:
            return {"label": "scam", "reason": f"Matched English scam keyword: '{hits[0]}'."}
        return {"label": "legitimate", "reason": "No known scam keywords found."}


class OpenAICompatible:
    """
    Works with any server that speaks the OpenAI /chat/completions format.
    Configure with environment variables (put them in a .env file or export them):
        LLM_BASE_URL   e.g. https://api.openai.com/v1
        LLM_API_KEY    your key
        LLM_MODEL      model name / id
    Use a second set (prefix MINE_) for your Adaption-trained model, see get_backend().
    """

    def __init__(self, base_url, api_key, model, name):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.name = name

    def classify(self, text: str) -> dict:
        body = json.dumps({
            "model": self.model,
            "temperature": 0,
            "max_tokens": 150,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        return parse_output(data["choices"][0]["message"]["content"])


class HFLocal:
    """Run downloaded model weights locally. Needs: pip install transformers torch"""

    def __init__(self, model_path, name):
        from transformers import pipeline  # imported lazily so other backends don't need it
        self.pipe = pipeline("text-generation", model=model_path, device_map="auto")
        self.name = name

    def classify(self, text: str) -> dict:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        out = self.pipe(messages, max_new_tokens=120, do_sample=False)
        return parse_output(out[0]["generated_text"][-1]["content"])


def get_backend(kind: str):
    """kind = keyword | base | mine | local"""
    if kind == "keyword":
        return KeywordFilter()
    if kind == "base":      # general-purpose model (the baseline you are trying to beat)
        return OpenAICompatible(os.environ["BASE_URL"], os.environ["BASE_API_KEY"],
                                os.environ["BASE_MODEL"], "base_model")
    if kind == "mine":      # YOUR Adaption-trained model, served through an OpenAI-style API
        return OpenAICompatible(os.environ["MINE_URL"], os.environ["MINE_API_KEY"],
                                os.environ["MINE_MODEL"], "adaption_model")
    if kind == "local":     # YOUR Adaption-trained model, weights downloaded to ./models/
        return HFLocal(os.environ.get("LOCAL_MODEL_PATH", "models/linguashield"),
                       "adaption_model")
    raise ValueError(f"Unknown backend '{kind}'. Use keyword, base, mine or local.")
