# Paste this into Adaption -> Invent a Dataset

Train a model that classifies short SMS and WhatsApp-style messages as "scam" or "legitimate" and gives a one-sentence reason in English.

Scam types to cover (roughly equal amounts): fake bank account alerts, fake prize or gift-card wins, fake delivery fee notices, fake work-from-home job offers, fake tech-support virus warnings, and "family member with a new number needs money" impersonation.

Legitimate messages to include (about the same amount as scams, as hard negatives): real bank statement notices, appointment reminders, delivery confirmations, library or school notices, one-time verification codes, and ordinary messages from family and friends. Some legitimate messages should mention money, links or urgency so the model learns that those alone do not mean scam.

Languages: English, Spanish, Swahili, Tagalog, Yoruba, Zulu. Write naturally in each language with everyday regional wording, not word-for-word translations of English.

Safety rules for the data: use only placeholder brand names (e.g. "ExampleBank", "SampleMart"), write "[link]" instead of any URL, and use fake placeholder phone numbers. Never include real people, real organizations, or real URLs.

Each row must have these fields:
- message: the text message
- label: exactly "scam" or "legitimate"
- language: the language name
- scam_type: the category (or "none" for legitimate)
- reason: one short English sentence explaining the label
