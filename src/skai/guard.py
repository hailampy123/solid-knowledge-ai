"""Guardrails: PII/secret redaction, prompt-injection detection, output policy.

Deterministic, offline, no LLM — so it runs on ingest (keyless) and in tests.
Called at two seams:
- ingest time (`ingest/chunk.py`): redact PII, quarantine injected chunks.
- generate time (`agent/nodes.py`): delimiter-wrap context, redact the answer,
  refuse denied topics.

Every pattern list below is the tuning knob: add patterns, don't rewrite logic.
"""
from __future__ import annotations

import re

# --- PII / secrets -----------------------------------------------------------

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# SSN: require the dashes so we don't redact any 9-digit number.
_SSN_RE = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
# Phone: US / E.164-ish, needs the 3-3-4 shape (bare or separated).
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}(?!\d)")
# Credit-card candidate: 13-19 digits, optional single space/dash. Luhn-checked.
_CC_RE = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
# Named secrets. The password/api_key form keeps the label, redacts the value.
_SK_RE = re.compile(r"\bsk-[A-Za-z0-9]{16,}\b")
_AKIA_RE = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
_GHP_RE = re.compile(r"\bghp_[A-Za-z0-9]{20,}\b")
_KV_SECRET_RE = re.compile(
    r"(?i)\b(password|passwd|api[_-]?key|secret|token)\b(\s*[:=]\s*)(\S+)"
)


def _luhn_ok(digits: str) -> bool:
    nums = [int(c) for c in digits]
    if not 13 <= len(nums) <= 19:
        return False
    checksum, parity = 0, len(nums) % 2
    for i, d in enumerate(nums):
        if i % 2 == parity:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        checksum += d
    return checksum % 10 == 0


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Return (redacted_text, sorted kinds found). Kinds: email, ssn, phone,
    credit_card, secret. Idempotent: [REDACTED_*] tokens carry no PII to re-match.
    """
    found: set[str] = set()

    def sub(regex, kind, repl, s):
        new, n = regex.subn(repl, s)
        if n:
            found.add(kind)
        return new

    # Secrets first so their (possibly digit-heavy) values aren't half-eaten.
    text = sub(_SK_RE, "secret", "[REDACTED_SECRET]", text)
    text = sub(_AKIA_RE, "secret", "[REDACTED_SECRET]", text)
    text = sub(_GHP_RE, "secret", "[REDACTED_SECRET]", text)
    text = sub(_KV_SECRET_RE, "secret", r"\1\2[REDACTED_SECRET]", text)
    text = sub(_EMAIL_RE, "email", "[REDACTED_EMAIL]", text)
    text = sub(_SSN_RE, "ssn", "[REDACTED_SSN]", text)

    def cc_repl(m):
        digits = re.sub(r"\D", "", m.group(0))
        if _luhn_ok(digits):
            found.add("credit_card")
            return "[REDACTED_CREDIT_CARD]"
        return m.group(0)  # 16 digits that fail Luhn are not a card

    text = _CC_RE.sub(cc_repl, text)
    text = sub(_PHONE_RE, "phone", "[REDACTED_PHONE]", text)
    return text, sorted(found)


# --- prompt injection --------------------------------------------------------

_INJECTION_RES = [
    ("ignore_instructions", re.compile(
        r"(?i)ignore\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+instructions")),
    ("disregard", re.compile(
        r"(?i)disregard\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above)")),
    ("role_override", re.compile(r"(?i)you\s+are\s+now\s+")),
    ("system_prompt", re.compile(
        r"(?i)system\s+prompt|reveal\s+your\s+(?:system\s+)?(?:instructions|prompt)")),
    ("developer_mode", re.compile(r"(?i)developer\s+mode")),
]


def scan_injection(text: str) -> list[str]:
    """Return names of matched injection patterns; empty list means clean."""
    return [name for name, rx in _INJECTION_RES if rx.search(text)]


# --- output policy -----------------------------------------------------------

def check_output_policy(question: str, refusal_topics: list[str] | None) -> str | None:
    """Return a refusal reason if the question hits a denied topic, else None.

    `refusal_topics` empty/None => always None (feature off, no behavior change).
    """
    ql = question.lower()
    for topic in refusal_topics or []:
        t = topic.strip().lower()
        if t and t in ql:
            return f"This assistant is not permitted to answer questions about {topic.strip()}."
    return None
