"""
Prompt injection defense.

Two layers, same as the honeypot sandbox's philosophy (static check first,
structural containment second):

1. PATTERN LAYER: regex signatures for known override / jailbreak phrasing.
   Cheap, catches the common cases, easy to extend.
2. HEURISTIC LAYER: structural signals that don't rely on exact wording --
   suspiciously long base64-like tokens, excessive delimiter/markup density
   (attempts to fake a system message), repeated control characters, and
   raw length (a query 20x longer than typical is worth flagging even if no
   pattern matches).

Neither layer claims to be complete. This stops copy-pasted jailbreak
templates and naive override attempts; it will not stop a novel, carefully
obfuscated attack. That's a genuine limitation, not a gap to paper over --
call it out if asked.
"""
import re
from dataclasses import dataclass, field

# --- Pattern layer ---------------------------------------------------------

_OVERRIDE_PATTERNS = [
    r"ignore (all|any|the)? ?(previous|prior|above|earlier) (instructions|prompts|rules)",
    r"disregard (all|any|the)? ?(previous|prior|above|earlier)",
    r"you are now\b",
    r"forget (all|everything|your instructions)",
    r"new instructions?:",
    r"system prompt",
    r"reveal (your|the) (system )?(prompt|instructions)",
    r"act as if you (have no|had no) (restrictions|rules|filters)",
    r"pretend (you are|to be) (an? )?(unfiltered|unrestricted|jailbroken)",
    r"\bDAN\b",
    r"jailbreak",
    r"do anything now",
    r"override (your|the) (rules|instructions|guidelines)",
    r"\[system\]|\{system\}|<system>",
    r"you (must|will) (comply|obey) (with )?(no|without) (exception|restriction)",
    r"respond (only|purely) in (character|persona) (no matter|regardless)",
]

_COMPILED_OVERRIDE = [re.compile(p, re.IGNORECASE) for p in _OVERRIDE_PATTERNS]

# base64-ish blob: long run of base64 alphabet chars with no spaces
_BASE64_BLOB = re.compile(r"[A-Za-z0-9+/]{60,}={0,2}")

# fake delimiter injection: user trying to close/open tags we use internally
_DELIMITER_SPOOF = re.compile(
    r"</?\s*(user_query|system|assistant|instructions?)\s*>", re.IGNORECASE
)

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

TYPICAL_QUERY_LEN = 400  # chars; queries much longer than this get a length penalty


@dataclass
class RiskAssessment:
    risk_score: float
    matched_patterns: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        from config import INJECTION_RISK_THRESHOLD
        return self.risk_score >= INJECTION_RISK_THRESHOLD


class InjectionDefense:
    """Scores and sanitizes user input before it reaches the LLM."""

    def score(self, text: str) -> RiskAssessment:
        matched: list[str] = []
        flags: list[str] = []
        score = 0.0

        # Pattern layer: each distinct hit adds weight, capped so one
        # extremely repetitive payload doesn't just max out trivially.
        pattern_hits = 0
        for pattern in _COMPILED_OVERRIDE:
            m = pattern.search(text)
            if m:
                pattern_hits += 1
                matched.append(m.group(0))
        if pattern_hits:
            score += min(0.35 * pattern_hits, 0.75)
            flags.append(f"{pattern_hits} override-phrase pattern(s) matched")

        # Heuristic: delimiter spoofing (trying to forge our own tags)
        if _DELIMITER_SPOOF.search(text):
            score += 0.4
            flags.append("attempted delimiter/tag spoofing")

        # Heuristic: long base64-like blob (possible encoded payload)
        if _BASE64_BLOB.search(text):
            score += 0.25
            flags.append("suspicious high-entropy/base64-like blob")

        # Heuristic: control characters (possible terminal/format escape attempt)
        if _CONTROL_CHARS.search(text):
            score += 0.3
            flags.append("embedded control characters")

        # Heuristic: excessive length relative to a normal query
        if len(text) > TYPICAL_QUERY_LEN * 5:
            score += 0.2
            flags.append("abnormally long input")

        score = min(score, 1.0)
        return RiskAssessment(risk_score=round(score, 3), matched_patterns=matched, flags=flags)

    def sanitize(self, text: str, max_len: int = 2000) -> str:
        """Truncate and strip control chars. Does NOT strip flagged phrases --
        we either block on risk score, or we pass the query through wrapped
        in hard delimiters so the model treats it as inert data. Silently
        stripping words gives a false sense of safety while letting subtler
        attacks through unchanged.
        """
        cleaned = _CONTROL_CHARS.sub("", text)
        return cleaned[:max_len]

    def wrap_as_data(self, text: str) -> str:
        """Wrap sanitized user input in explicit delimiters for the prompt,
        paired with a system-level instruction (see graph/debate_graph.py)
        that tells the model this block is data to analyze, never
        instructions to follow.
        """
        safe = self.sanitize(text)
        return f"<user_query>\n{safe}\n</user_query>"


defense = InjectionDefense()
