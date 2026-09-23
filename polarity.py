"""Polarity guard: cheap lexical signal that catches antonym conflicts.

Calibration on bge-small-zh-v1.5 showed cosine cannot separate duplicates from
conflicts (duplicate median 0.823 vs conflict median 0.753, max 1.000 for
"Prefer tabs over spaces" / "Prefer spaces over tabs"). Cosine measures topic
relatedness and antonyms are maximally topically related.

This guard is the second signal: high cosine PLUS a polarity asymmetry means
"same topic, likely opposite policy" -> do not hard-reject, hand the decision
back to the LLM. It is a heuristic, not a classifier: it favours precision
(only fire on explicit markers) because a missed polarity signal costs a false
rejection, which is worse than a missed duplicate.

Keep this list in sync with pkg/imprint/polarity.go.
"""

from __future__ import annotations

import re

NEGATION_MARKERS = (
    # English
    "never", "do not", "don't", "dont", "avoid", "without", "instead of",
    "rather than", "skip", "ignore", "forbid", "no ", "not ", "bare ",
    "must not", "should not", "refrain",
    # Chinese
    "不要", "别", "禁止", "避免", "不可", "不能", "而非", "而不是", "无需", "禁用",
)

# Antonym token pairs: one side in each claim signals an opposite policy.
ANTONYM_PAIRS: tuple[tuple[str, ...], tuple[str, ...]] = (
    (("tabs", "tab"), ("spaces", "space")),
    (("snake_case", "snakecase"), ("camelcase", "camel_case")),
    (("short", "shorter", "small"), ("long", "longer", "large")),
    (("wrap", "wrapping", "wrapped"), ("bare", "unwrap", "raw")),
    (("structured",), ("plain", "formatted")),
    (("composition",), ("inheritance",)),
    (("vendor", "vendored"), ("novendor", "no_vendor")),
    (("early return", "early returns"), ("single exit", "one exit")),
    (("add", "write", "include"), ("skip", "omit", "remove")),
    (("explicit", "explicitly"), ("ignore", "discard")),
    (("pin", "pinned", "fixed"), ("floating", "range", "latest")),
    (("english",), ("chinese",)),
)

_WORD = re.compile(r"[a-z0-9_]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def has_negation(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in NEGATION_MARKERS)


def has_antonym_pair(a: str, b: str) -> bool:
    ta, tb = _tokens(a), _tokens(b)
    for left, right in ANTONYM_PAIRS:
        if (ta & set(left) and tb & set(right)) or (tb & set(left) and ta & set(right)):
            return True
    # Same claim shape with swapped word order ("prefer X over Y" vs "prefer Y
    # over X") is also an inversion signal.
    if ta and tb and ta == tb and a.lower() != b.lower():
        return True
    return False


def polarity_conflict(a: str, b: str) -> bool:
    """True when the two claims likely express opposite policies."""
    if has_antonym_pair(a, b):
        return True
    return has_negation(a) != has_negation(b)
