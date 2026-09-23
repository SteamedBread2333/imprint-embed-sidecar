"""Calibrate the semantic duplicate threshold for imprint-embed.

Run after scripts/setup.sh:

    ./run.sh --warm            # in one terminal, or
    python3 calibrate.py       # standalone, loads the model itself

Output
  1. cosine distribution per group (duplicate / conflict / unrelated)
  2. a decision matrix over candidate thresholds: duplicate recall vs
     false-positive rate on conflict and unrelated pairs
  3. the acceptance check: the PascalCase pair (Jaccard 0.38) must be a duplicate
  4. a recommended threshold

Selection rule (conservative by design): among thresholds whose conflict
false-positive rate is 0, pick the one with the highest duplicate recall. If no
threshold achieves 0 conflict false positives, report the trade-off instead of
silently picking a permissive value — a false rejection costs the user a real
memory, a missed duplicate only costs a duplicate.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys

from polarity import polarity_conflict
from samples import ACCEPTANCE_PAIR, GROUPS

DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_CACHE = os.path.expanduser("~/.cache/imprint/models")
CANDIDATES = [round(0.60 + 0.01 * i, 2) for i in range(33)]  # 0.60 .. 0.92


def load(model: str, cache: str):
    try:
        from fastembed import TextEmbedding
    except ImportError:
        print("calibrate: fastembed missing; run scripts/setup.sh", file=sys.stderr)
        raise SystemExit(1)
    return TextEmbedding(model_name=model, cache_dir=cache)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


def describe(name: str, values: list[float]) -> None:
    ordered = sorted(values)
    q = statistics.quantiles(ordered, n=4, method="inclusive")
    print(
        f"  {name:<10} n={len(values):<3} "
        f"min={ordered[0]:.3f} p25={q[0]:.3f} median={q[1]:.3f} "
        f"p75={q[2]:.3f} max={ordered[-1]:.3f} mean={statistics.mean(values):.3f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("IMPRINT_EMBED_MODEL", DEFAULT_MODEL))
    parser.add_argument("--cache-dir", default=os.environ.get("IMPRINT_EMBED_CACHE", DEFAULT_CACHE))
    args = parser.parse_args()

    print(f"model: {args.model}")
    model = load(args.model, args.cache_dir)

    pairs = [(g, a, b) for g, group in GROUPS.items() for a, b in group]
    texts = [t for _, a, b in pairs for t in (a, b)]
    vectors = [list(map(float, v)) for v in model.embed(texts)]
    print(f"embedded {len(vectors)} texts, dim={len(vectors[0])}\n")

    scores: dict[str, list[float]] = {}
    idx = 0
    for g, a, b in pairs:
        scores.setdefault(g, []).append(cosine(vectors[idx], vectors[idx + 1]))
        idx += 2

    print("cosine distribution by group")
    for g in ("duplicate", "conflict", "unrelated"):
        describe(g, scores[g])

    dup, con, unr = scores["duplicate"], scores["conflict"], scores["unrelated"]
    print("\nthreshold sweep, cosine only (recall = duplicates caught, fpr = wrongly caught)")
    print(f"  {'thresh':>7} {'dup_recall':>11} {'conflict_fpr':>13} {'unrelated_fpr':>14}")
    for t in CANDIDATES:
        recall = sum(1 for s in dup if s >= t) / len(dup)
        cfpr = sum(1 for s in con if s >= t) / len(con)
        ufpr = sum(1 for s in unr if s >= t) / len(unr)
        print(f"  {t:>7.2f} {recall:>10.0%} {cfpr:>12.0%} {ufpr:>13.0%}")

    # Second pass: cosine AND polarity guard. A pair is only hard-rejected when
    # cosine is high and no polarity asymmetry was detected.
    print("\nthreshold sweep, cosine + polarity guard (hard-reject path)")
    print(f"  {'thresh':>7} {'dup_recall':>11} {'conflict_fpr':>13} {'unrelated_fpr':>14}")
    best = None
    dscores = [(cosine(vectors[i], vectors[i + 1]), pairs[i // 2][1], pairs[i // 2][2])
               for i in range(0, len(pairs) * 2, 2)]
    by_group: dict[str, list[tuple[float, bool]]] = {}
    k = 0
    for g, group in GROUPS.items():
        rows = []
        for a, b in group:
            s = cosine(vectors[k], vectors[k + 1])
            rows.append((s, polarity_conflict(a, b)))
            k += 2
        by_group[g] = rows
    for t in CANDIDATES:
        recall = sum(1 for s, p in by_group["duplicate"] if s >= t and not p) / len(dup)
        cfpr = sum(1 for s, p in by_group["conflict"] if s >= t and not p) / len(con)
        ufpr = sum(1 for s, p in by_group["unrelated"] if s >= t and not p) / len(unr)
        print(f"  {t:>7.2f} {recall:>10.0%} {cfpr:>12.0%} {ufpr:>13.0%}")
        if cfpr == 0.0 and ufpr == 0.0:
            if best is None or recall > best[1]:
                best = (t, recall)
    flagged = sum(1 for s, p in by_group["conflict"] if p)
    print(f"\n  polarity guard fired on {flagged}/{len(con)} conflict pairs")

    print("\nacceptance pair")
    va, vb = vectors[0], vectors[1]
    acc = cosine(va, vb)
    print(f"  {ACCEPTANCE_PAIR[0]!r}")
    print(f"  {ACCEPTANCE_PAIR[1]!r}")
    print(f"  cosine = {acc:.3f}")

    print("\nrecommendation")
    if best is None:
        overlap = max(max(con), max(unr))
        print(
            "  no threshold separates duplicates from conflicts on this sample; "
            f"lowest safe floor is above {overlap:.3f} — treat the embed gate as "
            "advisory (return candidates, do not hard-reject) or pick a better model."
        )
        return 1
    thresh, recall = best
    print(
        f"  duplicate_threshold = {thresh:.2f} "
        f"(duplicate recall {recall:.0%}, zero conflict/unrelated false positives)"
    )
    print(f"  acceptance pair caught at {thresh:.2f}: {acc >= thresh}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
