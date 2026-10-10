"""Calibrate embed cosine and relate thresholds for imprint duplicate gate."""

from __future__ import annotations

import argparse
import os
import statistics
import sys

from relate import Relater
from samples import ACCEPTANCE_PAIR, GROUPS

DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_RELATE = "cross-encoder/nli-deberta-v3-small"
DEFAULT_CACHE = os.path.expanduser("~/.cache/imprint/models")
CANDIDATES = [round(0.60 + 0.01 * i, 2) for i in range(33)]


def load_embed(model: str, cache: str):
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
    if not values:
        print(f"  {name}: (empty)")
        return
    ordered = sorted(values)
    q = statistics.quantiles(ordered, n=4, method="inclusive")
    print(
        f"  {name:<16} n={len(values):<3} "
        f"min={ordered[0]:.3f} p25={q[0]:.3f} median={q[1]:.3f} "
        f"p75={q[2]:.3f} max={ordered[-1]:.3f} mean={statistics.mean(values):.3f}"
    )


def pick_label(scores: dict[str, float], *names: str) -> float:
    for name in names:
        for key, val in scores.items():
            if key.lower() == name.lower():
                return float(val)
    return 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("IMPRINT_EMBED_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--relate-model",
        default=os.environ.get("IMPRINT_RELATE_MODEL", DEFAULT_RELATE),
    )
    parser.add_argument("--cache-dir", default=os.environ.get("IMPRINT_EMBED_CACHE", DEFAULT_CACHE))
    parser.add_argument(
        "--write-yaml",
        default="",
        help="optional path to write recommended imprint.yaml snippet",
    )
    args = parser.parse_args()

    print(f"embed model: {args.model}")
    print(f"relate model: {args.relate_model}\n")
    embed = load_embed(args.model, args.cache_dir)
    relater = Relater(args.relate_model, args.cache_dir)

    pairs = [(g, a, b) for g, group in GROUPS.items() for a, b in group]
    texts = [t for _, a, b in pairs for t in (a, b)]
    vectors = [list(map(float, v)) for v in embed.embed(texts)]
    print(f"embedded {len(vectors)} texts, dim={len(vectors[0])}\n")

    cos_scores: dict[str, list[float]] = {}
    idx = 0
    for g, a, b in pairs:
        cos_scores.setdefault(g, []).append(cosine(vectors[idx], vectors[idx + 1]))
        idx += 2

    print("cosine distribution by group")
    for g in ("duplicate", "conflict", "unrelated"):
        describe(g, cos_scores[g])

    relate_rows: dict[str, list[tuple[float, float, float]]] = {}
    batch_pairs = [(a, b) for _, a, b in pairs]
    rel_scores = relater.score_pairs(batch_pairs)
    for (g, _a, _b), sc in zip(pairs, rel_scores, strict=True):
        relate_rows.setdefault(g, []).append(
            (
                pick_label(sc, "contradiction"),
                pick_label(sc, "entailment"),
                pick_label(sc, "neutral"),
            )
        )

    print("\nrelate contradiction by group")
    for g in ("duplicate", "conflict", "unrelated"):
        describe(g, [r[0] for r in relate_rows[g]])
    print("\nrelate entailment by group")
    for g in ("duplicate", "conflict", "unrelated"):
        describe(g, [r[1] for r in relate_rows[g]])

    dup_cos = cos_scores["duplicate"]
    con_cos = cos_scores["conflict"]
    unr_cos = cos_scores["unrelated"]

    print("\nthreshold sweep, cosine only")
    print(f"  {'thresh':>7} {'dup_recall':>11} {'conflict_fpr':>13} {'unrelated_fpr':>14}")
    best_cos = None
    for t in CANDIDATES:
        recall = sum(1 for s in dup_cos if s >= t) / len(dup_cos)
        cfpr = sum(1 for s in con_cos if s >= t) / len(con_cos)
        ufpr = sum(1 for s in unr_cos if s >= t) / len(unr_cos)
        print(f"  {t:>7.2f} {recall:>10.0%} {cfpr:>12.0%} {ufpr:>13.0%}")
        if cfpr <= 0.10 and ufpr == 0.0:
            if best_cos is None or recall > best_cos[1]:
                best_cos = (t, recall)

    dup_rows = relate_rows["duplicate"]
    con_rows = relate_rows["conflict"]
    aligned: list[tuple[str, float, float, float]] = []
    for g, group in GROUPS.items():
        for j in range(len(group)):
            aligned.append(
                (
                    g,
                    cos_scores[g][j],
                    relate_rows[g][j][0],
                    relate_rows[g][j][1],
                )
            )

    print("\nthreshold sweep, cosine + relate (Go duplicate gate)")
    print(
        "  dup: cosine>=T and entail>=E and contradiction<C  |  advisory: contradiction>=C"
    )
    best = None
    for t in CANDIDATES:
        for c_thr in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
            for e_thr in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
                dup_hit = sum(
                    1
                    for g, cos, con, ent in aligned
                    if g == "duplicate"
                    and cos >= t
                    and ent >= e_thr
                    and con < c_thr
                )
                con_block = sum(
                    1
                    for g, cos, con, ent in aligned
                    if g == "conflict"
                    and cos >= t
                    and ent >= e_thr
                    and con < c_thr
                )
                unr_block = sum(
                    1
                    for g, cos, con, ent in aligned
                    if g == "unrelated"
                    and cos >= t
                    and ent >= e_thr
                    and con < c_thr
                )
                recall = dup_hit / len(dup_rows)
                cfpr = con_block / len(con_rows)
                ufpr = unr_block / len(unr_cos)
                if cfpr <= 0.10 and ufpr == 0.0 and recall >= 0.85:
                    if best is None or recall > best[0]:
                        best = (recall, t, c_thr, e_thr)

    acc_cos = cosine(vectors[0], vectors[1])
    print("\nacceptance pair")
    print(f"  {ACCEPTANCE_PAIR[0]!r}")
    print(f"  {ACCEPTANCE_PAIR[1]!r}")
    print(f"  cosine = {acc_cos:.3f}")

    dup_t = best_cos[0] if best_cos else 0.70
    con_min = 0.50
    ent_min = 0.50
    if best:
        _, dup_t, con_min, ent_min = best

    yaml_snippet = f"""# Recommended plugins.embed.config (from calibrate.py)
plugins:
  embed:
    config:
      model: {args.model}
      relate_model: {args.relate_model}
      duplicate_threshold: {dup_t:.2f}
      relate_contradiction_min: {con_min:.2f}
      relate_entailment_min: {ent_min:.2f}
      timeout_seconds: 4
"""

    print("\nrecommended imprint.yaml snippet:")
    print(yaml_snippet)

    if args.write_yaml:
        os.makedirs(os.path.dirname(args.write_yaml) or ".", exist_ok=True)
        with open(args.write_yaml, "w", encoding="utf-8") as f:
            f.write(yaml_snippet)
        print(f"wrote {args.write_yaml}")

    if best_cos is None and best is None:
        print("warning: no conservative threshold found; tune manually from distributions above")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
