"""Batch-evaluate imprint semantic gate on labeled pairs via live sidecar HTTP."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

from samples import GROUPS

DEFAULT_URL = "http://127.0.0.1:4174"
DEFAULT_T = 0.65
DEFAULT_C = 0.40
DEFAULT_E = 0.35


def post_json(url: str, path: str, body: dict, timeout: float) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{url.rstrip('/')}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def classify_relate(sc: dict[str, float], c_min: float, e_min: float) -> str:
    con = max((v for k, v in sc.items() if k.lower() == "contradiction"), default=0.0)
    ent = max((v for k, v in sc.items() if k.lower() == "entailment"), default=0.0)
    if con >= c_min:
        return "advisory"
    if ent >= e_min:
        return "block"
    return "pass"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("IMPRINT_EMBED_URL", DEFAULT_URL))
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--threshold", type=float, default=float(os.environ.get("IMPRINT_DUP_T", DEFAULT_T)))
    parser.add_argument("--contradiction-min", type=float, default=DEFAULT_C)
    parser.add_argument("--entailment-min", type=float, default=DEFAULT_E)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    pairs: list[tuple[str, str, str]] = []
    for group, rows in GROUPS.items():
        for a, b in rows:
            pairs.append((group, a, b))

    texts = [t for g, a, b in pairs for t in (a, b)]
    try:
        emb = post_json(args.url, "/embed", {"texts": texts}, args.timeout)
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"eval_gate: embed failed: {exc}", file=sys.stderr)
        return 1

    vectors = emb["vectors"]
    idx = 0
    cos_rows: list[tuple[str, str, str, float]] = []
    for g, a, b in pairs:
        va, vb = vectors[idx], vectors[idx + 1]
        idx += 2
        dot = sum(x * y for x, y in zip(va, vb))
        na = sum(x * x for x in va) ** 0.5
        nb = sum(y * y for y in vb) ** 0.5
        cos = 0.0 if na == 0 or nb == 0 else dot / (na * nb)
        cos_rows.append((g, a, b, cos))

    batch_pairs = [[a, b] for _g, a, b, _ in cos_rows]
    try:
        rel = post_json(args.url, "/relate", {"pairs": batch_pairs}, args.timeout)
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"eval_gate: relate failed: {exc}", file=sys.stderr)
        return 1

    results = rel["results"]
    stats: dict[str, dict[str, int]] = {g: {"block": 0, "advisory": 0, "pass": 0, "below_cos": 0} for g in GROUPS}
    failures: list[str] = []

    print(
        f"sidecar {args.url}  T={args.threshold:.2f}  C={args.contradiction_min:.2f}  "
        f"E={args.entailment_min:.2f}\n"
    )
    print(f"{'group':<11} {'cos':>6} {'outcome':<10} claim_a … claim_b")
    print("-" * 100)

    for (g, a, b, cos), sc in zip(cos_rows, results, strict=True):
        if cos < args.threshold:
            outcome = "below_cos"
            stats[g]["below_cos"] += 1
        else:
            outcome = classify_relate(sc, args.contradiction_min, args.entailment_min)
            stats[g][outcome] += 1

        if g == "duplicate" and outcome != "block":
            failures.append(f"duplicate MISS ({outcome}) cos={cos:.3f}: {a!r} vs {b!r}")
        if g == "conflict" and outcome == "block":
            failures.append(f"conflict FALSE-BLOCK cos={cos:.3f}: {a!r} vs {b!r}")
        if g == "unrelated" and outcome == "block":
            failures.append(f"unrelated FALSE-BLOCK cos={cos:.3f}: {a!r} vs {b!r}")

        if args.verbose or outcome in ("block", "advisory") or g == "duplicate":
            short_a = (a[:42] + "…") if len(a) > 45 else a
            short_b = (b[:42] + "…") if len(b) > 45 else b
            print(f"{g:<11} {cos:>6.3f} {outcome:<10} {short_a} | {short_b}")

    print("\nsummary (semantic path only: cosine + relate, no Jaccard/vault)")
    for g in ("duplicate", "conflict", "unrelated"):
        s = stats[g]
        n = len(GROUPS[g])
        caught = s["block"]
        print(
            f"  {g:<11} n={n}  block={caught}/{n} ({100 * caught / n:.0f}%)  "
            f"advisory={s['advisory']}  pass={s['pass']}  below_cos={s['below_cos']}"
        )

    dup_recall = stats["duplicate"]["block"] / len(GROUPS["duplicate"])
    con_fpr = stats["conflict"]["block"] / len(GROUPS["conflict"])
    unr_fpr = stats["unrelated"]["block"] / len(GROUPS["unrelated"])
    print(
        f"\n  duplicate recall={dup_recall:.0%}  conflict false-block={con_fpr:.0%}  "
        f"unrelated false-block={unr_fpr:.0%}"
    )

    if failures:
        print(f"\n{len(failures)} expectation mismatch(es):")
        for line in failures:
            print(f"  - {line}")
        return 1
    print("\nall labeled pairs match expectations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
