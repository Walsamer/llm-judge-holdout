#!/usr/bin/env python3
"""Theoretical jury-size analysis.

Estimates p_s (self-endorsement rate when rejected) and p_p (peer endorsement rate
when rejected) per model from votes.json, then derives the analytical holdout benefit
Delta(n) = FER_inc(n) - FER_hol(n) as a function of jury size n.

Key design note — odd vs even n:
  For ODD n the holdout jury has n-1 (even) judges. Both inclusive and holdout juries
  share the same strict-majority threshold ((n+1)/2 votes needed). Delta is always > 0
  and has the closed-form pivotal-voter formula:

      Delta(n) = p_s * P(Bin(n-1, p_p) = (n-1)/2)

  For EVEN n the holdout jury has n-1 (odd) judges whose threshold is n/2 — one vote
  lower than the inclusive jury's threshold of n/2+1. This makes the holdout jury more
  permissive regardless of self-preference, so Delta can be negative. This is a threshold-
  asymmetry artifact, not a real self-preference effect. We therefore restrict the
  minimum-jury-size analysis to ODD n and recommend designing juries with an odd count.
"""
import json
from math import floor
from pathlib import Path

from scipy.stats import binom as binom_dist

ROOT = Path(__file__).parents[1]
VOTES_PATH = ROOT / "results" / "votes.json"
SIM_PATH = ROOT / "results" / "jury_size_simulation.json"
OUT_PATH = ROOT / "results" / "theoretical_jury_size.json"

ODD_JURY_SIZES = list(range(3, 102, 2))   # odd n: 3, 5, 7, ..., 101
ALL_JURY_SIZES = list(range(3, 52))        # n = 3..51 for the full alternating picture

DELTA_THRESHOLDS = [0.05, 0.02, 0.01, 0.005]

MODEL_NAMES = {
    "gpt-4.1-2025-04-14": "GPT-4.1",
    "claude-opus-4-20250514": "Claude Opus 4",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "grok-3-preview-02-24": "Grok 3",
    "deepseek-v3-0324": "DeepSeek V3",
    "llama-4-maverick-03-26-experimental": "Llama 4 Maverick",
    "mistral-medium-2505": "Mistral Medium",
}


# ---------------------------------------------------------------------------
# Core formulas
# ---------------------------------------------------------------------------

def fer_inclusive(n: int, p_s: float, p_p: float) -> float:
    """FER for inclusive jury of n (1 self + n-1 peers), strict majority."""
    m = floor(n / 2) + 1
    return (
        p_s * binom_dist.sf(m - 2, n - 1, p_p)      # P(B >= m-1)
        + (1 - p_s) * binom_dist.sf(m - 1, n - 1, p_p)  # P(B >= m)
    )


def fer_holdout(n: int, p_p: float) -> float:
    """FER for holdout jury of n-1 peers (self removed), strict majority."""
    m = floor((n - 1) / 2) + 1
    return binom_dist.sf(m - 1, n - 1, p_p)


def theoretical_delta(n: int, p_s: float, p_p: float) -> float:
    return fer_inclusive(n, p_s, p_p) - fer_holdout(n, p_p)


def delta_odd_closed_form(n: int, p_s: float, p_p: float) -> float:
    """Closed-form for odd n: p_s * P(Bin(n-1, p_p) == (n-1)/2)."""
    assert n % 2 == 1
    return p_s * binom_dist.pmf((n - 1) // 2, n - 1, p_p)


# ---------------------------------------------------------------------------
# Parameter estimation from votes.json
# ---------------------------------------------------------------------------

def compute_params(matchups: list) -> dict:
    """Estimate p_s and p_p per model.

    Restricted to matchups where model M is human-rejected.
    p_s = P(self-judge endorses M | M is rejected)
    p_p = P(peer judge endorses M | M is rejected), averaged over all peer judges
    """
    stats: dict[str, dict] = {mid: {"self_for": 0, "self_total": 0,
                                     "peer_for": 0, "peer_total": 0}
                               for mid in MODEL_NAMES}

    for m in matchups:
        if m["human_winner"] == "tie":
            continue
        model_a, model_b = m["model_a"], m["model_b"]

        for model_id, side in [(model_a, "A"), (model_b, "B")]:
            if model_id not in stats:
                continue
            is_rejected = (
                (side == "A" and m["human_winner"] == "model_b") or
                (side == "B" and m["human_winner"] == "model_a")
            )
            if not is_rejected:
                continue

            for v in m["votes"]:
                if v["position_swapped"]:
                    continue
                judge = v["judge_model_id"]
                endorses = int(v["preference"] == side)
                if judge == model_id:
                    stats[model_id]["self_for"] += endorses
                    stats[model_id]["self_total"] += 1
                elif judge in MODEL_NAMES:
                    stats[model_id]["peer_for"] += endorses
                    stats[model_id]["peer_total"] += 1

    result = {}
    for mid, s in stats.items():
        if s["self_total"] > 0 and s["peer_total"] > 0:
            result[mid] = {
                "name": MODEL_NAMES[mid],
                "p_s": s["self_for"] / s["self_total"],
                "p_p": s["peer_for"] / s["peer_total"],
                "n_self": s["self_total"],
                "n_peer": s["peer_total"],
            }
    return result


def pooled_params(params: dict) -> tuple[float, float]:
    total_s = sum(p["n_self"] for p in params.values())
    total_p = sum(p["n_peer"] for p in params.values())
    p_s = sum(p["p_s"] * p["n_self"] for p in params.values()) / total_s
    p_p = sum(p["p_p"] * p["n_peer"] for p in params.values()) / total_p
    return p_s, p_p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    matchups = json.loads(VOTES_PATH.read_text())
    print(f"Loaded {len(matchups)} matchups\n")

    params = compute_params(matchups)

    # --- empirical parameters ---
    print("Per-model empirical parameters (human-rejected matchups only):")
    print(f"{'Model':<20} {'p_s':>6} {'p_p':>6} {'n_self':>7} {'n_peer':>7}")
    print("-" * 55)
    for mid, p in sorted(params.items(), key=lambda x: -x[1]["p_s"]):
        print(f"{p['name']:<20} {p['p_s']:>6.3f} {p['p_p']:>6.3f} "
              f"{p['n_self']:>7} {p['n_peer']:>7}")

    pool_p_s, pool_p_p = pooled_params(params)
    print(f"\nPooled  p_s = {pool_p_s:.4f}   p_p = {pool_p_p:.4f}")

    # --- verify closed form ---
    print("\nClosed-form check (odd n, pooled):")
    for n in [3, 5, 7, 9]:
        g = theoretical_delta(n, pool_p_s, pool_p_p)
        c = delta_odd_closed_form(n, pool_p_s, pool_p_p)
        ok = "YES" if abs(g - c) < 1e-9 else "NO"
        print(f"  n={n}  general={g:+.6f}  closed-form={c:+.6f}  match={ok}")

    # --- odd-n decay table (pooled) ---
    print("\nOdd-n theoretical Delta(n) — pooled parameters:")
    print(f"{'n':>4}  {'FER_inc':>8}  {'FER_hol':>8}  {'Δ':>8}")
    print("-" * 38)
    odd_curve = []
    for n in ODD_JURY_SIZES:
        d = theoretical_delta(n, pool_p_s, pool_p_p)
        fi = fer_inclusive(n, pool_p_s, pool_p_p)
        fh = fer_holdout(n, pool_p_p)
        odd_curve.append({"n": n, "delta": round(d, 6),
                          "fer_inc": round(fi, 6), "fer_hol": round(fh, 6)})
        if n <= 31:
            print(f"{n:>4}  {fi:>8.4f}  {fh:>8.4f}  {d:>+8.4f}")

    # --- threshold crossings (odd n, pooled) ---
    print(f"\nMinimum ODD jury size n where pooled Δ(n) < threshold:")
    for thr in DELTA_THRESHOLDS:
        min_n = next((e["n"] for e in odd_curve if e["delta"] < thr), None)
        print(f"  Δ < {thr:.3f}: n = {min_n if min_n else '> 101'}")

    # --- per-model threshold table ---
    print(f"\nPer-model minimum odd n where Δ < 0.01:")
    print(f"{'Model':<20} {'p_s':>6} {'p_p':>6} {'min odd n':>10}")
    print("-" * 50)
    per_model_results = {}
    for mid, p in sorted(params.items(), key=lambda x: -x[1]["p_s"]):
        curve = [{"n": n, "delta": theoretical_delta(n, p["p_s"], p["p_p"])}
                 for n in ODD_JURY_SIZES]
        thresholds = {}
        for thr in DELTA_THRESHOLDS:
            found = next((e["n"] for e in curve if e["delta"] < thr), None)
            thresholds[str(thr)] = found
        per_model_results[mid] = thresholds
        min_n = thresholds["0.01"]
        print(f"{p['name']:<20} {p['p_s']:>6.3f} {p['p_p']:>6.3f} "
              f"{str(min_n) if min_n else '> 101':>10}")

    # --- comparison with empirical simulation (odd k only) ---
    if SIM_PATH.exists():
        sim = json.loads(SIM_PATH.read_text())
        sim_odd = [s for s in sim if s["k"] % 2 == 1 and s["k"] >= 3]
        if sim_odd:
            print("\nTheoretical vs empirical simulation (odd k, pooled):")
            print(f"{'k':>4}  {'theory Δ':>10}  {'empirical Δ':>12}  {'residual':>9}")
            print("-" * 43)
            for s in sim_odd:
                k = s["k"]
                theory = next((e["delta"] for e in odd_curve if e["n"] == k), None)
                if theory is not None:
                    residual = theory - s["delta"]
                    print(f"{k:>4}  {theory:>+10.4f}  {s['delta']:>+12.4f}  "
                          f"{residual:>+9.4f}")
            print(f"\n  Note: theory systematically overestimates the empirical delta.")
            print(f"  The IID peer assumption does not hold in a heterogeneous panel:")
            print(f"  peer vote distributions have higher variance than Binomial(n-1, p_p),")
            print(f"  which reduces the deadlock probability P(Bin(n-1,p_p)=(n-1)/2).")

    # --- full alternating curve (n=3..25) for documentation ---
    print("\nFull curve n=3..25 (showing odd/even alternation):")
    print(f"{'n':>4}  {'parity':>6}  {'Δ':>8}  {'sign':>5}")
    print("-" * 35)
    for n in range(3, 26):
        d = theoretical_delta(n, pool_p_s, pool_p_p)
        parity = "odd" if n % 2 == 1 else "even"
        sign = "+" if d >= 0 else "-"
        print(f"{n:>4}  {parity:>6}  {d:>+8.4f}  {sign:>5}")

    # --- save ---
    full_odd_curve = [
        {"n": n,
         "delta": round(theoretical_delta(n, pool_p_s, pool_p_p), 6),
         "fer_inc": round(fer_inclusive(n, pool_p_s, pool_p_p), 6),
         "fer_hol": round(fer_holdout(n, pool_p_p), 6)}
        for n in ODD_JURY_SIZES
    ]
    output = {
        "params": {mid: {k: v for k, v in p.items()} for mid, p in params.items()},
        "pooled": {"p_s": pool_p_s, "p_p": pool_p_p},
        "delta_thresholds": {
            str(thr): next((e["n"] for e in full_odd_curve if e["delta"] < thr), None)
            for thr in DELTA_THRESHOLDS
        },
        "pooled_odd_curve": full_odd_curve,
        "per_model_thresholds": per_model_results,
    }
    OUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nSaved to results/theoretical_jury_size.json")


if __name__ == "__main__":
    main()
