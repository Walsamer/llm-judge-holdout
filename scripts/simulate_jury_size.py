#!/usr/bin/env python3
"""Jury-size simulation: derive holdout effect at k=2..7 from existing votes.json.

For each jury size k, the inclusive jury always contains the self-judge plus k-1
randomly drawn peers (sampled without replacement). The holdout jury contains only
those k-1 peers. This guarantees the self-judge is always present in the inclusive
condition, making the comparison with the theoretical model clean.

Majority rule: same as bias.py — endorses model M if n_for > n_against,
where individual 'tie' votes abstain from the A/B count, and equal counts = no endorsement.
"""
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

VOTES_PATH = Path(__file__).parents[1] / "results" / "votes.json"
N_ITER = 1000
JURY_SIZES = [2, 3, 4, 5, 6, 7]
RANDOM_SEED = 42


@dataclass
class SimResult:
    k: int
    inclusive_fer: float
    holdout_fer: float
    delta: float
    ci_low: float
    ci_high: float


def load_votes(path: Path):
    return json.loads(path.read_text())


def _jury_endorses(prefs: list[str], side: str) -> bool:
    """True if jury endorses model on the given side.

    Matches bias.py: n_for > n_against (tie votes abstain).
    Equal counts = no endorsement.
    """
    n_for = sum(1 for p in prefs if p == side)
    n_against = sum(1 for p in prefs if p != side and p != "tie")
    return n_for > n_against


def _build_cases(matchups: list) -> list[dict]:
    """Pre-compute all (rejected model, available peers, vote lookup) triples."""
    cases = []
    for m in matchups:
        if m["human_winner"] == "tie":
            continue
        model_a, model_b = m["model_a"], m["model_b"]

        # votes indexed by judge, non-swapped only
        vote_map: dict[str, str] = {}
        for v in m["votes"]:
            if not v["position_swapped"]:
                vote_map[v["judge_model_id"]] = v["preference"]

        all_judges = list(vote_map.keys())

        for model_id, side in [(model_a, "A"), (model_b, "B")]:
            is_rejected = (
                (side == "A" and m["human_winner"] == "model_b") or
                (side == "B" and m["human_winner"] == "model_a")
            )
            if not is_rejected:
                continue
            if model_id not in vote_map:
                # self-judge vote missing for this matchup — skip
                continue

            peers = [j for j in all_judges if j != model_id]
            cases.append({
                "model_id": model_id,
                "side": side,
                "self_pref": vote_map[model_id],
                "peer_vote_map": {j: vote_map[j] for j in peers},
                "peers": peers,
            })
    return cases


def simulate(cases: list[dict], k: int, n_iter: int, rng: random.Random) -> SimResult:
    inc_samples = []
    hol_samples = []
    delta_samples = []

    for _ in range(n_iter):
        inc_fe = 0
        hol_fe = 0
        inc_total = 0
        hol_total = 0

        for case in cases:
            side = case["side"]
            peers = case["peers"]
            peer_map = case["peer_vote_map"]
            self_pref = case["self_pref"]

            # sample k-1 peers without replacement (or all peers if fewer available)
            n_draw = min(k - 1, len(peers))
            sampled_peers = rng.sample(peers, n_draw)

            # inclusive: self-judge + k-1 peers
            inc_prefs = [self_pref] + [peer_map[j] for j in sampled_peers]
            inc_total += 1
            if _jury_endorses(inc_prefs, side):
                inc_fe += 1

            # holdout: k-1 peers only
            if sampled_peers:
                hol_prefs = [peer_map[j] for j in sampled_peers]
                hol_total += 1
                if _jury_endorses(hol_prefs, side):
                    hol_fe += 1

        inc_rate = inc_fe / inc_total if inc_total > 0 else float("nan")
        hol_rate = hol_fe / hol_total if hol_total > 0 else float("nan")
        inc_samples.append(inc_rate)
        hol_samples.append(hol_rate)
        if not (math.isnan(inc_rate) or math.isnan(hol_rate)):
            delta_samples.append(inc_rate - hol_rate)

    mean_inc = sum(inc_samples) / len(inc_samples)
    mean_hol = sum(hol_samples) / len(hol_samples)
    mean_delta = sum(delta_samples) / len(delta_samples) if delta_samples else float("nan")

    delta_samples.sort()
    n = len(delta_samples)
    # use numpy-style percentile: index at 0.025*(n-1) and 0.975*(n-1) with interpolation
    def percentile(arr, pct):
        idx = pct * (len(arr) - 1)
        lo, hi = int(idx), min(int(idx) + 1, len(arr) - 1)
        return arr[lo] + (arr[hi] - arr[lo]) * (idx - lo)

    ci_low = percentile(delta_samples, 0.025) if n > 1 else mean_delta
    ci_high = percentile(delta_samples, 0.975) if n > 1 else mean_delta
    degenerate = (ci_low == ci_high)

    return SimResult(k, mean_inc, mean_hol, mean_delta, ci_low, ci_high)


def main():
    matchups = load_votes(VOTES_PATH)
    print(f"Loaded {len(matchups)} matchups")

    cases = _build_cases(matchups)
    print(f"Evaluation cases (rejected models with self-judge present): {len(cases)}\n")

    rng = random.Random(RANDOM_SEED)
    results = []

    for k in JURY_SIZES:
        print(f"  k={k} ({N_ITER} iterations)...", end=" ", flush=True)
        r = simulate(cases, k, N_ITER, rng)
        results.append(r)
        ci_label = f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}]"
        print(f"done  inc={r.inclusive_fer:.4f}  hol={r.holdout_fer:.4f}  "
              f"Δ={r.delta:+.4f}  95%CI={ci_label}")

    print()
    print("=" * 80)
    print("JURY SIZE SIMULATION — Holdout effect as a function of jury size k")
    print("Inclusive jury: self-judge + k-1 random peers | Holdout: k-1 peers only")
    print("Δ = inclusive FER − holdout FER  (positive = holdout reduces false endorsement)")
    print("-" * 80)
    print(f"{'k':>3}  {'Inc FER':>8}  {'Hol FER':>8}  {'Δ':>8}  {'95% CI':>22}")
    for r in results:
        ci_label = f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}]"
        print(f"{r.k:>3}  {r.inclusive_fer:>8.4f}  {r.holdout_fer:>8.4f}  "
              f"{r.delta:>+8.4f}  {ci_label:>22}")

    out_path = Path(__file__).parents[1] / "results" / "jury_size_simulation.json"
    out_path.write_text(json.dumps([
        {**r.__dict__, "degenerate_ci": (r.ci_low == r.ci_high)}
        for r in results
    ], indent=2))
    print(f"\nSaved to results/jury_size_simulation.json")


if __name__ == "__main__":
    main()
