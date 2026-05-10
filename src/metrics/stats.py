"""Wilcoxon Signed-Rank test and effect size for inclusive vs holdout bias scores."""
from __future__ import annotations
from dataclasses import dataclass

from .bias import BiasScore


@dataclass
class StatResult:
    statistic: float
    p_value: float
    effect_size: float   # rank-biserial correlation
    n: int
    significant: bool    # p < 0.05


def wilcoxon_test(inclusive: list[BiasScore], holdout: list[BiasScore]) -> StatResult:
    from scipy.stats import wilcoxon
    import numpy as np

    by_model_inc = {s.model_id: s.false_endorsement_rate for s in inclusive}
    by_model_hol = {s.model_id: s.false_endorsement_rate for s in holdout}
    common = sorted(set(by_model_inc) & set(by_model_hol))

    d = np.array([by_model_inc[m] - by_model_hol[m] for m in common])
    stat, p = wilcoxon(d, alternative="greater")  # H1: inclusive bias > holdout bias

    # rank-biserial correlation as effect size.
    # scipy's wilcoxon (zero_method='wilcox') drops zero-difference pairs before ranking,
    # so the effective n for the effect size formula is the count of non-zero pairs.
    n = len(d)
    n_eff = int(np.sum(d != 0))
    r = 1 - (2 * stat) / (n_eff * (n_eff + 1) / 2) if n_eff > 0 else float("nan")

    return StatResult(float(stat), float(p), float(r), int(n), bool(p < 0.05))
