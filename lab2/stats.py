from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence


def cohen_d(x: Sequence[float], y: Sequence[float]) -> float:
    """
    Cohen's d for two independent samples.
    """

    if len(x) < 2 or len(y) < 2:
        raise ValueError("Need at least 2 samples per group for Cohen's d.")

    mx = sum(x) / len(x)
    my = sum(y) / len(y)

    vx = sum((v - mx) ** 2 for v in x) / (len(x) - 1)
    vy = sum((v - my) ** 2 for v in y) / (len(y) - 1)

    pooled = math.sqrt(((len(x) - 1) * vx + (len(y) - 1) * vy) / (len(x) + len(y) - 2))
    if pooled == 0:
        return 0.0
    return (mx - my) / pooled


def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    """
    Cliff's delta (stochastic dominance), in [-1, 1].

    Efficient implementation using sorting + binary search:
    delta = (#{x>y} - #{x<y}) / (n_x * n_y)
    """

    if not x or not y:
        raise ValueError("Empty input.")

    xs = sorted(x)
    ys = sorted(y)

    import bisect

    n_x = len(xs)
    n_y = len(ys)
    greater = 0
    less = 0
    for v in xs:
        less += bisect.bisect_left(ys, v)
        greater += n_y - bisect.bisect_right(ys, v)

    # `less` counts y-values smaller than x, i.e. x > y.
    return (less - greater) / float(n_x * n_y)


@dataclass(frozen=True)
class BootstrapCI:
    estimate: float
    low: float
    high: float


def bootstrap_ci(
    data: Sequence[float],
    *,
    statistic: Callable[[Sequence[float]], float] = lambda a: sum(a) / len(a),
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> BootstrapCI:
    if not data:
        raise ValueError("Empty data.")
    if n_boot <= 0:
        raise ValueError("n_boot must be positive.")

    rng = random.Random(seed)
    n = len(data)
    est = statistic(data)
    samples = [statistic([data[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot)]
    samples.sort()
    alpha = (1.0 - ci) / 2.0
    lo = samples[int(math.floor(alpha * n_boot))]
    hi = samples[int(math.floor((1 - alpha) * n_boot)) - 1]
    return BootstrapCI(estimate=est, low=lo, high=hi)


def bootstrap_ci_2samp(
    x: Sequence[float],
    y: Sequence[float],
    *,
    statistic: Callable[[Sequence[float], Sequence[float]], float],
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> BootstrapCI:
    if not x or not y:
        raise ValueError("Empty input.")
    rng = random.Random(seed)
    nx = len(x)
    ny = len(y)
    est = statistic(x, y)
    samples = [
        statistic([x[rng.randrange(nx)] for _ in range(nx)], [y[rng.randrange(ny)] for _ in range(ny)])
        for _ in range(n_boot)
    ]
    samples.sort()
    alpha = (1.0 - ci) / 2.0
    lo = samples[int(math.floor(alpha * n_boot))]
    hi = samples[int(math.floor((1 - alpha) * n_boot)) - 1]
    return BootstrapCI(estimate=est, low=lo, high=hi)


def mann_whitney_u(x: Sequence[float], y: Sequence[float]) -> float:
    """
    Two-sided Mann–Whitney U test p-value (non-parametric).
    """

    from scipy.stats import mannwhitneyu

    res = mannwhitneyu(x, y, alternative="two-sided")
    return float(res.pvalue)
