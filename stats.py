"""The few statistics the track needs, with no dependency beyond the standard library."""

import math


def mean_sd(xs):
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else float("nan")
    return m, sd


def _betacf(a, b, x):
    # continued fraction for the regularized incomplete beta function (Numerical Recipes, section 6.4)
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)), -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1 + aa * d
            d = 1 / (d if abs(d) > 1e-300 else 1e-300)
            c = 1 + aa / c
            c = c if abs(c) > 1e-300 else 1e-300
            h *= d * c
        if abs(d * c - 1) < 1e-14:
            break
    return h


def betainc(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x) / a
    return 1 - front * _betacf(b, a, 1 - x) / b


def t_cdf(t, df):
    tail = 0.5 * betainc(df / 2, 0.5, df / (df + t * t))
    return 1 - tail if t >= 0 else tail


def t_ppf(p, df):
    lo, hi = -1e3, 1e3
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if t_cdf(mid, df) < p else (lo, mid)
    return (lo + hi) / 2


def ci95(xs):
    m, sd = mean_sd(xs)
    half = t_ppf(0.975, len(xs) - 1) * sd / math.sqrt(len(xs))
    return m, half


def welch(a, b):
    """Is mean(a) below mean(b)? Returns the difference, its 95% CI half-width, and the one-sided p."""
    ma, sa = mean_sd(a)
    mb, sb = mean_sd(b)
    va, vb = sa * sa / len(a), sb * sb / len(b)
    se = math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va * va / (len(a) - 1) + vb * vb / (len(b) - 1))
    diff = ma - mb
    return {"diff": diff, "ci95": t_ppf(0.975, df) * se, "p_one_sided": t_cdf(diff / se, df), "df": df}


def tost(a, b, margin):
    """Two one-sided tests: are the means of a and b within +-margin of each other? Returns the larger p."""
    ma, sa = mean_sd(a)
    mb, sb = mean_sd(b)
    va, vb = sa * sa / len(a), sb * sb / len(b)
    se = math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va * va / (len(a) - 1) + vb * vb / (len(b) - 1))
    d = ma - mb
    p_low = 1 - t_cdf((d + margin) / se, df)
    p_high = t_cdf((d - margin) / se, df)
    return {"diff": d, "p": max(p_low, p_high), "ci90": t_ppf(0.95, df) * se}


def wilson(k, n, z=1.959964):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half
