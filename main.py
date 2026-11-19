"""GTA VI trophy rarity forecaster for make.fun's gta6-trophy-rarity-forecast-01.

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, five
quantiles of its Exophase earn rate at the case's target, from the house's
reads of the trophy page up to the case's checkpoint.

Model (standard library only, deterministic, no network):
  A trophy's earn rate is a ratio: earned_i(t) / players_tracked(t).  Both
  sides move over launch week -- players keep arriving (which dilutes every
  rate) while the players already tracked keep progressing (which lifts the
  earned counts) -- so the entry models the two sides separately instead of
  extrapolating the ratio itself:
    * log players_tracked(t) and log (earned_i(t) + 0.5), earned_i = rate_i
      * N / 100, are each fit against x(t) = ln(1 + hours_since_launch /
      24), hours since 2026-11-19T00:00Z.
  Each fit is a recency-weighted least squares over the case's distinct
  reads (consecutive identical reads are one stale page, not new data, so
  they are de-duplicated first), taken over a RECENT window only: launch
  week decelerates -- the player influx spikes at launch and fades, a
  trophy's earners saturate -- so a fit over the whole history averages in
  the launch spike and overshoots the target, while the recent window's
  slope is the current growth rate.  Slopes are clamped to >= 0 (counts
  never fall) and each trophy's slope is shrunk toward its tier's median
  slope, which steadies the rare trophies whose counts are a handful of
  players.  The median forecast is 100 * earned_i(target) / N(target).
  The spread is the fits' prediction error at the target (slope uncertainty
  grows with how far the target sits past the history) plus a random-walk
  term in horizon days plus a floor for a refresh the history has not seen.

Run as the grader runs it: `python3 -P -s /bot/main.py` (Python 3.13,
standard library only).  Outside the grader, MAKEFUN_INPUT_DIR and
MAKEFUN_OUTPUT_DIR stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import (none is shipped; this is one file).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level (exact to 16 digits).
Z = {
    0.05: -1.6448536269514722,
    0.25: -0.6744897501960817,
    0.5: 0.0,
    0.75: 0.6744897501960817,
    0.95: 1.6448536269514722,
}

# Launch reference: the game released 2026-11-19.  x(t) is log-time since
# then, so growth is diminishing in clock time and the target sits a short
# way past the history in x, not a long way.
LAUNCH = datetime(2026, 11, 19, tzinfo=timezone.utc)

# Recency half-life for the fits: launch-night reads are a different regime
# from the reads near the checkpoint, so they count for less.
HALF_LIFE_HOURS = 24.0
# The fit window, in x units back from the last read: x is log-time, so
# 0.5 back from the checkpoint covers roughly the last day and a half of
# a two-day history, and the last few hours of a short one.
WINDOW_X = 0.5
# Minimum distinct reads in the window; widen it until there are this many.
MIN_WINDOW = 4
# Shrinkage strength for a trophy's slope toward its tier's median slope:
# a trophy with n distinct reads gets weight n / (n + SHRINK_K) on its own.
SHRINK_K = 3.0
# Slope caps: earned counts and tracked players never fall, and no trophy's
# earned count grows faster than this in log-time.
MAX_SLOPE_EARNED = 3.0
MAX_SLOPE_PLAYERS = 2.0
# Spread pieces, in logit units.
RW_SIGMA = 0.30          # random-walk spread per sqrt(day) of horizon
FLOOR_SIGMA = 0.30       # regime change / refresh the history has not seen
FALLBACK_EXTRA = 0.5     # extra spread when the history carries no trend
MAX_SPREAD = 2.5         # never wider than this
SATURATED_SPREAD = 0.25  # a trophy at >= 95% is pinned; keep it narrow


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_launch(t: datetime) -> float:
    return max(0.0, (t - LAUNCH).total_seconds() / 3600.0)


def x_of(t: datetime) -> float:
    return math.log1p(hours_since_launch(t) / 24.0)


def clamp_pct(p: float) -> float:
    return min(99.99, max(0.01, p))


def logit(pct: float) -> float:
    c = clamp_pct(pct)
    return math.log(c / (100.0 - c))


def inv_logit(l: float) -> float:
    if l >= 0:
        return 100.0 / (1.0 + math.exp(-l))
    e = math.exp(l)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """x rounded to 4 decimals, half up, as JavaScript's
    Math.round(x * 10000) / 10000 rounds it.  A whole number comes back as
    an int, so json.dumps writes 100 where JavaScript writes 100."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's approximation, ~1e-9).  The
    five levels the contest uses are looked up exactly; anything else goes
    through this."""
    if p in Z:
        return Z[p]
    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"level {p} outside (0, 1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285984469534e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def wls_linear(xs, ys, ws):
    """Weighted least squares of y on [1, x].  Returns
    (intercept, slope, resid_std, sxx, xbar, n)."""
    n = len(xs)
    sw = sum(ws)
    xbar = sum(w * x for w, x in zip(ws, xs)) / sw
    ybar = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - xbar) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        var = sum(w * (y - ybar) ** 2 for w, y in zip(ws, ys)) / sw
        return ybar, 0.0, math.sqrt(max(var, 0.0)), 0.0, xbar, n
    sxy = sum(w * (x - xbar) * (y - ybar) for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx
    intercept = ybar - slope * xbar
    dof = max(n - 2, 1)
    var = sum(w * (y - (intercept + slope * x)) ** 2 for w, x, y in zip(ws, xs, ys)) / sw
    resid = math.sqrt(max(var, 0.0) * n / dof) if n > 2 else math.sqrt(max(var, 0.0))
    return intercept, slope, resid, sxx, xbar, n


def fit_curve(xs, ys, ws):
    """Fit y against x over the recent window only, linear, recency
    weighted.  Returns a dict with the fitted value at the largest x
    (y_max), the recent slope, the residual scale, sxx, xbar and n.  The
    window is the reads within WINDOW_X of the last x, widened until it
    holds MIN_WINDOW reads; with too few reads overall it is all of them.
    With no spread in x the slope is 0 and the value is the mean."""
    xmax = max(xs)
    idx = [k for k, x in enumerate(xs) if x >= xmax - WINDOW_X]
    if len(idx) < MIN_WINDOW:
        idx = list(range(max(0, len(xs) - MIN_WINDOW), len(xs)))
    wx = [xs[k] for k in idx]
    wy = [ys[k] for k in idx]
    ww = [ws[k] for k in idx]
    intercept, slope, resid, sxx, xbar, n = wls_linear(wx, wy, ww)
    return {"y_max": intercept + slope * xmax, "slope": slope,
            "resid": resid, "sxx": sxx,
            "xbar": xbar, "n": n, "xmax": xmax}


def predict_from(fit, x_star, slope=None):
    """Extend a fit from the end of the history to x_star: the fitted value
    at xmax plus the recent slope there (optionally replaced) times the
    gap.  The recent slope is the current growth rate; extending it is the
    honest extrapolation, and it never replays the launch spike."""
    delta = x_star - fit["xmax"]
    if delta <= 0:
        return fit["y_max"]
    s = fit["slope"] if slope is None else slope
    return fit["y_max"] + s * delta


def median(xs):
    s = sorted(xs)
    m = len(s)
    if m == 0:
        return 0.0
    if m % 2:
        return s[m // 2]
    return 0.5 * (s[m // 2 - 1] + s[m // 2])


def distinct_reads(history):
    """Consecutive reads that print the same page are one stale page, not
    new data: keep a read only when its players-tracked count or any rate
    differs from the previous kept read."""
    out = []
    prev = None
    for h in history:
        sig = (h.get("players_tracked"), tuple(h.get("rates") or ()))
        if sig != prev:
            out.append(h)
            prev = sig
    return out


def forecast_case(case: dict, name: str) -> dict:
    levels = case["levels"]
    zvals = [norm_ppf(float(lv)) for lv in levels]
    trophies = case["trophies"]
    history = case.get("history") or []
    if not history:
        # No history at all: say 1% everywhere, very wide.  Never crash.
        rows = [{"id": t["id"], "q": [round4(inv_logit(logit(1.0) + z * 2.0)) for z in zvals]}
                for t in trophies]
        return {"trophies": rows}

    target = parse_time(case["target"])
    x_star = x_of(target)
    horizon_days = max(case["horizon_minutes"], 0) / 1440.0

    reads = distinct_reads(history)
    # Keep one aligned list of usable reads: a read counts only when it
    # carries a players-tracked count, so the per-trophy loops below index
    # the same rows as the players fit does.
    valid = [h for h in reads if h.get("players_tracked") and h["players_tracked"] > 0]
    if not valid:
        rows = [{"id": t["id"], "q": [round4(inv_logit(logit(1.0) + z * 2.0)) for z in zvals]}
                for t in trophies]
        return {"trophies": rows}
    t_last = parse_time(valid[-1]["at"])
    xs, wts, log_n = [], [], []
    for h in valid:
        t = parse_time(h["at"])
        xs.append(x_of(t))
        age_h = max((t_last - t).total_seconds() / 3600.0, 0.0)
        wts.append(0.5 ** (age_h / HALF_LIFE_HOURS))
        log_n.append(math.log(float(h["players_tracked"])))

    # --- players-tracked fit -------------------------------------------
    if len(xs) >= 3:
        fit_n = fit_curve(xs, log_n, wts)
        fit_n["slope"] = min(max(fit_n["slope"], 0.0), MAX_SLOPE_PLAYERS)
        log_n_star = predict_from(fit_n, x_star)
        var_n_star = (fit_n["resid"] ** 2) * (
            1.0 + 1.0 / len(xs) + (x_star - fit_n["xbar"]) ** 2 / fit_n["sxx"]) \
            if fit_n["sxx"] > 1e-12 else (fit_n["resid"] ** 2) * 2.0
    else:
        log_n_star = log_n[-1]
        fit_n = {"resid": 0.2, "sxx": 0.0, "xbar": xs[-1] if xs else 0.0}
        var_n_star = 0.04
    n_star = math.exp(log_n_star)

    # --- per-trophy earned-count fits ----------------------------------
    # First pass: raw slopes, to take each tier's median for shrinkage.
    raw = []
    for i, trophy in enumerate(trophies):
        ex, ey, ew = [], [], []
        for k, h in enumerate(valid):
            rates = h.get("rates") or []
            if i >= len(rates) or rates[i] is None:
                continue
            n = h.get("players_tracked")
            if not n or n <= 0:
                continue
            earned = float(rates[i]) / 100.0 * float(n)
            ex.append(xs[k])
            ey.append(math.log(earned + 0.5))
            ew.append(wts[k])
        raw.append((ex, ey, ew))
    tier_slopes = {}
    for tier in {t["tier"] for t in trophies}:
        slopes = []
        for i, t in enumerate(trophies):
            if t["tier"] != tier:
                continue
            ex, ey, ew = raw[i]
            if len(ex) >= 3:
                f = fit_curve(ex, ey, ew)
                slopes.append(min(max(f["slope"], 0.0), MAX_SLOPE_EARNED))
        tier_slopes[tier] = median(slopes) if slopes else 0.0

    rows = []
    for i, trophy in enumerate(trophies):
        ex, ey, ew = raw[i]
        last_rate = None
        for h in reversed(valid):
            rates = h.get("rates") or []
            if i < len(rates) and rates[i] is not None:
                last_rate = float(rates[i])
                break
        if last_rate is None:
            last_rate = 0.01

        if len(ex) >= 3:
            fit_e = fit_curve(ex, ey, ew)
            slope_e = min(max(fit_e["slope"], 0.0), MAX_SLOPE_EARNED)
            # Shrink toward the tier median: a trophy with few distinct
            # reads borrows its tier's slope; a well-observed one keeps its.
            w_self = fit_e["n"] / (fit_e["n"] + SHRINK_K)
            slope_e = w_self * slope_e + (1.0 - w_self) * tier_slopes.get(trophy["tier"], 0.0)
            log_e_star = predict_from(fit_e, x_star, slope=slope_e)
            var_e_star = (fit_e["resid"] ** 2) * (
                1.0 + 1.0 / fit_e["n"] + (x_star - fit_e["xbar"]) ** 2 / fit_e["sxx"]) \
                if fit_e["sxx"] > 1e-12 else (fit_e["resid"] ** 2) * 2.0
            earned_star = max(math.exp(log_e_star) - 0.5, 0.0)
            med = clamp_pct(100.0 * earned_star / n_star)
            # Logit-scale spread: the two fits' prediction errors, mapped
            # through d logit / d ln rate = 1 / (1 - p), plus a random-walk
            # term in horizon days and a floor for an unseen refresh.
            p = med / 100.0
            jac = 1.0 / max(1.0 - p, 0.25)
            var_logit = (var_e_star + var_n_star) * jac * jac
            var_logit += (RW_SIGMA ** 2) * horizon_days + FLOOR_SIGMA ** 2
            spread = math.sqrt(max(var_logit, 0.0))
        else:
            # No trend signal (the practice cases: one distinct read):
            # hold the last rate, with a spread wider than the house's to
            # admit a refresh the history has not seen.
            med = clamp_pct(last_rate)
            spread = 0.5 * math.sqrt(horizon_days) + FALLBACK_EXTRA

        spread = min(spread, MAX_SPREAD)
        if med >= 95.0:
            spread = min(spread, SATURATED_SPREAD)
        lg = logit(med)
        rows.append({"id": trophy["id"],
                     "q": [round4(inv_logit(lg + z * spread)) for z in zvals]})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
