# Strategy

## The problem
A trophy's earn rate is `earned / players_tracked`. Both sides move over
launch week: players keep arriving (diluting every rate) while tracked
players keep progressing (lifting earned counts). Extrapolating the ratio
itself confuses the two.

## The model (`main.py`, Python 3.13 stdlib only)
- Fit `log players_tracked` and `log (earned_i + 0.5)` separately against
  `x(t) = ln(1 + hours_since_launch / 24)`.
- Each fit is recency-weighted least squares over the case's **distinct**
  reads (consecutive identical reads are one stale page) and over a
  **recent window only** (last ~0.5 in x): launch week decelerates, so a
  fit over the whole history averages in the launch spike and overshoots.
  A straight line over all of history, and a concave quadratic
  extrapolated past the history, both failed on synthetic finals; the
  recent window's slope is the current growth rate.
- Slopes are clamped to >= 0 and each trophy's slope is shrunk toward its
  tier's median slope (rare trophies have counts of a handful of players).
- Median forecast: `100 * earned_i(target) / N(target)`.
- Spread: the fits' prediction error at the target, mapped to the logit
  scale, plus a random-walk term in horizon days plus a floor for a
  refresh the history has not seen. Near-saturated trophies (>= 95%) get
  a narrow spread; with no trend signal (the practice cases: one distinct
  read) the entry holds the last rate with a wide spread.

## Validation (synthetic finals, /tmp/synth_test.py)
Three simulated finals (different growth rates, player influx, refresh
cadences), scored by the contest's pinball loss on the logit scale:

| scenario | this entry | house curve | house baseline |
|----------|-----------:|------------:|---------------:|
| A        | 0.149      | 0.223       | 0.288          |
| B        | 0.149      | 0.224       | 0.232          |
| C        | 0.155      | 0.220       | 0.477          |

Practice cases p00/p01: valid output, 55 rows, deterministic across runs,
~0.13 s per run (limit 8 s).
