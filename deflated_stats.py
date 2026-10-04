"""
Best-of-N distribution of a performance statistic.

Question answered
-----------------
Take a strategy whose daily log returns follow a fixed distribution D. Run N
variants of it (all with the *same* true return distribution) over T business
days and keep the best value of statistic S. What is the distribution of that
best value? Any "improvement" you find by searching over N variants should be
judged against this distribution, not against the single-trial value.

Core objects
------------
- Marginals (``ParametricMarginal``, ``EmpiricalMarginal``) generate return paths.
  ``rho`` is the correlation between trials (variants of the same strategy are
  rarely independent).
- ``compute_stat`` computes a statistic over the last axis of a return array.
- ``simulate_best_of_n`` returns a ``BestOfN`` object with cdf, p-value,
  critical value, expected best, and a single-trial reference.

Conventions
-----------
- Inputs are daily log returns. Annualisation uses 252 business days.
- Return  = mean daily log return x 252 (annualised log return).
- Vol     = std of daily log returns x sqrt(252).
- Sharpe  = Return / Vol (risk-free assumed already netted out, or zero).
- Sortino = Return / (sqrt(mean(min(r, 0)^2)) x sqrt(252)).
- MaxDD   = largest peak-to-trough fall of the equity curve exp(cumsum r), as
            a positive fraction. The starting value counts as a peak.
- Calmar  = Return / MaxDD.
- PnL     = G x sum(exp(r) - 1): constant gross market value G, P&L not compounded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy import optimize, special
from scipy import stats as sps

TRADING_DAYS = 252
EULER_GAMMA = 0.5772156649015329

# Normal-score grid on which parametric quantile functions are tabulated.
# Mapping z -> F^{-1}(Phi(z)) by interpolation keeps sampling fast for every
# distribution and lets all of them share a Gaussian copula across trials.
_Z_GRID = np.linspace(-9.0, 9.0, 7201)


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StatSpec:
    key: str
    label: str
    higher_is_better: bool
    display_scale: float  # multiply raw value by this for display (100 for %)
    unit: str             # suffix shown after the displayed value


STATS: dict[str, StatSpec] = {
    "return": StatSpec("return", "Annualised return", True, 100.0, "%"),
    "vol": StatSpec("vol", "Annualised volatility", False, 100.0, "%"),
    "sharpe": StatSpec("sharpe", "Sharpe ratio", True, 1.0, ""),
    "sortino": StatSpec("sortino", "Sortino ratio", True, 1.0, ""),
    "calmar": StatSpec("calmar", "Calmar ratio", True, 1.0, ""),
    "maxdd": StatSpec("maxdd", "Max drawdown", False, 100.0, "%"),
    "pnl": StatSpec("pnl", "PnL (constant GMV)", True, 1.0, ""),
}


def max_drawdown(r: np.ndarray) -> np.ndarray:
    """Max drawdown (positive fraction) of exp(cumsum(r)) along the last axis."""
    log_eq = np.cumsum(r, axis=-1)
    peak = np.maximum(np.maximum.accumulate(log_eq, axis=-1), 0.0)
    return (1.0 - np.exp(log_eq - peak)).max(axis=-1)


def compute_stat(r: np.ndarray, stat: str, gmv: float = 1.0) -> np.ndarray:
    """Statistic over the last axis of an array of daily log returns."""
    ann = np.sqrt(TRADING_DAYS)
    mean = r.mean(axis=-1)
    if stat == "return":
        return mean * TRADING_DAYS
    if stat == "vol":
        return r.std(axis=-1, ddof=1) * ann
    if stat == "sharpe":
        return mean / r.std(axis=-1, ddof=1) * ann
    if stat == "sortino":
        downside = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2, axis=-1))
        with np.errstate(divide="ignore", invalid="ignore"):
            return mean / downside * ann
    if stat == "maxdd":
        return max_drawdown(r)
    if stat == "calmar":
        mdd = max_drawdown(r)
        with np.errstate(divide="ignore", invalid="ignore"):
            return mean * TRADING_DAYS / mdd
    if stat == "pnl":
        return gmv * np.expm1(r).sum(axis=-1)
    raise ValueError(f"Unknown statistic: {stat}")


# --------------------------------------------------------------------------- #
# Marginal return distributions
# --------------------------------------------------------------------------- #
def _standardised_quantile_table(dist) -> np.ndarray:
    """x(z) = F^{-1}(Phi(z)) for a frozen scipy distribution, scaled to mean 0, var 1.

    The upper half uses isf on the mirrored tail to keep precision where
    Phi(z) rounds to 1.
    """
    z = _Z_GRID
    x = np.empty_like(z)
    lo = z <= 0
    x[lo] = dist.ppf(special.ndtr(z[lo]))
    x[~lo] = dist.isf(special.ndtr(-z[~lo]))
    mean, var = dist.stats(moments="mv")
    return (x - float(mean)) / np.sqrt(float(var))


def skewnorm_alpha_for_skew(skew: float) -> float:
    """Skew-normal shape alpha with the requested skewness (|skew| < 0.995)."""
    g = float(np.clip(skew, -0.99, 0.99))
    if g == 0:
        return 0.0
    b = (2 * abs(g) / (4 - np.pi)) ** (2 / 3)
    delta = np.sign(g) * np.sqrt(np.pi / 2 * b / (1 + b))
    return float(delta / np.sqrt(1 - delta**2))


def fit_johnson_su(skew: float, excess_kurt: float) -> tuple[float, float, float, float]:
    """Johnson SU (a, b) matching skewness and excess kurtosis.

    Returns (a, b, achieved_skew, achieved_excess_kurt). If the target sits
    outside the Johnson SU region (it needs more kurtosis than a lognormal with
    the same skew), the nearest feasible point is returned.
    """
    target = np.array([skew, excess_kurt])

    def resid(p):
        a, log_b = p
        s, k = sps.johnsonsu(a, np.exp(log_b)).stats(moments="sk")
        return np.array([float(s), float(k)]) - target

    best = None
    for a0 in (0.0, -np.sign(skew) * 0.5, -np.sign(skew) * 2.0):
        for lb0 in (np.log(1.0), np.log(2.0), np.log(4.0)):
            try:
                sol = optimize.least_squares(
                    resid, x0=[a0, lb0], bounds=([-20, np.log(0.3)], [20, np.log(50)])
                )
            except Exception:
                continue
            if best is None or sol.cost < best.cost:
                best = sol
    a, log_b = best.x
    s, k = sps.johnsonsu(a, np.exp(log_b)).stats(moments="sk")
    return float(a), float(np.exp(log_b)), float(s), float(k)


@dataclass
class ParametricMarginal:
    """Daily log returns = daily_mean + daily_vol * X, X standardised (mean 0, var 1)."""

    name: str
    daily_mean: float
    daily_vol: float
    table: Optional[np.ndarray] = None  # None means Gaussian (identity map)
    info: dict = field(default_factory=dict)

    def from_normal_scores(self, z: np.ndarray) -> np.ndarray:
        x = z if self.table is None else np.interp(z, _Z_GRID, self.table)
        return self.daily_mean + self.daily_vol * x

    # Path generation with a Gaussian copula across trials.
    def common_state(self, rng, m: int, T: int):
        return rng.standard_normal((m, T))

    def paths(self, rng, common, b: int, T: int, rho: float) -> np.ndarray:
        m = common.shape[0]
        if rho == 0:
            z = rng.standard_normal((m, b, T))
        else:
            z = np.sqrt(rho) * common[:, None, :] + np.sqrt(1 - rho) * rng.standard_normal((m, b, T))
        return self.from_normal_scores(z)


def make_parametric(
    kind: str,
    ann_return: float,
    ann_vol: float,
    dof: float = 5.0,
    skew: float = 0.0,
    excess_kurt: float = 3.0,
) -> ParametricMarginal:
    """Build a marginal from annualised log-return mean and vol plus shape parameters.

    kind: "gaussian" | "student_t" | "skew_normal" | "johnson_su".
    """
    mu_d = ann_return / TRADING_DAYS
    sd_d = ann_vol / np.sqrt(TRADING_DAYS)
    if kind == "gaussian":
        return ParametricMarginal("Gaussian", mu_d, sd_d, None, {"skew": 0.0, "excess_kurt": 0.0})
    if kind == "student_t":
        if dof <= 2:
            raise ValueError("Student-t needs dof > 2 for a finite variance.")
        ek = 6 / (dof - 4) if dof > 4 else float("inf")
        return ParametricMarginal(
            f"Student-t (dof={dof:g})", mu_d, sd_d,
            _standardised_quantile_table(sps.t(dof)), {"skew": 0.0, "excess_kurt": ek},
        )
    if kind == "skew_normal":
        alpha = skewnorm_alpha_for_skew(skew)
        dist = sps.skewnorm(alpha)
        s, k = dist.stats(moments="sk")
        return ParametricMarginal(
            "Skew-normal", mu_d, sd_d, _standardised_quantile_table(dist),
            {"skew": float(s), "excess_kurt": float(k), "alpha": alpha},
        )
    if kind == "johnson_su":
        a, b, s, k = fit_johnson_su(skew, excess_kurt)
        return ParametricMarginal(
            "Johnson SU", mu_d, sd_d, _standardised_quantile_table(sps.johnsonsu(a, b)),
            {"skew": s, "excess_kurt": k, "a": a, "b": b},
        )
    raise ValueError(f"Unknown distribution kind: {kind}")


@dataclass
class EmpiricalMarginal:
    """Bootstrap from an observed series of daily log returns.

    block = 1: i.i.d. resampling. Trials are correlated through a Gaussian
               copula on the resampling ranks.
    block > 1: circular block bootstrap, which keeps autocorrelation and
               volatility clustering up to the block length. Trials are
               correlated by copying each block from a common path with
               probability sqrt(rho), giving pairwise correlation rho.
    """

    returns: np.ndarray
    block: int = 1

    def __post_init__(self):
        r = np.asarray(self.returns, dtype=float).ravel()
        self.returns = r[np.isfinite(r)]
        if self.returns.size < 20:
            raise ValueError("Need at least 20 finite returns.")
        self._sorted = np.sort(self.returns)
        self.block = max(1, int(self.block))

    @property
    def n(self) -> int:
        return self.returns.size

    def common_state(self, rng, m: int, T: int):
        if self.block == 1:
            return rng.standard_normal((m, T))
        nb = -(-T // self.block)
        return rng.integers(0, self.n, size=(m, nb))

    def paths(self, rng, common, b: int, T: int, rho: float) -> np.ndarray:
        m = common.shape[0]
        if self.block == 1:
            if rho == 0:
                z = rng.standard_normal((m, b, T))
            else:
                z = np.sqrt(rho) * common[:, None, :] + np.sqrt(1 - rho) * rng.standard_normal((m, b, T))
            idx = np.minimum((special.ndtr(z) * self.n).astype(np.int64), self.n - 1)
            return self._sorted[idx]
        nb = common.shape[1]
        starts = rng.integers(0, self.n, size=(m, b, nb))
        if rho > 0:
            # Each trial copies the common block with prob sqrt(rho), so two trials
            # share a block with prob rho: pairwise return correlation = rho.
            share = rng.random((m, b, nb)) < np.sqrt(rho)
            starts = np.where(share, common[:, None, :], starts)
        idx = (starts[..., None] + np.arange(self.block)) % self.n
        return self.returns[idx.reshape(m, b, nb * self.block)[..., :T]]

    def summary(self) -> dict:
        r = self.returns
        vol = r.std(ddof=1) * np.sqrt(TRADING_DAYS)
        ret = r.mean() * TRADING_DAYS
        return {
            "n_obs": self.n,
            "ann_return": ret,
            "ann_vol": vol,
            "sharpe": ret / vol if vol > 0 else np.nan,
            "skew": float(sps.skew(r)),
            "excess_kurt": float(sps.kurtosis(r)),
            "max_dd": float(max_drawdown(r)),
        }


# --------------------------------------------------------------------------- #
# Best-of-N distribution
# --------------------------------------------------------------------------- #
@dataclass
class BestOfN:
    """Distribution of the best of N trials of a statistic.

    Internally works with y = sign * x, so that "best" is always the maximum of
    y (sign = -1 for statistics where lower is better).

    The CDF of max(y) is (ECDF(ys))**power:
      - independent trials (rho = 0): ys are single-trial samples, power = N.
        This is exact for i.i.d. trials and uses every simulated trial.
      - correlated trials: ys are simulated maxima, power = 1.
    """

    stat: str
    sign: float
    n_trials: int
    ys: np.ndarray               # sorted samples (see class docstring)
    power: int
    single_ys: np.ndarray        # sorted single-trial samples, for reference
    n_paths_simulated: int

    # ---- y-space helpers ----
    def _cdf_y(self, t, ys=None, power=None, side="right"):
        ys = self.ys if ys is None else ys
        power = self.power if power is None else power
        frac = np.searchsorted(ys, t, side=side) / ys.size
        return frac**power

    def _ppf_y(self, q, ys=None, power=None):
        ys = self.ys if ys is None else ys
        power = self.power if power is None else power
        return np.quantile(ys, np.clip(np.asarray(q, dtype=float), 0, 1) ** (1.0 / power))

    # ---- natural units ----
    def cdf(self, x):
        """P(best of N <= x)."""
        x = np.asarray(x, dtype=float)
        if self.sign > 0:
            return self._cdf_y(x)
        return 1.0 - self._cdf_y(-x, side="left")

    def single_cdf(self, x):
        """P(single trial <= x)."""
        x = np.asarray(x, dtype=float)
        if self.sign > 0:
            return self._cdf_y(x, self.single_ys, 1)
        return 1.0 - self._cdf_y(-x, self.single_ys, 1, side="left")

    def p_value(self, v: float) -> float:
        """P(best of N is at least as good as v): P(max >= v), or P(min <= v) if lower is better."""
        return float(1.0 - self._cdf_y(self.sign * v, side="left"))

    def critical_value(self, p: float) -> float:
        """Value the best of N only reaches (or betters) with probability p."""
        return float(self.sign * self._ppf_y(1.0 - p))

    def best_quantile(self, q: float) -> float:
        """q-quantile of the best-of-N value in natural units."""
        qy = q if self.sign > 0 else 1.0 - q
        return float(self.sign * self._ppf_y(qy))

    def single_quantile(self, q: float) -> float:
        qy = q if self.sign > 0 else 1.0 - q
        return float(self.sign * self._ppf_y(qy, self.single_ys, 1))

    def expected_best(self) -> float:
        k = self.ys.size
        i = np.arange(1, k + 1)
        w = np.exp(self.power * np.log(i / k)) - np.exp(self.power * np.log(np.maximum(i - 1, 1e-300) / k))
        w[0] = (1 / k) ** self.power
        return float(self.sign * np.sum(w * self.ys))

    def expected_single(self) -> float:
        return float(self.sign * self.single_ys.mean())

    def p_value_resolution(self) -> float:
        """Smallest non-zero p-value the simulation can resolve."""
        k = self.ys.size
        return float(1.0 - ((k - 1) / k) ** self.power)

    def tail_is_resolved(self, p: float) -> bool:
        """Whether the (1 - p) quantile is backed by at least ~10 samples."""
        q = (1.0 - p) ** (1.0 / self.power)
        return (1.0 - q) * self.ys.size >= 10


def simulate_best_of_n(
    marginal,
    T: int,
    N: int,
    stat: str,
    rho: float = 0.0,
    gmv: float = 1.0,
    draw_budget: float = 5e7,
    seed: int = 0,
    chunk_elems: int = 4_000_000,
    progress: Optional[Callable[[float], None]] = None,
) -> BestOfN:
    """Simulate the best-of-N distribution.

    draw_budget is the total number of simulated daily returns. With rho = 0,
    budget / T single trials feed the exact (ECDF)^N formula. With rho > 0,
    budget / (N * T) groups of N correlated trials are simulated (at least 200).
    """
    if T < 2 or N < 1:
        raise ValueError("Need T >= 2 and N >= 1.")
    rho = float(np.clip(rho, 0.0, 0.999))
    spec = STATS[stat]
    sign = 1.0 if spec.higher_is_better else -1.0
    rng = np.random.default_rng(seed)

    def stat_y(r):
        y = sign * compute_stat(r, stat, gmv)
        return np.where(np.isnan(y), -np.inf, y)  # undefined values count as worst

    if rho == 0.0 or N == 1:
        K = int(max(2_000, draw_budget // T))
        out = np.empty(K)
        step = max(1, chunk_elems // T)
        done = 0
        while done < K:
            m = min(step, K - done)
            common = marginal.common_state(rng, m, T)
            r = marginal.paths(rng, common, 1, T, 0.0)[:, 0, :]
            out[done:done + m] = stat_y(r)
            done += m
            if progress:
                progress(done / K)
        out.sort()
        return BestOfN(stat, sign, N, out, N, out, K)

    M = int(max(200, draw_budget // (N * T)))
    maxima = np.empty(M)
    singles = []
    single_cap = 400_000
    m_step = max(1, chunk_elems // (T * N))
    b_step = N if m_step > 1 or N * T <= chunk_elems else max(1, chunk_elems // T)
    done = 0
    while done < M:
        m = min(m_step, M - done)
        common = marginal.common_state(rng, m, T)
        best = np.full(m, -np.inf)
        t_done = 0
        while t_done < N:
            b = min(b_step, N - t_done)
            y = stat_y(marginal.paths(rng, common, b, T, rho))
            best = np.maximum(best, y.max(axis=1))
            if sum(s.size for s in singles) < single_cap:
                singles.append(y.ravel())
            t_done += b
        maxima[done:done + m] = best
        done += m
        if progress:
            progress(done / M)
    maxima.sort()
    single = np.sort(np.concatenate(singles))
    return BestOfN(stat, sign, N, maxima, 1, single, M * N)


# --------------------------------------------------------------------------- #
# Analytic reference (Bailey & Lopez de Prado, 2014)
# --------------------------------------------------------------------------- #
def expected_max_sharpe_gaussian(sharpe: float, T: int, N: int, skew: float = 0.0, excess_kurt: float = 0.0) -> float:
    """Approximate E[max annualised Sharpe] of N independent trials.

    Uses the false-strategy theorem's extreme-value approximation and the
    Mertens / Lo standard error of the Sharpe ratio. Only meaningful for N >= ~5.
    """
    sr_d = sharpe / np.sqrt(TRADING_DAYS)
    se_d = np.sqrt((1 - skew * sr_d + excess_kurt / 4 * sr_d**2 + 0.5 * sr_d**2) / T)
    if N == 1:
        return sharpe
    z = (1 - EULER_GAMMA) * sps.norm.ppf(1 - 1 / N) + EULER_GAMMA * sps.norm.ppf(1 - 1 / (N * np.e))
    return float(sharpe + se_d * np.sqrt(TRADING_DAYS) * z)


# --------------------------------------------------------------------------- #
# Loading returns from files
# --------------------------------------------------------------------------- #
def load_returns_table(file_obj, filename: str):
    """Load a CSV or pickle into a pandas DataFrame (Series and arrays are wrapped)."""
    import pandas as pd

    name = filename.lower()
    if name.endswith((".pkl", ".pickle")):
        obj = pd.read_pickle(file_obj)  # only load pickles you trust
    elif name.endswith(".parquet"):
        obj = pd.read_parquet(file_obj)
    else:
        obj = pd.read_csv(file_obj)
    if isinstance(obj, pd.Series):
        obj = obj.to_frame(name=obj.name or "returns")
    elif not isinstance(obj, pd.DataFrame):
        arr = np.asarray(obj, dtype=float)
        obj = pd.DataFrame(arr.reshape(arr.shape[0], -1))
    return obj


_DATE_NAMES = {"date", "dates", "datetime", "time", "timestamp", "day", "asof", "as_of", "trade_date"}


def date_index_of(df):
    """Return (DatetimeIndex or None, name of the date column used or None).

    Uses a DatetimeIndex if present, else a column with a date-like name, else
    the first non-numeric column that parses as dates.
    """
    import pandas as pd

    if isinstance(df.index, pd.DatetimeIndex):
        return df.index, None
    candidates = [c for c in df.columns if str(c).strip().lower() in _DATE_NAMES]
    candidates += [c for c in df.columns if c not in candidates and not pd.api.types.is_numeric_dtype(df[c])]
    for c in candidates:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return pd.DatetimeIndex(df[c]), c
        try:
            parsed = pd.to_datetime(df[c], errors="coerce")
        except Exception:  # noqa: BLE001
            continue
        if parsed.notna().mean() > 0.9:
            return pd.DatetimeIndex(parsed), c
    return None, None


def numeric_columns(df) -> list:
    import pandas as pd

    _, date_col = date_index_of(df)
    return [c for c in df.columns if c != date_col and pd.api.types.is_numeric_dtype(df[c])]


def extract_series(df, columns, simple_returns: bool = False, label_prefix: str = "") -> dict:
    """Pull return columns out of a table as pandas Series of daily log returns.

    Series are indexed by date when the table has dates, else by row position.
    """
    import pandas as pd

    idx, _ = date_index_of(df)
    out = {}
    for c in columns:
        s = pd.to_numeric(df[c], errors="coerce")
        s.index = idx if idx is not None else pd.RangeIndex(len(df))
        if simple_returns:
            s = np.log1p(s)
        s = s[np.isfinite(s)]
        if idx is not None:
            s = s[~s.index.isna()]
            s = s.groupby(level=0).last().sort_index()  # one value per date
        out[f"{label_prefix}{c}"] = s
    return out


@dataclass
class CorrelationEstimate:
    rho: float               # average pairwise correlation (off-diagonal mean)
    n_series: int
    matrix: object           # pandas DataFrame of pairwise correlations
    min_overlap: int         # fewest overlapping days used by any pair
    aligned_by: str          # "date" or "position"


def average_pairwise_correlation(series: dict, min_overlap: int = 20) -> CorrelationEstimate:
    """Average off-diagonal correlation across return series.

    If every series is date-indexed, each pair is correlated on its common dates.
    Otherwise all series are aligned by row position (first rows together).
    Pairs with fewer than ``min_overlap`` common observations are ignored.
    """
    import pandas as pd

    if len(series) < 2:
        raise ValueError("Need at least two series to estimate a correlation.")
    all_dated = all(isinstance(s.index, pd.DatetimeIndex) for s in series.values())
    if all_dated:
        frame = pd.concat(series, axis=1, join="outer")
        aligned_by = "date"
    else:
        frame = pd.concat({k: s.reset_index(drop=True) for k, s in series.items()}, axis=1)
        aligned_by = "position"
    corr = frame.corr(min_periods=min_overlap)
    notna = frame.notna().astype(int)
    overlap = notna.T @ notna
    off = ~np.eye(len(series), dtype=bool)
    vals = corr.to_numpy()[off]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError(f"No pair of series shares at least {min_overlap} observations.")
    used = overlap.to_numpy()[off][np.isfinite(corr.to_numpy()[off])]
    return CorrelationEstimate(float(vals.mean()), len(series), corr, int(used.min()), aligned_by)
