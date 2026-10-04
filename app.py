"""
Best-of-N dashboard.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import deflated_stats as ds

st.set_page_config(page_title="Best-of-N significance", layout="wide")

PRECISION = {"Fast (2e7 draws)": 2e7, "Standard (1e8 draws)": 1e8, "Precise (4e8 draws)": 4e8}
DIST_KINDS = {
    "Gaussian": "gaussian",
    "Student-t (fat tails)": "student_t",
    "Skew-normal (skew)": "skew_normal",
    "Johnson SU (skew + fat tails)": "johnson_su",
}


# --------------------------------------------------------------------------- #
# Cached computations
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def run_parametric(kind, ann_return, ann_vol, dof, skew, ekurt, T, N, stat, rho, gmv, budget, seed):
    marginal = ds.make_parametric(kind, ann_return, ann_vol, dof=dof, skew=skew, excess_kurt=ekurt)
    res = ds.simulate_best_of_n(marginal, T, N, stat, rho=rho, gmv=gmv, draw_budget=budget, seed=seed)
    return res, marginal.info


@st.cache_data(show_spinner=False)
def run_empirical(returns, block, T, N, stat, rho, gmv, budget, seed):
    marginal = ds.EmpiricalMarginal(returns, block=block)
    res = ds.simulate_best_of_n(marginal, T, N, stat, rho=rho, gmv=gmv, draw_budget=budget, seed=seed)
    return res, marginal.summary()


@st.cache_data(show_spinner=False)
def load_table(data: bytes, name: str) -> pd.DataFrame:
    import io

    return ds.load_returns_table(io.BytesIO(data), name)


# --------------------------------------------------------------------------- #
# Sidebar inputs
# --------------------------------------------------------------------------- #
sb = st.sidebar
sb.header("1 · Return distribution D")
source = sb.radio("Source", ["Parametric", "Upload returns file"], horizontal=True)

returns = None
main_series: dict = {}
ann_return = ann_vol = None
dof, skew, ekurt = 5.0, 0.0, 3.0
kind = "gaussian"
block = 1

if source == "Parametric":
    dist_label = sb.selectbox("Distribution", list(DIST_KINDS))
    kind = DIST_KINDS[dist_label]
    ann_return = sb.number_input("Annualised log return (%)", value=10.0, step=1.0, format="%.2f") / 100
    sharpe_in = sb.number_input("Sharpe ratio", value=1.0, step=0.1, format="%.2f")
    if sharpe_in == 0:
        ann_vol = sb.number_input("Annualised vol (%) — needed when Sharpe = 0", value=10.0, min_value=0.01) / 100
    else:
        ann_vol = abs(ann_return / sharpe_in) if ann_return != 0 else None
        if ann_vol is None or ann_vol == 0:
            ann_vol = sb.number_input("Annualised vol (%) — needed when return = 0", value=10.0, min_value=0.01) / 100
            if sharpe_in != 0:
                sb.caption("With zero return the Sharpe is 0; vol taken from this box.")
        else:
            sb.caption(f"Implied annualised vol: **{ann_vol*100:.2f}%**")
        if sharpe_in < 0 and ann_return > 0 or sharpe_in > 0 and ann_return < 0:
            sb.warning("Return and Sharpe have opposite signs; using |return / Sharpe| for the vol.")

    if kind == "student_t":
        dof = sb.number_input("Degrees of freedom ν (> 2)", value=5.0, min_value=2.1, step=0.5,
                              help="Lower ν means fatter tails. Excess kurtosis = 6 / (ν − 4) for ν > 4.")
    if kind == "skew_normal":
        skew = sb.slider("Skewness", -0.99, 0.99, -0.5, 0.01,
                         help="The skew-normal can only reach |skew| < 0.995.")
    if kind == "johnson_su":
        skew = sb.number_input("Skewness", value=-0.5, step=0.1, format="%.2f")
        ekurt = sb.number_input("Excess kurtosis", value=3.0, min_value=0.0, step=0.5, format="%.2f",
                                help="Johnson SU needs more kurtosis than a lognormal with the same skew; "
                                     "if the target is infeasible the nearest feasible point is used.")
else:
    up = sb.file_uploader("Daily log returns (.csv, .pkl, .parquet)", type=["csv", "pkl", "pickle", "parquet"],
                          help="One row per business day. Only upload pickles you trust.")
    if up is not None:
        try:
            df = load_table(up.getvalue(), up.name)
        except Exception as exc:  # noqa: BLE001
            sb.error(f"Could not read the file: {exc}")
            df = None
        if df is not None:
            numeric_cols = ds.numeric_columns(df)
            if not numeric_cols:
                sb.error("No numeric column found.")
            else:
                col = sb.selectbox("Returns column", numeric_cols)
                simple = sb.checkbox("Column holds simple returns (convert to log)", value=False)
                main_series = ds.extract_series(df, [col], simple, label_prefix="[study] ")
                returns = next(iter(main_series.values())).to_numpy(dtype=float)
                block = sb.number_input(
                    "Bootstrap block length (days)", value=1, min_value=1, step=1,
                    help="1 resamples days independently. A longer block keeps autocorrelation and "
                         "volatility clustering (try 5–20).")

sb.header("2 · Horizon, statistic, trials")
T = int(sb.number_input("Horizon T (business days)", value=504, min_value=5, step=21))
sb.caption(f"≈ {T / ds.TRADING_DAYS:.2f} years")
stat_label = sb.selectbox("Statistic S", [s.label for s in ds.STATS.values()], index=2)
stat = next(k for k, s in ds.STATS.items() if s.label == stat_label)
spec = ds.STATS[stat]
gmv = 1.0
if stat == "pnl":
    gmv = sb.number_input("Constant GMV G", value=100_000_000.0, min_value=1.0, step=10_000_000.0, format="%.0f")
sb.caption("Best = **lowest** value" if not spec.higher_is_better else "Best = **highest** value")
N = int(sb.number_input("Number of trials N", value=20, min_value=1, step=1))
sb.header("3 · Correlation between trials ρ")
RHO_HELP = ("Pairwise correlation of daily returns between variants. Tweaks of the same strategy are "
            "often 0.5–0.9 correlated, which makes the best of N much less extreme than for "
            "independent trials. 0 is the conservative (strictest) choice.")
rho_mode = sb.radio("ρ from", ["Enter value", "Estimate from return series"], horizontal=True, help=RHO_HELP)
rho_est = None
rho_note = None
if rho_mode == "Estimate from return series":
    corr_files = sb.file_uploader(
        "Variant return series (.csv, .pkl, .parquet)", type=["csv", "pkl", "pickle", "parquet"],
        accept_multiple_files=True, key="corr_files",
        help="Daily returns of the variants you tried. Each file can hold one or several columns; "
             "each selected column counts as one series. With a date column or index, pairs are "
             "matched on common dates; otherwise rows are matched by position.")
    corr_simple = sb.checkbox("These files hold simple returns (convert to log)", value=False, key="corr_simple")
    corr_series: dict = {}
    for i, f in enumerate(corr_files or []):
        try:
            cdf = load_table(f.getvalue(), f.name)
        except Exception as exc:  # noqa: BLE001
            sb.error(f"{f.name}: could not read ({exc})")
            continue
        cols = ds.numeric_columns(cdf)
        if not cols:
            sb.warning(f"{f.name}: no numeric column, skipped.")
            continue
        if len(cols) > 1:
            cols = sb.multiselect(f"Columns from {f.name}", cols, default=cols, key=f"corr_cols_{i}_{f.name}")
        corr_series.update(ds.extract_series(cdf, cols, corr_simple, label_prefix=f"{f.name}:"))
    include_main = False
    if main_series:
        include_main = sb.checkbox("Include the returns under study", value=True, key="corr_incl_main")
        if include_main:
            corr_series = {**main_series, **corr_series}
    if len(corr_series) >= 2:
        try:
            rho_est = ds.average_pairwise_correlation(corr_series)
        except ValueError as exc:
            sb.error(str(exc))
    elif corr_files:
        sb.info("Need at least two series in total to estimate ρ; using the box below meanwhile.")

if rho_est is not None:
    rho_raw = rho_est.rho
    rho = float(np.clip(rho_raw, 0.0, 0.99))
    sb.metric("Average pairwise correlation", f"{rho_raw:.3f}")
    sb.caption(f"{rho_est.n_series} series · aligned by {rho_est.aligned_by} · "
               f"min overlap {rho_est.min_overlap:,} days")
    if rho_raw < 0:
        rho_note = f"Average correlation is negative ({rho_raw:.3f}); ρ = 0 is used."
        sb.warning(rho_note)
    elif rho_raw > 0.99:
        sb.warning("Average correlation above 0.99; capped at 0.99.")
else:
    rho = sb.number_input("ρ", value=0.0, min_value=0.0, max_value=0.99, step=0.05, format="%.3f",
                          key="rho_box", help=RHO_HELP)

sb.header("4 · Simulation")
budget = PRECISION[sb.selectbox("Precision", list(PRECISION), index=1)]
seed = int(sb.number_input("Random seed", value=42, step=1))

# --------------------------------------------------------------------------- #
# Main panel
# --------------------------------------------------------------------------- #
st.title("Is the improvement real, or just the best of N draws?")
st.caption(
    "Distribution of the best value of a statistic across N variants that all share the same true "
    "return distribution. A backtest improvement only counts if it beats this distribution, "
    "not the single-run value."
)

if source == "Upload returns file" and returns is None:
    st.info("Upload a file of daily returns in the sidebar to start.")
    st.stop()

with st.spinner("Simulating…"):
    try:
        if source == "Parametric":
            res, info = run_parametric(kind, ann_return, ann_vol, dof, skew, ekurt, T, N, stat, rho, gmv, budget, seed)
        else:
            res, info = run_empirical(returns, block, T, N, stat, rho, gmv, budget, seed)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Simulation failed: {exc}")
        st.stop()


def fmt(x: float) -> str:
    if not np.isfinite(x):
        return "∞" if x > 0 else "−∞"
    v = x * spec.display_scale
    if stat == "pnl":
        return f"{v:,.0f}"
    return f"{v:.2f}{spec.unit}" if spec.unit else f"{v:.3f}"


# Distribution facts
if source == "Parametric":
    shape = ""
    if kind != "gaussian":
        shape = f" · skew {info['skew']:.2f} · excess kurtosis {info['excess_kurt']:.2f}"
        if kind == "johnson_su" and (abs(info["skew"] - skew) > 0.02 or abs(info["excess_kurt"] - ekurt) > 0.05):
            st.warning("The requested skew/kurtosis is outside the Johnson SU region; the nearest feasible "
                       "shape is used (shown below).")
    st.caption(f"D: {dist_label} · return {ann_return*100:.2f}% · vol {ann_vol*100:.2f}% · "
               f"Sharpe {ann_return/ann_vol:.2f}{shape}")
else:
    st.caption(
        f"D: empirical, {info['n_obs']:,} days · return {info['ann_return']*100:.2f}% · vol {info['ann_vol']*100:.2f}% · "
        f"Sharpe {info['sharpe']:.2f} · skew {info['skew']:.2f} · excess kurtosis {info['excess_kurt']:.2f} · "
        f"block {block}")

st.caption(f"ρ = {rho:.3f} " + (f"(average of {rho_est.n_series} series)" if rho_est is not None else "(entered)"))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Single trial: median", fmt(res.single_quantile(0.5)))
c2.metric(f"Best of {N}: median", fmt(res.best_quantile(0.5)))
c3.metric(f"Best of {N}: mean", fmt(res.expected_best()))
c4.metric("Needed for p = 5%", fmt(res.critical_value(0.05)))

# ---- Inputs: V and P ----
left, right = st.columns(2)
with left:
    st.subheader("p-value of an observed value")
    default_v = float(res.critical_value(0.10) * spec.display_scale)
    if not np.isfinite(default_v):
        default_v = float(res.best_quantile(0.5) * spec.display_scale)
    v_disp = st.number_input(
        f"Observed value V{f' ({spec.unit})' if spec.unit else ''}", value=round(default_v, 4),
        key=f"v_{stat}", format="%.4f")
    V = v_disp / spec.display_scale
    p = res.p_value(V)
    cmp = "≤" if not spec.higher_is_better else "≥"
    res_floor = res.p_value_resolution()
    if p == 0:
        st.markdown(f"**p < {res_floor:.1e}** (beyond the simulation's resolution)")
    else:
        st.markdown(f"**p = {p:.4f}**")
    st.caption(f"P(best of {N} {cmp} V) under D: how often pure luck across {N} trials does at least this well.")
with right:
    st.subheader("Value needed for a given p-value")
    P = st.number_input("p-value P", value=0.05, min_value=1e-4, max_value=0.5, step=0.01, format="%.4f")
    crit = res.critical_value(P)
    st.markdown(f"**{fmt(crit)}**")
    st.caption(f"Beat this to be {100*(1-P):.2f}% confident the result is better than the best of {N} lucky draws.")
    if not res.tail_is_resolved(P):
        st.warning("Few simulated samples sit beyond this level; raise precision for a stable estimate.")

# ---- CDF plot ----
lo = min(res.single_quantile(0.001), res.best_quantile(0.001))
hi = max(res.single_quantile(0.999), res.best_quantile(0.999))
if not (np.isfinite(lo) and np.isfinite(hi)) or lo == hi:
    lo, hi = res.single_quantile(0.01), res.best_quantile(0.99)
pad = 0.05 * (hi - lo)
grid = np.linspace(lo - pad, hi + pad, 600)
xs = grid * spec.display_scale

fig = go.Figure()
fig.add_trace(go.Scatter(x=xs, y=res.single_cdf(grid), name="Single trial (N = 1)",
                         line=dict(color="#9AA5B1", width=2, dash="dash")))
fig.add_trace(go.Scatter(x=xs, y=res.cdf(grid), name=f"Best of {N}",
                         line=dict(color="#2F6FDE", width=3)))
fig.add_vline(x=V * spec.display_scale, line=dict(color="#D9480F", width=2),
              annotation_text=f"V · p={p:.3g}", annotation_position="top left")
fig.add_vline(x=crit * spec.display_scale, line=dict(color="#2B8A3E", width=2, dash="dot"),
              annotation_text=f"needed for p={P:g}", annotation_position="bottom right")
fig.update_layout(
    height=480, margin=dict(l=10, r=10, t=30, b=10),
    xaxis_title=f"{spec.label}{f' ({spec.unit})' if spec.unit else ''}",
    yaxis_title="Cumulative probability P(S ≤ x)", yaxis=dict(range=[0, 1.0]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

# ---- Quantile table ----
qs = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
table = pd.DataFrame({
    "Quantile": [f"{q:.0%}" for q in qs],
    "Single trial": [fmt(res.single_quantile(q)) for q in qs],
    f"Best of {N}": [fmt(res.best_quantile(q)) for q in qs],
})
with st.expander("Quantiles"):
    st.dataframe(table, hide_index=True, use_container_width=True)

if rho_est is not None:
    with st.expander(f"Correlation matrix ({rho_est.n_series} series, average {rho_est.rho:.3f})"):
        st.dataframe(rho_est.matrix.round(3), use_container_width=True)
        st.caption(f"Pairs aligned by {rho_est.aligned_by}; pairs with fewer than 20 shared days are left out. "
                   f"ρ used in the simulation: {rho:.3f}.")

if stat == "sharpe" and rho == 0 and N >= 5 and source == "Parametric":
    ref = ds.expected_max_sharpe_gaussian(ann_return / ann_vol, T, N, info.get("skew", 0.0),
                                          min(info.get("excess_kurt", 0.0), 50.0))
    st.caption(f"Check: Bailey & López de Prado approximation of E[best Sharpe] = {ref:.3f} "
               f"(simulated {res.expected_best():.3f}).")

with st.expander("Method notes"):
    st.markdown(f"""
- Each trial is a fresh path of **T = {T}** daily log returns from D. Every trial has the same true
  distribution, so any spread between them is luck.
- **Independent trials (ρ = 0):** single-trial values are simulated and the best-of-N CDF is computed exactly
  as F(x)<sup>N</sup>. Every simulated path contributes, which keeps the tails accurate even for large N.
- **Correlated trials (ρ > 0):** N trials are generated together. Parametric D and i.i.d. bootstrap use a Gaussian
  copula, which preserves each trial's marginal. Block bootstrap copies each block from a common path with
  probability √ρ. Simulated groups: {res.ys.size:,}.
- Statistics: Return = mean log return × 252; Vol = std × √252; Sharpe = Return / Vol;
  Sortino uses downside deviation below 0; Max drawdown is on exp(cumulative log return);
  Calmar = Return / Max drawdown; PnL = G × Σ(eʳ − 1) with constant GMV G.
- p-value = P(best of N is at least as good as V). For Vol and Max drawdown, lower is better, so best = min.
- Choosing N: count every variant you looked at, including ones you discarded. If they were highly correlated,
  set ρ accordingly rather than shrinking N.
""", unsafe_allow_html=True)
