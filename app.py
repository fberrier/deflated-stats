"""
Best-of-N dashboard.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import deflated_stats as ds

st.set_page_config(page_title="Deflated Stats Dashboard", layout="wide")

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


@st.cache_data(show_spinner=False)
def make_sample_series(kind, ann_return, ann_vol, dof, skew, ekurt, n_days, seed, end, dated) -> pd.Series:
    """Sample daily log returns from the chosen distribution."""
    marginal = ds.make_parametric(kind, ann_return, ann_vol, dof=dof, skew=skew, excess_kurt=ekurt)
    return ds.generate_returns(marginal, n_days, seed=seed, end=end, dated=dated)


def series_to_bytes(s: pd.Series, file_format: str) -> bytes:
    import io

    buf = io.BytesIO()
    if file_format == "csv":
        dated = isinstance(s.index, pd.DatetimeIndex)
        s.to_frame().to_csv(buf, index=dated, date_format="%Y-%m-%d", float_format="%.10g")
    else:
        s.to_pickle(buf)
    return buf.getvalue()


def _new_draw():
    st.session_state["gen_seed"] = int(st.session_state.get("gen_seed", 1)) + 1


# --------------------------------------------------------------------------- #
# Formatting (UK style: comma thousands separator, point decimal)
# --------------------------------------------------------------------------- #
MINUS = "−"


def fmt_money(x: float) -> str:
    if not np.isfinite(x):
        return "n/a"
    return f"{MINUS if x < 0 else ''}${abs(x):,.0f}"


def fmt_pct(x: float, dp: int = 2) -> str:
    if not np.isfinite(x):
        return "n/a"
    return f"{MINUS if x < 0 else ''}{abs(x) * 100:,.{dp}f}%"


def fmt_num(x: float, dp: int = 2) -> str:
    if not np.isfinite(x):
        return "∞" if x > 0 else ("n/a" if np.isnan(x) else f"{MINUS}∞")
    return f"{MINUS if x < 0 else ''}{abs(x):,.{dp}f}"


def stats_table_html(sections: list[tuple[str, list[tuple[str, str]]]]) -> str:
    """Compact two-column table with section headers; colours follow the Streamlit theme."""
    rows = []
    for title, items in sections:
        rows.append(f'<tr class="sec"><td colspan="2">{title}</td></tr>')
        rows += [f"<tr><td>{k}</td><td class='v'>{v}</td></tr>" for k, v in items]
    css = """
    <style>
    table.dstats {width:100%; border-collapse:collapse; font-size:0.86rem;}
    table.dstats td {padding:3px 6px; border-bottom:1px solid rgba(128,128,128,0.18);}
    table.dstats td.v {text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap;}
    table.dstats tr.sec td {font-weight:600; font-size:0.74rem; text-transform:uppercase;
        letter-spacing:0.04em; opacity:0.65; padding-top:10px; border-bottom:none;}
    </style>"""
    return css + "<table class='dstats'>" + "".join(rows) + "</table>"


# --------------------------------------------------------------------------- #
# Sidebar inputs
# --------------------------------------------------------------------------- #
sb = st.sidebar
sb.header("1 · Return distribution D")
source = sb.radio("Source", ["Parametric", "Upload returns file"], horizontal=True)

returns = None
main_series: dict = {}
gen_on = False
gen_series = None
ann_return = ann_vol = None
dof, skew, ekurt = 5.0, 0.0, 3.0
kind = "gaussian"
block = 1

if source == "Parametric":
    dist_label = sb.selectbox("Distribution", list(DIST_KINDS))
    kind = DIST_KINDS[dist_label]
    ann_return = sb.number_input("Annualised log return (%)", value=2.5, step=1.0, format="%.2f") / 100
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

    gen_on = sb.toggle("Generate a sample series from D", value=False, key="gen_on",
                       help="Draw daily log returns from this distribution, inspect them in the main "
                            "panel, redraw until you like one, then download it (e.g. to test the "
                            "upload mode).")
    if gen_on:
        gen_days = int(sb.number_input("Number of business days", value=1260, min_value=20, step=252,
                                       key="gen_days"))
        gen_dated = sb.checkbox("Include business-day dates", value=True, key="gen_dated")
        gen_end = sb.date_input("Last date", value=pd.Timestamp.today().date(), key="gen_end",
                                format="DD/MM/YYYY", disabled=not gen_dated)
        if "gen_seed" not in st.session_state:
            st.session_state["gen_seed"] = 1
        gen_seed = int(sb.number_input("Seed", step=1, format="%d", key="gen_seed",
                                       help="'New draw' in the main panel moves this on by one."))
        gen_fmt = sb.radio("File format", ["csv", "pkl"], horizontal=True, key="gen_fmt",
                           help="csv: columns date, log_return. pkl: a pandas Series named log_return.")
        try:
            gen_series = make_sample_series(kind, ann_return, ann_vol, dof, skew, ekurt, gen_days, gen_seed,
                                            str(gen_end), gen_dated)
        except Exception as exc:  # noqa: BLE001
            sb.error(f"Could not generate: {exc}")
            gen_series = None
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
GMV_DEFAULT = 10_000_000.0
gmv = GMV_DEFAULT
series_panel_shown = source != "Parametric" or gen_series is not None
if stat == "pnl":
    if not series_panel_shown:
        gmv = sb.number_input("Constant GMV G ($)", value=GMV_DEFAULT, min_value=1.0, step=1_000_000.0,
                              format="%.0f", key="gmv_sidebar")
        sb.caption(f"GMV: {fmt_money(gmv)}")
    else:
        sb.caption("GMV is set in the series panel.")
sb.caption("Best = **lowest** value" if not spec.higher_is_better else "Best = **highest** value")
N = int(sb.number_input("Number of trials N", value=1, min_value=1, step=1))
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
st.title("Deflated Stats Dashboard")
st.caption(
    "Distribution of the best value of a statistic across N variants that all share the same true "
    "return distribution. A backtest improvement only counts if it beats this distribution, "
    "not the single-run value."
)

if source == "Upload returns file" and returns is None:
    st.info("Upload a file of daily returns in the sidebar to start.")
    st.stop()

# ---- Series panel: observed (upload mode) or generated (parametric mode) ----
def series_panel(title: str, obs: pd.Series, gmv_key: str, generated: bool = False) -> float:
    """Chart + stats table for one return series, in a collapsible section. Returns the GMV."""
    r = obs.to_numpy(dtype=float)
    dated = isinstance(obs.index, pd.DatetimeIndex)
    with st.expander(title, expanded=True):
        chart_col, table_col = st.columns([2.3, 1], gap="large")
        with table_col:
            if generated:
                b1, b2 = st.columns(2)
                b1.button("New draw", on_click=_new_draw, width="stretch",
                          help="Redraw with the next seed.")
                b2.download_button(
                    "Download", data=series_to_bytes(obs, gen_fmt),
                    file_name=f"sample_{kind}_{len(obs)}d_seed{gen_seed}.{gen_fmt}",
                    mime="text/csv" if gen_fmt == "csv" else "application/octet-stream", width="stretch")
                st.caption(f"Seed {gen_seed} · target: return {fmt_pct(ann_return)} · vol {fmt_pct(ann_vol)} · "
                           f"Sharpe {fmt_num(ann_return / ann_vol)}")
            g = st.number_input("GMV ($)", value=GMV_DEFAULT, min_value=1.0, step=1_000_000.0, format="%.0f",
                                key=gmv_key, help="Constant gross market value used for the $ P&L figures "
                                                  "and for the PnL statistic.")
            rep = ds.series_report(r, g)
            period = ([("Start", obs.index[0].strftime("%d %b %Y")), ("End", obs.index[-1].strftime("%d %b %Y"))]
                      if dated else [])
            period += [("Business days", f"{rep['n_days']:,}"), ("Years", fmt_num(rep["years"], 2))]
            st.markdown(stats_table_html([
                ("Period", period),
                ("Return & risk", [
                    ("Annualised return", fmt_pct(rep["ann_return"])),
                    ("Cumulative log return", fmt_pct(rep["total_log_return"])),
                    ("Annualised volatility", fmt_pct(rep["ann_vol"])),
                    ("Max drawdown", fmt_pct(-rep["max_dd"])),
                ]),
                ("Ratios", [
                    ("Sharpe", fmt_num(rep["sharpe"])),
                    ("Sortino", fmt_num(rep["sortino"])),
                    ("Calmar", fmt_num(rep["calmar"])),
                ]),
                (f"P&L at {fmt_money(g)} GMV", [
                    ("Total P&L", fmt_money(rep["total_pnl"])),
                    ("Annualised P&L", fmt_money(rep["ann_pnl"])),
                    ("Max drawdown ($)", fmt_money(rep["max_dd_usd"])),
                    ("Best day", fmt_money(rep["best_day_usd"])),
                    ("Worst day", fmt_money(rep["worst_day_usd"])),
                ]),
                ("Distribution", [
                    ("Hit rate", fmt_pct(rep["hit_rate"], 1)),
                    ("Skew", fmt_num(rep["skew"])),
                    ("Excess kurtosis", fmt_num(rep["excess_kurt"])),
                ]),
            ]), unsafe_allow_html=True)
        with chart_col:
            x = obs.index if dated else np.arange(1, len(obs) + 1)
            pnl = ds.pnl_path(r, g)
            dd = ds.drawdown_path(r)
            sfig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.04)
            sfig.add_trace(go.Scatter(x=x, y=pnl, name="Cumulative P&L", line=dict(color="#2F6FDE", width=2),
                                      hovertemplate="%{y:$,.0f}<extra>Cumulative P&L</extra>"), row=1, col=1)
            sfig.add_trace(go.Scatter(x=x, y=dd * 100, name="Drawdown", fill="tozeroy",
                                      line=dict(color="#D9480F", width=1), fillcolor="rgba(217,72,15,0.25)",
                                      hovertemplate="%{y:,.2f}%<extra>Drawdown</extra>"), row=2, col=1)
            sfig.update_yaxes(title_text="Cumulative P&L ($)", tickprefix="$", tickformat=",.3s", row=1, col=1)
            sfig.update_yaxes(title_text="Drawdown (%)", ticksuffix="%", row=2, col=1)
            if not dated:
                sfig.update_xaxes(title_text="Business day", row=2, col=1)
            sfig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                               hovermode="x unified", separators=".,")
            st.plotly_chart(sfig, width="stretch")
            st.caption("P&L at constant GMV, not compounded (as in the PnL statistic). Drawdown is measured on "
                       "the compounded equity curve (as in the Max drawdown statistic).")
    return g


if source == "Upload returns file":
    gmv = series_panel("Observed series", next(iter(main_series.values())), "gmv_obs")
elif gen_series is not None:
    gmv = series_panel("Generated series", gen_series, "gmv_gen", generated=True)
if series_panel_shown:
    st.subheader("Best-of-N distribution")

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
        return fmt_money(v)
    return f"{fmt_num(v, 2)}{spec.unit}" if spec.unit else fmt_num(v, 3)


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
    P = st.number_input("p-value P", value=0.25, min_value=1e-4, max_value=0.9999, step=0.05, format="%.4f",
                        help="The chart is centred on the value needed for this p-value.")
    crit = res.critical_value(P)
    st.markdown(f"**{fmt(crit)}**")
    st.caption(f"Beat this to be {100*(1-P):.2f}% confident the result is better than the best of {N} lucky draws.")
    if not res.tail_is_resolved(P):
        st.warning("Few simulated samples sit beyond this level; raise precision for a stable estimate.")

# ---- CDF plot ----
def _finite(vals):
    return [v for v in vals if np.isfinite(v)]


# The x-axis is centred on the value needed for the chosen p-value P (the critical
# value). The half-width is the furthest of: the best-of-N 1% and 99% quantiles and
# the observed value V, plus a small margin. The single-trial curve is drawn as a
# reference wherever it falls. "Show full tails" widens the symmetric window to the
# 0.1%–99.9% quantiles of both curves.
centre = crit if np.isfinite(crit) else res.best_quantile(0.5)


def _symmetric_window(points, margin):
    half = max([abs(x - centre) for x in _finite(points)] + [0.0])
    if half == 0.0:
        half = 0.1 * abs(centre) + 1e-9
    half *= 1.0 + margin
    return centre - half, centre + half


focus_pts = [res.best_quantile(0.01), res.best_quantile(0.99), V, crit]
view = _symmetric_window(focus_pts, 0.06)
wide = _symmetric_window(focus_pts + [res.best_quantile(0.001), res.best_quantile(0.999),
                                      res.single_quantile(0.001), res.single_quantile(0.999)], 0.03)

show_full = st.checkbox("Show full tails (0.1%–99.9%)", value=False)
# Fine grid over the focused range plus a coarser one over the full range, so zooming
# out (double-click the chart) still shows the tails.
grid = np.unique(np.concatenate([np.linspace(*view, 800), np.linspace(*wide, 400)]))
xs = grid * spec.display_scale
x_range = [v * spec.display_scale for v in (wide if show_full else view)]

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
    xaxis=dict(range=x_range),
    yaxis_title="Cumulative probability P(S ≤ x)", yaxis=dict(range=[0, 1.0]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), hovermode="x unified",
)
st.plotly_chart(fig, width="stretch")

# ---- Quantile table ----
qs = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
table = pd.DataFrame({
    "Quantile": [f"{q:.0%}" for q in qs],
    "Single trial": [fmt(res.single_quantile(q)) for q in qs],
    f"Best of {N}": [fmt(res.best_quantile(q)) for q in qs],
})
with st.expander("Quantiles"):
    st.dataframe(table, hide_index=True, width="stretch")

if rho_est is not None:
    with st.expander(f"Correlation matrix ({rho_est.n_series} series, average {rho_est.rho:.3f})"):
        st.dataframe(rho_est.matrix.round(3), width="stretch")
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
