# Best-of-N significance dashboard

When you try N variants of a strategy and keep the best one, the winner looks better than the true strategy even if nothing changed. This tool simulates the distribution of the **best of N trials** of a performance statistic over a horizon T, under a fixed return distribution D. An observed improvement only counts if it is unlikely under that distribution.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Inputs

| Input | Meaning |
|---|---|
| D | Gaussian, Student-t (ν), skew-normal (skew), Johnson SU (skew + excess kurtosis), each set by annualised log return and Sharpe; or an uploaded file of daily returns (.csv / .pkl / .parquet), bootstrapped i.i.d. or in blocks |
| T | Horizon in business days (252 per year) |
| S | Return, vol, Sharpe, Sortino, Calmar, max drawdown, PnL (with constant GMV G) |
| N | Number of trials (count every variant tried, including discarded ones) |
| ρ | Pairwise daily-return correlation between trials. 0 is the strictest choice. Enter it in a box, or estimate it from uploaded return series (see below) |

### Estimating ρ from return series

Choose "Estimate from return series" and upload one or more files with the daily returns of the variants you tried. Each selected column counts as one series, and a file can hold several columns. If the returns under study were uploaded as D, they are included by default, giving n + 1 series.

- ρ is the average off-diagonal correlation across all series.
- If every series has dates (a date column or index), each pair is correlated on its common dates. Otherwise rows are matched by position.
- Pairs that share fewer than 20 days are ignored.
- With fewer than two series, the dashboard falls back to the box.
- A negative average is floored at 0.

The correlation matrix appears in an expander under the chart.

## Generating a sample series

In parametric mode, switch on **Generate a sample series from D** in the sidebar and set the number of business days, dates (and the last date), the seed and the file format.

The draw appears in a collapsible **Generated series** panel with the same chart and stats table as an uploaded series, plus the target return, vol and Sharpe for comparison.

- **New draw** moves on to the next seed. Keep redrawing until a realised path suits you.
- **Download** saves the draw to your browser's downloads folder.

File formats:

- **csv:** columns `date, log_return`, or just `log_return` without dates.
- **pkl:** a pandas Series named `log_return`.

Either file can be loaded straight back in upload mode.

## Observed series panel (upload mode)

When a returns file is loaded, a collapsible **Observed series** panel shows:

- **Chart:** cumulative P&L at a constant GMV, with the drawdown underneath.
- **GMV box:** defaults to $10,000,000. Whenever a series panel is shown, its GMV also sets the GMV for the PnL statistic.
- **Stats table:**
  - period
  - annualised and cumulative return
  - volatility and max drawdown
  - Sharpe, Sortino and Calmar
  - total, annualised and drawdown P&L in $, plus the best and worst day
  - hit rate, skew and excess kurtosis

P&L is not compounded, matching the PnL statistic. Drawdown % is measured on the compounded equity curve, matching the Max drawdown statistic.

## Outputs

- CDF of the best of N, with the single-trial CDF for reference
- p-value of an observed value V: P(best of N is at least as good as V)
- Value needed to reach a chosen p-value P: the (1 − P) quantile of the best of N
- Quantile table, plus a Bailey & López de Prado check for the Gaussian Sharpe case

For vol and max drawdown, lower is better, so "best" means the minimum.

## Method

- **ρ = 0:** single-trial values are simulated and the best-of-N CDF is F(x)^N. This is exact for independent trials and uses every simulated path.
- **ρ > 0:** groups of N trials are simulated together. Parametric D and i.i.d. bootstrap use a Gaussian copula, which keeps each trial's marginal distribution. Block bootstrap copies each block from a common path with probability √ρ.
- Parametric quantile functions are tabulated on a normal-score grid, so fat-tailed and skewed distributions sample as fast as the Gaussian.

## Using the core from Python

```python
import deflated_stats as ds

d = ds.make_parametric("student_t", ann_return=0.10, ann_vol=0.10, dof=5)
res = ds.simulate_best_of_n(d, T=504, N=50, stat="sharpe", rho=0.3)
res.p_value(2.4), res.critical_value(0.05), res.expected_best()

series = ds.extract_series(variants_df, ["v1", "v2", "v3"])   # dict of log-return Series
rho = ds.average_pairwise_correlation(series).rho

emp = ds.EmpiricalMarginal(my_log_returns, block=10)
res = ds.simulate_best_of_n(emp, T=504, N=50, stat="maxdd")
```

## Checks run

- Simulated E[max Sharpe] matches the Bailey & López de Prado approximation: 2.10 vs 2.11 at N=10, 2.80 vs 2.79 at N=100, 3.34 vs 3.30 at N=1000. Setup: T=504, Sharpe 1.
- The exact F^N path matches brute-force simulation.
- Realised pairwise return correlation is close to ρ in both bootstrap modes.
- Parametric marginals hit their target mean, vol, skew and kurtosis.
