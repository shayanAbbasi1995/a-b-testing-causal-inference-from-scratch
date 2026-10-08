# A/B Testing & Causal Inference Toolkit

An experimentation toolkit written from first principles in Python and NumPy: the statistics
behind an A/B test, the diagnostics that tell you whether to trust it, and two
quasi-experimental estimators for when you cannot randomize. It ends in a small decision
rule that turns a primary metric and its guardrails into a ship / no-ship call.

Everything is in one file, [`model.py`](model.py), as 22 small functions with no dependencies
beyond NumPy and the standard library. [`scaffold.py`](scaffold.py) runs them end to end on
simulated data. A short walkthrough of how it was built is on the
[project page](https://shayanabbasi1995.github.io/a-b-testing-causal-inference-from-scratch/).

## What is inside

| Part | Functions | What it does |
|---|---|---|
| Normal distribution | `standard_normal_cdf`, `standard_normal_ppf` | CDF via the error function; quantiles for critical values |
| Two-proportion test | `pooled_proportion`, `pooled_standard_error`, `two_proportion_z_statistic`, `two_sided_p_value`, `unpooled_standard_error`, `confidence_interval_from_se` | z-test for a difference in conversion rates, plus a confidence interval for the lift |
| Experiment design | `required_sample_size_per_variant`, `statistical_power` | sample size for a target power and minimum detectable effect, and the power of a given design |
| Randomization check | `chi_square_statistic`, `sample_ratio_mismatch_check` | Pearson chi-square test for sample ratio mismatch (SRM) |
| Multiple testing | `bonferroni_correction`, `benjamini_hochberg_correction` | family-wise error and false-discovery-rate control across many metrics |
| Difference-in-differences | `group_mean_change`, `difference_in_differences_simple`, `build_did_design_matrix`, `ols_normal_equations`, `did_effect_from_regression` | DiD from group means and as the interaction coefficient of an OLS regression |
| Synthetic control | `fit_synthetic_control_weights`, `synthetic_control_effect` | donor weights on the simplex fitted by gradient descent; post-period gap as the effect |
| Decision | `ship_decision` | ship only if the primary metric is significant and positive and no guardrail is significantly harmed |

## Quick start

Requires Python 3.9+ and NumPy.

```bash
git clone https://github.com/shayanAbbasi1995/a-b-testing-causal-inference-from-scratch.git
cd a-b-testing-causal-inference-from-scratch
pip install numpy
python scaffold.py
```

Output of the demo (seeded, so it is reproducible):

```text
Required per-variant sample size: 3841
Observed rates: A=0.1008 (n=3841), B=0.1234 (n=3841)
SRM chi-square=0.0000, mismatch_detected=False
Primary test: z=3.1465, p=0.0017
95% CI for lift (B-A): [0.0086, 0.0368]
Achieved power at MDE: 0.8000
Guardrail p-values: [0.012, 0.2, 0.045, 0.6]
Bonferroni significant: [ True False False False]
BH-FDR significant:    [ True False False False]
DiD (simple means): 1.7000
DiD (OLS interaction coef): 1.7000
Synthetic control weights: [0.308 0.373 0.319]
Synthetic control average post-period effect: 1.0269
Ship decision: {'ship': True, 'reason': 'ship'}
```

Reading it: to detect a lift from 10% to 12% at alpha = 0.05 with 80% power you need 3,841
users per variant; the simulated test finds a significant lift whose 95% interval excludes
zero; the two DiD estimators agree exactly; and synthetic control recovers the simulated
treatment effect of 1.0 (estimate 1.03).

## Using the functions

```python
from model import *

# Plan: users per variant to detect 10% -> 12% at alpha 0.05, power 0.80
n = required_sample_size_per_variant(0.10, 0.02, alpha=0.05, power=0.80)   # 3841
statistical_power(n, 0.10, 0.02, alpha=0.05)                               # 0.80

# Check the split before reading results
sample_ratio_mismatch_check([5000, 5300], [0.5, 0.5], alpha=0.05)
# {'chi_square': 8.74, 'p_value': 0.0031, 'srm_detected': True}

# Correct many metrics at once (returns a boolean mask of significant hypotheses)
p = [0.001, 0.012, 0.03, 0.04, 0.2]
bonferroni_correction(p, 0.05)            # [ True False False False False]
benjamini_hochberg_correction(p, 0.05)    # [ True  True  True  True False]
```

Arguments are fractions, not percentages: pass `0.95` for a 95% interval and `0.05` for alpha.

## Method notes

- **Two-proportion z-test.** The test statistic uses the pooled standard error (the null is
  equal rates); the confidence interval for the lift uses the unpooled standard error.
- **Sample size and power.** Normal-approximation formula for two proportions,
  `n = (z_{1-α/2}·sqrt(2·p̄(1-p̄)) + z_{power}·sqrt(p1(1-p1) + p2(1-p2)))² / (p1 - p2)²`,
  rounded up. `statistical_power` inverts the same formula, so the two are consistent.
- **Benjamini–Hochberg.** Step-up procedure: find the largest rank `k` with
  `p_(k) ≤ (k/m)·α` and reject the `k` smallest p-values.
- **Difference-in-differences.** `(treated post − treated pre) − (control post − control pre)`.
  The regression version fits `y = β0 + β1·treated + β2·post + β3·treated·post` by the normal
  equations and returns `β3`; on a balanced 2×2 design it equals the difference of means.
- **Synthetic control.** Weights are parameterized as a softmax, so they are non-negative and
  sum to one by construction, and are fitted by gradient descent on the pre-period squared
  error. The effect is the average post-period gap between the treated unit and its synthetic
  counterpart.

## Limitations

This is a learning implementation: it shows how the methods work, and is not a replacement for
`statsmodels`, `scipy.stats` or a production experimentation platform.

- All tests use the normal approximation; there are no exact or small-sample tests.
- The SRM p-value is computed from a one-degree-of-freedom reference, so it is exact for two
  arms only. With three or more arms it is too small (for counts 1000/1000/1100 it reports
  0.011 where the chi-square distribution with 2 degrees of freedom gives about 0.040).
- `standard_normal_ppf` is defined for `0 < p < 1` and uses `statistics.NormalDist` from the
  standard library; `p = 0` and `p = 1` currently raise an error.
- Difference-in-differences returns a point estimate only: no standard errors, clustering or
  parallel-trends check.
- Synthetic control matches on pre-period outcomes only and does no inference (no placebo
  tests). When donors are close to collinear the weights are not unique: in the demo they
  differ from the weights used to simulate the data, although the effect is still recovered.
- `ship_decision` is a simple rule over p-values and effect signs; it has no sequential
  testing or minimum practical effect.

## Repository layout

```text
model.py        the 22 functions, in the order they were built
scaffold.py     end-to-end demo on simulated data
docs/           project page (GitHub Pages)
```

---

Built step by step as a guided project on [Deep-ML](https://www.deep-ml.com/).
