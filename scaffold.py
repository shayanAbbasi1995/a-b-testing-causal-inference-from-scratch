"""
A/B Testing & Causal Inference Toolkit scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""End-to-end demo of the A/B testing & causal inference toolkit."""

import numpy as np


def main():
    np.random.seed(0)

    # ---- 1. Sample sizing before we launch ----
    baseline_rate = 0.10
    mde = 0.02  # detect a lift from 10% -> 12%
    alpha = 0.05
    power = 0.80
    per_variant_n = required_sample_size_per_variant(baseline_rate, mde, alpha, power)
    print(f"Required per-variant sample size: {per_variant_n}")

    # ---- 2. Simulate an experiment ----
    n_a = n_b = int(per_variant_n)
    true_p_a, true_p_b = 0.10, 0.12
    successes_a = int(np.random.binomial(n_a, true_p_a))
    successes_b = int(np.random.binomial(n_b, true_p_b))
    p_a = successes_a / n_a
    p_b = successes_b / n_b
    print(f"Observed rates: A={p_a:.4f} (n={n_a}), B={p_b:.4f} (n={n_b})")

    # ---- 3. Sample Ratio Mismatch check ----
    srm_result = sample_ratio_mismatch_check([n_a, n_b], [0.5, 0.5], alpha)
    if isinstance(srm_result, dict):
        srm_stat = srm_result.get('chi_square', srm_result.get('statistic', 0.0))
        srm_flag = srm_result.get('srm_detected', srm_result.get('mismatch_detected', False))
    elif isinstance(srm_result, tuple):
        if len(srm_result) == 3:
            srm_stat, _srm_p, srm_flag = srm_result
        elif len(srm_result) == 2:
            srm_stat, srm_flag = srm_result
        else:
            srm_stat = srm_result[0]
            srm_flag = srm_result[-1]
    else:
        srm_stat, srm_flag = srm_result, False
    print(f"SRM chi-square={srm_stat:.4f}, mismatch_detected={srm_flag}")

    # ---- 4. Two-proportion z-test on the primary metric ----
    pooled_p = pooled_proportion(successes_a, n_a, successes_b, n_b)
    pooled_se = pooled_standard_error(pooled_p, n_a, n_b)
    z_stat = two_proportion_z_statistic(p_a, p_b, pooled_se)
    p_value = two_sided_p_value(z_stat)
    print(f"Primary test: z={z_stat:.4f}, p={p_value:.4f}")

    unpooled_se = unpooled_standard_error(successes_a, n_a, successes_b, n_b)
    ci_low, ci_high = confidence_interval_from_se(p_b - p_a, unpooled_se, 0.95)
    print(f"95% CI for lift (B-A): [{ci_low:.4f}, {ci_high:.4f}]")

    achieved_power = statistical_power(n_a, baseline_rate, mde, alpha)
    print(f"Achieved power at MDE: {achieved_power:.4f}")

    # ---- 5. Guardrail metrics with multiple-testing correction ----
    guardrail_p_values = [0.012, 0.20, 0.045, 0.6]
    bonf_flags = bonferroni_correction(guardrail_p_values, alpha)
    bh_flags = benjamini_hochberg_correction(guardrail_p_values, alpha)
    print(f"Guardrail p-values: {guardrail_p_values}")
    print(f"Bonferroni significant: {bonf_flags}")
    print(f"BH-FDR significant:    {bh_flags}")

    # ---- 6. Difference-in-differences on a secondary rollout ----
    treated_pre = np.array([10.0, 10.5, 11.0, 10.8])
    treated_post = np.array([12.5, 13.0, 12.8, 13.2])
    control_pre = np.array([9.8, 10.1, 10.3, 10.0])
    control_post = np.array([10.4, 10.7, 10.9, 10.6])
    did_simple = difference_in_differences_simple(
        treated_pre, treated_post, control_pre, control_post
    )
    print(f"DiD (simple means): {did_simple:.4f}")

    outcomes = np.concatenate([treated_pre, treated_post, control_pre, control_post])
    treat_ind = np.array([1] * 4 + [1] * 4 + [0] * 4 + [0] * 4)
    post_ind = np.array([0] * 4 + [1] * 4 + [0] * 4 + [1] * 4)
    did_reg = did_effect_from_regression(treat_ind, post_ind, outcomes)
    print(f"DiD (OLS interaction coef): {did_reg:.4f}")

    # ---- 7. Synthetic control for a single treated unit ----
    n_pre, n_post = 20, 10
    time = np.arange(n_pre + n_post)
    donor1 = 5 + 0.10 * time + np.random.normal(0, 0.1, n_pre + n_post)
    donor2 = 4 + 0.15 * time + np.random.normal(0, 0.1, n_pre + n_post)
    donor3 = 6 + 0.05 * time + np.random.normal(0, 0.1, n_pre + n_post)
    donors = np.vstack([donor1, donor2, donor3]).T  # (T, n_donors)
    true_weights = np.array([0.5, 0.3, 0.2])
    treated_series = donors @ true_weights + np.random.normal(0, 0.05, n_pre + n_post)
    treated_series[n_pre:] += 1.0  # true treatment effect = 1.0

    donor_pre = donors[:n_pre]
    donor_post = donors[n_pre:]
    treated_pre_s = treated_series[:n_pre]
    treated_post_s = treated_series[n_pre:]

    weights = fit_synthetic_control_weights(
        treated_pre_s, donor_pre, num_iterations=3000, learning_rate=0.01
    )
    print(f"Synthetic control weights: {np.round(weights, 3)}")
    sc_result = synthetic_control_effect(treated_post_s, donor_post, weights)
    if isinstance(sc_result, dict):
        sc_effect = sc_result.get('average_effect', sc_result.get('effect', 0.0))
    else:
        sc_effect = float(sc_result)
    print(f"Synthetic control average post-period effect: {sc_effect:.4f}")

    # ---- 8. Final ship decision ----
    primary_result = {
        "p_value": p_value,
        "effect": p_b - p_a,
        "alpha": alpha,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "significant": p_value < alpha and (p_b - p_a) > 0,
    }
    guardrail_results = [
        {
            "p_value": gp,
            "alpha": alpha,
            "effect": 0.0,
            "significant": bool(flag),
            "direction": "harm",
            "harm_direction": "negative",
        }
        for gp, flag in zip(guardrail_p_values, bh_flags)
    ]
    decision = ship_decision(primary_result, guardrail_results)
    print(f"Ship decision: {decision}")


if __name__ == "__main__":
    main()
