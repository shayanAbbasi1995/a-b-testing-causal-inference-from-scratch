"""
A/B Testing & Causal Inference Toolkit

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - standard_normal_cdf
import math
import numpy as np

def standard_normal_cdf(z):
    # TODO: return P(Z <= z) for a standard normal Z, supporting float or numpy array input
    erf = np.vectorize(math.erf)
    norm_cdf = 0.5 * (1 + erf(z/np.sqrt(2)))

    return norm_cdf

# Step 2 - standard_normal_ppf
import math
from statistics import NormalDist

def standard_normal_ppf(p):
    """Return z such that Phi(z) = p for p in (0, 1)."""
    # TODO: implement a rational approximation to the inverse standard normal CDF
    if (p<0 or p>1):
        return float("nan")
    
    if p==0:
        return -float(inf)
        
    if p==1:
        return float(inf)

    z_score = NormalDist().inv_cdf(p)

    return z_score

# Step 3 - pooled_proportion
def pooled_proportion(successes_a, total_a, successes_b, total_b):
    # TODO: Compute the pooled success proportion across two groups for the null of equal rates.
    p_hat = (successes_a + successes_b) / (total_a + total_b)

    return p_hat

# Step 4 - pooled_standard_error
import math

def pooled_standard_error(pooled_p, total_a, total_b):
    """Standard error of the difference in two proportions under the pooled null."""
    # TODO: compute sqrt( p*(1-p) * (1/n_a + 1/n_b) ) using the pooled proportion.
    pooled_p = math.sqrt(pooled_p * (1-pooled_p) * ((1/total_a) + (1/total_b)))
    
    return pooled_p

# Step 5 - two_proportion_z_statistic
def two_proportion_z_statistic(p_a, p_b, pooled_se):
    z_score = (p_b - p_a) / pooled_se
    return z_score

# Step 6 - two_sided_p_value
def two_sided_p_value(z):
    # TODO: convert a z-statistic into a two-sided p-value under the standard normal
    p_value = 2 * (1 - standard_normal_cdf(np.abs(z)))
    return p_value

# Step 7 - unpooled_standard_error
def unpooled_standard_error(successes_a, total_a, successes_b, total_b):
    # TODO: return the unpooled SE of the difference between two sample proportions.
    if successes_a < 0 or successes_b < 0 or total_a <= 0 or total_b <= 0:
        return float("nan")
    
    if successes_a > total_a or successes_b > total_b:
        return float("nan")

    p_a = successes_a/total_a
    p_b = successes_b/total_b

    unpooled_se = math.sqrt(((p_a * (1 - p_a)) / total_a) + ((p_b * (1 - p_b)) / total_b))
    return unpooled_se

# Step 8 - confidence_interval_from_se
def confidence_interval_from_se(point_estimate, standard_error, confidence_level):
    # TODO: build a two-sided normal-approximation CI (lower, upper) from estimate and SE
    if confidence_level >= 100 or confidence_level <= 0:
        return float("nan")
    
    alpha = 1-confidence_level
    critical_z = standard_normal_ppf(1 - (alpha/2))
    lo = point_estimate - (critical_z * standard_error)
    hi = point_estimate + (critical_z * standard_error)

    return (round(lo,4), round(hi,4))

# Step 9 - required_sample_size_per_variant
import math
import statistics

def required_sample_size_per_variant(baseline_rate, minimum_detectable_effect, alpha, power):
    # TODO: return the minimum per-variant sample size for a two-proportion z-test at given alpha and power.
    if baseline_rate <= 0 or baseline_rate >= 1:
        raise ValueError("baseline_rate must be between 0 and 1 exclusive.")
    if minimum_detectable_effect == 0:
        raise ValueError("minimum_detectable_effect cannot be zero.")

    p1 = baseline_rate
    p2 = baseline_rate + minimum_detectable_effect
    
    if p2 <= 0 or p2 >= 1:
        raise ValueError("Resulting variant rate (baseline + MDE) must be between 0 and 1.")

    p_bar = (p1 + p2) / 2.0
    
    z_alpha = standard_normal_ppf(1.0 - (alpha / 2.0))
    z_beta = standard_normal_ppf(power)
    
    term1 = z_alpha * math.sqrt(2.0 * p_bar * (1.0 - p_bar))
    term2 = z_beta * math.sqrt((p1 * (1.0 - p1)) + (p2 * (1.0 - p2)))
    
    numerator = (term1 + term2) ** 2
    denominator = (p1 - p2) ** 2
    
    n = numerator / denominator
    
    return math.ceil(n)

# Step 10 - statistical_power
import math

def statistical_power(sample_size_per_variant, baseline_rate, effect_size, alpha):
    # TODO: return the power of a two-proportion z-test for the given design
    p1 = baseline_rate
    p2 = baseline_rate + effect_size
    n = sample_size_per_variant
    
    p_bar = (p1 + p2) / 2.0
    z_alpha = statistics.NormalDist().inv_cdf(1.0 - (alpha / 2.0))
    
    numerator = (abs(p1 - p2) * math.sqrt(n)) - (z_alpha * math.sqrt(2.0 * p_bar * (1.0 - p_bar)))
    denominator = math.sqrt((p1 * (1.0 - p1)) + (p2 * (1.0 - p2)))
    
    z_beta = numerator / denominator
    power = statistics.NormalDist().cdf(z_beta)
    
    return power

# Step 11 - chi_square_statistic
def chi_square_statistic(observed_counts, expected_counts):
    # TODO: return the Pearson chi-square statistic comparing observed to expected counts.
    if len(observed_counts) != len(expected_counts):
        raise ValueError("Observed and expected counts must have the same length.")
    
    chi_sq = 0.0
    for o, e in zip(observed_counts, expected_counts):
        if e == 0:
            raise ValueError("Expected counts cannot be zero.")
        chi_sq += ((o - e) ** 2) / e
        
    return chi_sq

# Step 12 - sample_ratio_mismatch_check
import math
import numpy as np

def sample_ratio_mismatch_check(observed_counts, expected_ratios, alpha):
    total_n = sum(observed_counts)
    expected_counts = [ratio * total_n for ratio in expected_ratios]
    
    chi_sq = chi_square_statistic(observed_counts, expected_counts)
    
    z = math.sqrt(chi_sq)
    p_value = 2.0 * (1.0 - standard_normal_cdf(z))
    
    srm_detected = p_value < alpha
    
    return {
        'chi_square': float(chi_sq),
        'p_value': float(p_value),
        'srm_detected': bool(srm_detected)
    }

# Step 13 - bonferroni_correction
import numpy as np

def bonferroni_correction(p_values, alpha):
    p_vals = np.array(p_values, dtype=float)
    m = p_vals.size
    if m == 0:
        return np.array([], dtype=bool)
    
    adjusted_alpha = alpha / m
    return p_vals <= adjusted_alpha

# Step 14 - benjamini_hochberg_correction
import numpy as np

def benjamini_hochberg_correction(p_values, alpha):
    """Return a boolean array marking BH-significant hypotheses at level alpha."""
    p_vals = np.asarray(p_values, dtype=float)
    m = len(p_vals)
    if m == 0:
        return np.array([], dtype=bool)
    
    sorted_idx = np.argsort(p_vals)
    sorted_p = p_vals[sorted_idx]
    
    ranks = np.arange(1, m + 1)
    below_threshold = sorted_p <= (ranks / m) * alpha
    
    if not np.any(below_threshold):
        return np.zeros(m, dtype=bool)
    
    max_idx = np.max(np.where(below_threshold)[0])
    
    significant_sorted = np.zeros(m, dtype=bool)
    significant_sorted[:max_idx + 1] = True
    
    significant = np.zeros(m, dtype=bool)
    significant[sorted_idx] = significant_sorted
    
    return significant

# Step 15 - group_mean_change
def group_mean_change(pre_outcomes, post_outcomes):
    pre_arr = np.asarray(pre_outcomes, dtype=float)
    post_arr = np.asarray(post_outcomes, dtype=float)
    return float(np.mean(post_arr) - np.mean(pre_arr))

# Step 16 - difference_in_differences_simple
def difference_in_differences_simple(treated_pre, treated_post, control_pre, control_post):
    treated_change = group_mean_change(treated_pre, treated_post)
    control_change = group_mean_change(control_pre, control_post)
    return float(treated_change - control_change)

# Step 17 - build_did_design_matrix
def build_did_design_matrix(treatment_indicator, post_indicator):
    T = np.asarray(treatment_indicator, dtype=float)
    P = np.asarray(post_indicator, dtype=float)
    n = len(T)
    
    intercept = np.ones(n, dtype=float)
    interaction = T * P
    
    return np.column_stack([intercept, T, P, interaction])

# Step 18 - ols_normal_equations
def ols_normal_equations(design_matrix, outcomes):
    X = np.asarray(design_matrix, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    
    A = X.T @ X
    b = X.T @ y
    
    beta = np.linalg.solve(A, b)
    return beta

# Step 19 - did_effect_from_regression
def did_effect_from_regression(treatment_indicator, post_indicator, outcomes):
    X = build_did_design_matrix(treatment_indicator, post_indicator)
    beta = ols_normal_equations(X, outcomes)
    return float(beta[3])

# Step 20 - fit_synthetic_control_weights
def fit_synthetic_control_weights(treated_pre, donor_pre, num_iterations=5000, learning_rate=0.01):
    y = np.asarray(treated_pre, dtype=float)
    X = np.asarray(donor_pre, dtype=float)
    
    n_donors = X.shape[1]
    theta = np.zeros(n_donors, dtype=float)
    
    for _ in range(num_iterations):
        # Softmax
        shifted_theta = theta - np.max(theta)
        exp_theta = np.exp(shifted_theta)
        w = exp_theta / np.sum(exp_theta)
        
        r = (X @ w) - y
        g_w = X.T @ r
        
        # softmax Jacobian
        w_dot_gw = np.dot(w, g_w)
        g_theta = w * (g_w - w_dot_gw)
        
        # Gradient descent update
        theta -= learning_rate * g_theta
        
    shifted_theta = theta - np.max(theta)
    exp_theta = np.exp(shifted_theta)
    w = exp_theta / np.sum(exp_theta)
    
    return w

# Step 21 - synthetic_control_effect
def synthetic_control_effect(treated_post, donor_post, weights):
    y_post = np.asarray(treated_post, dtype=float)
    X_post = np.asarray(donor_post, dtype=float)
    w = np.asarray(weights, dtype=float)
    
    synthetic = X_post @ w
    
    gap = y_post - synthetic
    
    average_effect = float(np.mean(gap))
    
    return {
        'synthetic': synthetic,
        'gap': gap,
        'average_effect': average_effect
    }

# Step 22 - ship_decision
def ship_decision(primary_result, guardrail_results):

    if primary_result['p_value'] >= primary_result['alpha']:
        return {'ship': False, 'reason': 'primary metric not significant'}
    
    if primary_result['effect'] <= 0:
        return {'ship': False, 'reason': 'primary metric effect not positive'}
    
    for i, g in enumerate(guardrail_results):
        is_sig = g['p_value'] < g['alpha']
        if not is_sig:
            continue
            
        harm_dir = g['harm_direction']
        effect = g['effect']
        
        harmed = (harm_dir == 'negative' and effect < 0) or (harm_dir == 'positive' and effect > 0)
        
        if harmed:
            return {'ship': False, 'reason': f'guardrail {i} harmed'}
            
    return {'ship': True, 'reason': 'ship'}

