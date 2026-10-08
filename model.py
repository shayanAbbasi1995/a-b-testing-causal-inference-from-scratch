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

