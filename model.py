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

