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

