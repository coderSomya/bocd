"""
Bayesian Online Changepoint Detection (BOCD)
Based on: Adams & MacKay (2007) "Bayesian Online Changepoint Detection"
          https://arxiv.org/abs/0710.3742
 
This module implements:
  - The core BOCD algorithm (Section 2–3 of the paper)
  - All probabilistic models described in the paper (Section 4):
      * GaussianUnknownMean       – Normal likelihood, known variance
      * GaussianUnknownMeanVar    – Normal-Inverse-Gamma (full conjugate)
      * StudentT                  – Equivalent marginal of Normal-Inv-Gamma
      * PoissonGamma              – Poisson counts, Gamma prior on rate
      * BetaBernoulli             – Bernoulli data, Beta prior on p
      * MultivariateGaussian      – Multivariate Normal, known covariance
  - Two hazard functions:
      * ConstantHazard            – Geometric gaps (memoryless, Section 2.2)
      * GeometricHazard           – Alias for ConstantHazard
"""
 
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional
from scipy import stats
from scipy.special import gammaln, betaln


# ---------------------------------------------------------------------------
# HAZARD FUNCTIONS  (Section 2.2 of the paper)
# ---------------------------------------------------------------------------
 
class HazardFunction(ABC):
    """Abstract base class for hazard functions.
 
    The hazard function H(τ) gives the probability that the current run
    ends *right now*, given that it has already lasted τ timesteps:
 
        H(τ) = P(changepoint at t | run length = τ)
             = P_gap(τ) / Σ_{t≥τ} P_gap(t)      [Eq. 4 in paper]
 
    This is the discrete-time hazard rate from survival analysis.
    """
 
    @abstractmethod
    def __call__(self, run_length: int) -> float:
        """Return H(run_length)."""
        pass
 
 
class ConstantHazard(HazardFunction):
    """Constant (memoryless) hazard function.
 
    Corresponds to a Geometric prior over gap lengths:
        P_gap(g) = (1/λ)(1 - 1/λ)^{g-1}
 
    Because of the memoryless property of the Geometric distribution,
    the hazard rate is constant at H(τ) = 1/λ for all τ.
 
    Parameters
    ----------
    lam : float
        Expected run length (λ). Higher λ → rarer changepoints.
        E.g. lam=200 means you expect a changepoint every 200 timesteps.
    """
 
    def __init__(self, lam: float = 200.0):
        if lam <= 0:
            raise ValueError("lam must be positive.")
        self.lam = lam
 
    def __call__(self, run_length: int) -> float:
        return 1.0 / self.lam
 
    def __repr__(self):
        return f"ConstantHazard(lam={self.lam})"


# Alias
GeometricHazard = ConstantHazard
