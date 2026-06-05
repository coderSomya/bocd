"""
utils.py
========
Utility functions for BOCD experiments.
 
Contains:
  - Synthetic data generators  (Gaussian, Poisson, Bernoulli, multivariate)
"""
 
import numpy as np
from typing import Optional
 
 
# ---------------------------------------------------------------------------
# SYNTHETIC DATA GENERATORS
# ---------------------------------------------------------------------------
 
def generate_gaussian_data(
    means: list[float],
    stds: list[float],
    lengths: list[int],
    noise: float = 0.0,
    seed: Optional[int] = None
) -> tuple[np.ndarray, list[int]]:
    """Generate piecewise Gaussian data with known changepoints.
 
    Each segment has its own mean and standard deviation.
    Data within a segment is i.i.d. N(mean, std²).
 
    Parameters
    ----------
    means   : list of means, one per segment
    stds    : list of std devs, one per segment
    lengths : list of segment lengths
    noise   : optional extra Gaussian noise added to all data
    seed    : random seed for reproducibility
 
    Returns
    -------
    data        : np.ndarray of shape (T,)
    changepoints : list of timesteps where a new segment starts
                  (0-indexed, always includes 0 as first segment start)
 
    Example
    -------
    >>> data, cps = generate_gaussian_data(
    ...     means=[0, 5, -3],
    ...     stds=[1, 1, 1],
    ...     lengths=[100, 100, 100]
    ... )
    """
    if not (len(means) == len(stds) == len(lengths)):
        raise ValueError("means, stds, and lengths must have the same length.")
    rng = np.random.default_rng(seed)
    segments = []
    changepoints = [0]
    pos = 0
    for i, (mu, sigma, n) in enumerate(zip(means, stds, lengths)):
        segment = rng.normal(mu, sigma, size=n)
        segment += rng.normal(0, noise, size=n)
        segments.append(segment)
        pos += n
        if i < len(lengths) - 1:
            changepoints.append(pos)
    return np.concatenate(segments), changepoints
 
 
def generate_poisson_data(
    rates: list[float],
    lengths: list[int],
    seed: Optional[int] = None
) -> tuple[np.ndarray, list[int]]:
    """Generate piecewise Poisson count data.
 
    Parameters
    ----------
    rates   : list of Poisson rates λ, one per segment
    lengths : list of segment lengths
    seed    : random seed
 
    Returns
    -------
    data         : np.ndarray of integer counts, shape (T,)
    changepoints : list of segment start indices (0-indexed)
    """
    rng = np.random.default_rng(seed)
    segments = []
    changepoints = [0]
    pos = 0
    for i, (lam, n) in enumerate(zip(rates, lengths)):
        segments.append(rng.poisson(lam, size=n))
        pos += n
        if i < len(lengths) - 1:
            changepoints.append(pos)
    return np.concatenate(segments).astype(float), changepoints
 
 
def generate_bernoulli_data(
    probs: list[float],
    lengths: list[int],
    seed: Optional[int] = None
) -> tuple[np.ndarray, list[int]]:
    """Generate piecewise Bernoulli (0/1) data.
 
    Parameters
    ----------
    probs   : list of success probabilities p, one per segment
    lengths : list of segment lengths
    seed    : random seed
 
    Returns
    -------
    data         : np.ndarray of 0s and 1s, shape (T,)
    changepoints : list of segment start indices
    """
    rng = np.random.default_rng(seed)
    segments = []
    changepoints = [0]
    pos = 0
    for i, (p, n) in enumerate(zip(probs, lengths)):
        segments.append(rng.binomial(1, p, size=n).astype(float))
        pos += n
        if i < len(lengths) - 1:
            changepoints.append(pos)
    return np.concatenate(segments), changepoints
 
 
def generate_multivariate_gaussian_data(
    means: list[np.ndarray],
    covs: list[np.ndarray],
    lengths: list[int],
    seed: Optional[int] = None
) -> tuple[np.ndarray, list[int]]:
    """Generate piecewise multivariate Gaussian data.
 
    Parameters
    ----------
    means   : list of (d,) mean vectors, one per segment
    covs    : list of (d,d) covariance matrices, one per segment
    lengths : list of segment lengths
    seed    : random seed
 
    Returns
    -------
    data         : np.ndarray of shape (T, d)
    changepoints : list of segment start indices
    """
    rng = np.random.default_rng(seed)
    segments = []
    changepoints = [0]
    pos = 0
    for i, (mu, cov, n) in enumerate(zip(means, covs, lengths)):
        segments.append(rng.multivariate_normal(mu, cov, size=n))
        pos += n
        if i < len(lengths) - 1:
            changepoints.append(pos)
    return np.vstack(segments), changepoints
 
 
def generate_variance_change_data(
    means: list[float],
    stds: list[float],
    lengths: list[int],
    seed: Optional[int] = None
) -> tuple[np.ndarray, list[int]]:
    """Generate data where only the variance changes (mean stays constant).
 
    Useful for testing models that track variance (GaussianUnknownMeanVar).
    """
    return generate_gaussian_data(means, stds, lengths, seed=seed)
 
