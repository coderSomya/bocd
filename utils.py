"""
utils.py
========
Utility functions for BOCD experiments.
 
Contains:
  - Synthetic data generators  (Gaussian, Poisson, Bernoulli, multivariate)
  - Changepoint detection helpers
  - Evaluation metrics (F1, precision, recall, detection delay)
  - Posterior analysis tools
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



def detect_changepoints(
    changepoint_scores: list[float],
    threshold: float = 0.5,
    min_gap: int = 10
) -> list[int]:
    """Extract changepoint locations from the sequence of CP scores.
 
    A changepoint is detected when P(rₜ=0 | x₁:ₜ) > threshold,
    with a minimum gap between consecutive detections.
 
    Parameters
    ----------
    changepoint_scores : P(rₜ=0 | x₁:ₜ) for each t
    threshold          : probability threshold for detection
    min_gap            : minimum timesteps between two detections
 
    Returns
    -------
    detected : list of timestep indices where changepoints are declared
    """
    detected = []
    last_t = -min_gap
    for t, score in enumerate(changepoint_scores):
        if score > threshold and (t - last_t) >= min_gap:
            detected.append(t)
            last_t = t
    return detected
 
 
def get_map_run_length(run_length_dist, run_length_vals) -> np.ndarray:
    """Return the MAP run length at each timestep.
 
    Parameters
    ----------
    run_length_dist : list of arrays, P(rₜ | x₁:ₜ) for each t
    run_length_vals : list of arrays, corresponding r values
 
    Returns
    -------
    map_rl : np.ndarray of shape (T,), MAP run length at each t
    """
    map_rl = []
    for probs, rs in zip(run_length_dist, run_length_vals):
        map_rl.append(rs[np.argmax(probs)])
    return np.array(map_rl)
 
 
def build_run_length_matrix(
    run_length_dist: list[np.ndarray],
    run_length_vals: list[np.ndarray],
    T: Optional[int] = None
) -> np.ndarray:
    """Build a dense (T × T) matrix for plotting the run-length distribution.
 
    R[t, r] = P(run length = r at time t | x₁:ₜ).
    Entries outside the active hypotheses are 0.
 
    Parameters
    ----------
    run_length_dist : list of length T, each entry = P(rₜ | x₁:ₜ)
    run_length_vals : list of length T, each entry = corresponding r values
    T               : number of timesteps (default: len(run_length_dist))
 
    Returns
    -------
    R : np.ndarray of shape (T, T)
    """
    T = T or len(run_length_dist)
    R = np.zeros((T, T))
    for t, (probs, rs) in enumerate(zip(run_length_dist, run_length_vals)):
        for p, r in zip(probs, rs):
            if r < T:
                R[t, r] = p
    return R
 
 
# ---------------------------------------------------------------------------
# EVALUATION METRICS
# ---------------------------------------------------------------------------
 
def detection_delay(
    true_cps: list[int],
    detected_cps: list[int],
    window: int = 20
) -> list[float]:
    """Compute detection delay for each true changepoint.
 
    For each true changepoint, the delay is the number of timesteps
    from the true location to the nearest detected changepoint within
    a tolerance window. Returns np.inf if no detection within window.
 
    Parameters
    ----------
    true_cps    : list of true changepoint locations (0-indexed)
    detected_cps: list of detected changepoint locations
    window      : maximum gap to consider a detection "correct"
 
    Returns
    -------
    delays : list of floats, one per true changepoint
    """
    delays = []
    for tcp in true_cps[1:]:   # skip the first (t=0 is not a "change")
        best_delay = np.inf
        for dcp in detected_cps:
            if 0 <= dcp - tcp <= window:
                best_delay = min(best_delay, dcp - tcp)
        delays.append(best_delay)
    return delays
 
 
def evaluate_detection(
    true_cps: list[int],
    detected_cps: list[int],
    T: int,
    window: int = 20
) -> dict:
    """Compute precision, recall, F1, and mean detection delay.
 
    A detected changepoint is a True Positive if it falls within
    `window` timesteps after a true changepoint.
 
    Parameters
    ----------
    true_cps     : true changepoint locations (skip index 0)
    detected_cps : detected changepoint locations
    T            : total number of timesteps
    window       : tolerance window for matching
 
    Returns
    -------
    metrics : dict with keys:
        precision, recall, f1, mean_delay, n_true, n_detected, n_tp
    """
    real_cps = set(true_cps[1:])  # exclude t=0
    tp = 0
    matched_true = set()
 
    for dcp in detected_cps:
        for tcp in real_cps:
            if 0 <= dcp - tcp <= window and tcp not in matched_true:
                tp += 1
                matched_true.add(tcp)
                break
 
    fp = len(detected_cps) - tp
    fn = len(real_cps) - tp
 
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
 
    delays = detection_delay(true_cps, detected_cps, window)
    mean_delay = np.mean([d for d in delays if d < np.inf]) if delays else np.inf
 
    return {
        "precision":  round(precision, 4),
        "recall":     round(recall, 4),
        "f1":         round(f1, 4),
        "mean_delay": round(float(mean_delay), 2) if mean_delay < np.inf else np.inf,
        "n_true":     len(real_cps),
        "n_detected": len(detected_cps),
        "n_tp":       tp,
    }
 
 
# ---------------------------------------------------------------------------
# POSTERIOR ANALYSIS HELPERS
# ---------------------------------------------------------------------------
 
def posterior_mean_run_length(
    run_length_dist: list[np.ndarray],
    run_length_vals: list[np.ndarray]
) -> np.ndarray:
    """Compute E[rₜ | x₁:ₜ] at each timestep.
 
    Returns
    -------
    mean_rl : np.ndarray of shape (T,)
    """
    means = []
    for probs, rs in zip(run_length_dist, run_length_vals):
        means.append(float(np.dot(probs, rs)))
    return np.array(means)
 
 
def posterior_credible_interval(
    run_length_dist: list[np.ndarray],
    run_length_vals: list[np.ndarray],
    alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (1-alpha) credible interval for rₜ at each timestep.
 
    Returns
    -------
    lower, upper : np.ndarrays of shape (T,)
    """
    lowers, uppers = [], []
    for probs, rs in zip(run_length_dist, run_length_vals):
        sorted_idx = np.argsort(rs)
        sorted_rs = rs[sorted_idx]
        sorted_probs = probs[sorted_idx]
        cdf = np.cumsum(sorted_probs)
        lower_idx = np.searchsorted(cdf, alpha / 2)
        upper_idx = np.searchsorted(cdf, 1 - alpha / 2)
        lower_idx = min(lower_idx, len(sorted_rs) - 1)
        upper_idx = min(upper_idx, len(sorted_rs) - 1)
        lowers.append(sorted_rs[lower_idx])
        uppers.append(sorted_rs[upper_idx])
    return np.array(lowers), np.array(uppers)
