""" 
This module implements:
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


# ---- Model 1: Gaussian, Unknown Mean, Known Variance ----------------------
 
class GaussianUnknownMean(ProbabilisticModel):
    """Normal likelihood with known variance; Gaussian prior on mean.
 
    Generative model:
        θ  ~ N(μ₀, σ₀²)          [prior on the mean]
        xₜ ~ N(θ,  σ²)           [likelihood, σ² known]
 
    Because the prior and likelihood are both Gaussian, the posterior
    on θ and the predictive distribution are also Gaussian (closed form).
 
    Sufficient statistic: sum of observations in the current run.
 
    After n observations with sum S = Σxᵢ:
        posterior mean     μₙ = (μ₀/σ₀² + S/σ²) / (1/σ₀² + n/σ²)
        posterior variance σₙ² = 1 / (1/σ₀² + n/σ²)
        predictive         N(x ; μₙ, σ² + σₙ²)
    """
 
    def __init__(self, mu0: float = 0.0, sigma0: float = 1.0, sigma: float = 1.0):
        """
        Parameters
        ----------
        mu0    : prior mean
        sigma0 : prior standard deviation on the mean
        sigma  : known observation noise standard deviation
        """
        self.mu0 = mu0
        self.sigma0 = sigma0
        self.sigma = sigma
        # Sufficient statistics
        self._n = 0          # count of observations
        self._sum = 0.0      # sum of observations
 
    # --- Posterior parameters ---
    @property
    def _post_var(self) -> float:
        prec_prior = 1.0 / self.sigma0**2
        prec_like  = self._n / self.sigma**2
        return 1.0 / (prec_prior + prec_like)
 
    @property
    def _post_mean(self) -> float:
        prec_prior = 1.0 / self.sigma0**2
        prec_like  = 1.0 / self.sigma**2
        num = self.mu0 * prec_prior + self._sum * prec_like
        return num * self._post_var
 
    def log_pred(self, x) -> float:
        # Predictive: N(x ; μₙ, σ² + σₙ²)
        pred_var = self.sigma**2 + self._post_var
        return stats.norm.logpdf(x, self._post_mean, np.sqrt(pred_var))
 
    def update(self, x) -> None:
        self._n   += 1
        self._sum += x
 
    def reset(self) -> 'GaussianUnknownMean':
        return GaussianUnknownMean(self.mu0, self.sigma0, self.sigma)
 
    def __repr__(self):
        return (f"GaussianUnknownMean(mu0={self.mu0}, "
                f"sigma0={self.sigma0}, sigma={self.sigma})")
 
 
# ---- Model 2: Gaussian, Unknown Mean AND Variance (Normal-Inverse-Gamma) --
 
class GaussianUnknownMeanVar(ProbabilisticModel):
    """Normal likelihood with unknown mean AND variance.
 
    Conjugate prior: Normal-Inverse-Gamma
        σ² ~ InvGamma(α₀, β₀)
        μ  | σ² ~ N(m₀, σ²/κ₀)
 
    Sufficient statistics: n, Σx, Σx²
 
    After n observations:
        κₙ = κ₀ + n
        mₙ = (κ₀ m₀ + Σx) / κₙ
        αₙ = α₀ + n/2
        βₙ = β₀ + ½[Σx² + κ₀ m₀² - κₙ mₙ²]
 
    Predictive: Student-t with 2αₙ degrees of freedom,
        location mₙ, scale sqrt(βₙ(κₙ+1)/(αₙ κₙ))
    """
 
    def __init__(self, m0: float = 0.0, kappa0: float = 1.0,
                 alpha0: float = 1.0, beta0: float = 1.0):
        """
        Parameters
        ----------
        m0     : prior mean of the mean
        kappa0 : prior precision (number of "pseudo-observations")
        alpha0 : shape of the Inverse-Gamma prior on variance
        beta0  : scale of the Inverse-Gamma prior on variance
        """
        self.m0     = m0
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta0  = beta0
        self._n     = 0
        self._sum_x = 0.0
        self._sum_x2 = 0.0
 
    @property
    def _kappa(self): return self.kappa0 + self._n
 
    @property
    def _m(self):     return (self.kappa0 * self.m0 + self._sum_x) / self._kappa
 
    @property
    def _alpha(self): return self.alpha0 + self._n / 2.0
 
    @property
    def _beta(self):
        kn, mn = self._kappa, self._m
        return (self.beta0
                + 0.5 * (self._sum_x2 + self.kappa0 * self.m0**2 - kn * mn**2))
 
    def log_pred(self, x) -> float:
        # Predictive: t_{2α}(m, β(κ+1)/(ακ))
        df    = 2.0 * self._alpha
        loc   = self._m
        scale = np.sqrt(self._beta * (self._kappa + 1) / (self._alpha * self._kappa))
        return stats.t.logpdf(x, df, loc=loc, scale=scale)
 
    def update(self, x) -> None:
        self._n      += 1
        self._sum_x  += x
        self._sum_x2 += x**2
 
    def reset(self) -> 'GaussianUnknownMeanVar':
        return GaussianUnknownMeanVar(self.m0, self.kappa0, self.alpha0, self.beta0)
 
    def __repr__(self):
        return (f"GaussianUnknownMeanVar(m0={self.m0}, kappa0={self.kappa0}, "
                f"alpha0={self.alpha0}, beta0={self.beta0})")
 
 
# ---- Model 3: Student-T (equivalent marginal form) ------------------------
 
class StudentT(ProbabilisticModel):
    """Convenience wrapper: same model as GaussianUnknownMeanVar.
 
    The predictive from the Normal-Inverse-Gamma model is a Student-T.
    This class exposes the Student-T parameterisation directly, making
    the connection to the paper's Section 4.2 explicit.
 
    Parameters are in the NIG parameterisation (same as above).
    """
 
    def __init__(self, alpha: float = 1.0, beta: float = 1.0,
                 kappa: float = 1.0, mu: float = 0.0):
        self._model = GaussianUnknownMeanVar(m0=mu, kappa0=kappa,
                                             alpha0=alpha, beta0=beta)
        # Keep original hyperparameters for reset()
        self._alpha0 = alpha
        self._beta0  = beta
        self._kappa0 = kappa
        self._mu0    = mu
 
    def log_pred(self, x) -> float:
        return self._model.log_pred(x)
 
    def update(self, x) -> None:
        self._model.update(x)
 
    def reset(self) -> 'StudentT':
        return StudentT(self._alpha0, self._beta0, self._kappa0, self._mu0)
 
    def __repr__(self):
        return (f"StudentT(alpha={self._alpha0}, beta={self._beta0}, "
                f"kappa={self._kappa0}, mu={self._mu0})")
 
 
# ---- Model 4: Poisson, Gamma Prior on Rate --------------------------------
 
class PoissonGamma(ProbabilisticModel):
    """Poisson counts with Gamma conjugate prior on the rate λ.
 
    Generative model:
        λ  ~ Gamma(α₀, β₀)      [rate prior; β₀ is the rate parameter]
        xₜ ~ Poisson(λ)
 
    Conjugate update after n observations with sum S = Σxᵢ:
        α_n = α₀ + S
        β_n = β₀ + n
 
    Predictive: Negative-Binomial(r=αₙ, p=βₙ/(βₙ+1))
    """
 
    def __init__(self, alpha0: float = 1.0, beta0: float = 1.0):
        """
        Parameters
        ----------
        alpha0 : Gamma shape (prior count of events)
        beta0  : Gamma rate  (prior count of time units)
        """
        self.alpha0 = alpha0
        self.beta0  = beta0
        self._n     = 0
        self._sum   = 0.0
 
    @property
    def _alpha(self): return self.alpha0 + self._sum
 
    @property
    def _beta(self):  return self.beta0  + self._n
 
    def log_pred(self, x) -> float:
        # Negative-Binomial predictive (marginalising out λ)
        a, b = self._alpha, self._beta
        x    = int(x)
        # log NB(x; r=a, p=b/(b+1))
        log_p = (gammaln(a + x) - gammaln(a) - gammaln(x + 1)
                 + a * np.log(b / (b + 1))
                 + x * np.log(1.0 / (b + 1)))
        return log_p
 
    def update(self, x) -> None:
        self._n   += 1
        self._sum += x
 
    def reset(self) -> 'PoissonGamma':
        return PoissonGamma(self.alpha0, self.beta0)
 
    def __repr__(self):
        return f"PoissonGamma(alpha0={self.alpha0}, beta0={self.beta0})"
 
 
# ---- Model 5: Bernoulli, Beta Prior on p ----------------------------------
 
class BetaBernoulli(ProbabilisticModel):
    """Bernoulli data (0/1) with Beta conjugate prior on success probability p.
 
    Generative model:
        p  ~ Beta(α₀, β₀)
        xₜ ~ Bernoulli(p)
 
    Conjugate update after n observations with S successes:
        α_n = α₀ + S
        β_n = β₀ + (n - S)
 
    Predictive: Beta-Bernoulli (Polya urn):
        P(x=1) = αₙ / (αₙ + βₙ)
    """
 
    def __init__(self, alpha0: float = 1.0, beta0: float = 1.0):
        self.alpha0 = alpha0
        self.beta0  = beta0
        self._n_ones  = 0    # successes
        self._n_total = 0    # total observations
 
    @property
    def _alpha(self): return self.alpha0 + self._n_ones
 
    @property
    def _beta(self):  return self.beta0  + (self._n_total - self._n_ones)
 
    def log_pred(self, x) -> float:
        a, b = self._alpha, self._beta
        # log P(x | a, b) = log[a/(a+b)] if x=1, log[b/(a+b)] if x=0
        p = a / (a + b)
        return np.log(p) if x == 1 else np.log(1.0 - p)
 
    def update(self, x) -> None:
        self._n_total += 1
        self._n_ones  += int(x)
 
    def reset(self) -> 'BetaBernoulli':
        return BetaBernoulli(self.alpha0, self.beta0)
 
    def __repr__(self):
        return f"BetaBernoulli(alpha0={self.alpha0}, beta0={self.beta0})"
 
 
# ---- Model 6: Multivariate Gaussian, Known Covariance ---------------------
 
class MultivariateGaussian(ProbabilisticModel):
    """Multivariate Normal with known covariance; Gaussian prior on mean.
 
    Generative model (d-dimensional):
        θ  ~ N(μ₀, Σ₀)          [prior on mean vector]
        xₜ ~ N(θ,  Σ)           [likelihood, Σ known]
 
    Posterior after n observations with sum vector S = Σxᵢ:
        Σₙ = (Σ₀⁻¹ + n Σ⁻¹)⁻¹
        μₙ = Σₙ (Σ₀⁻¹ μ₀ + Σ⁻¹ S)
 
    Predictive: N(x ; μₙ, Σ + Σₙ)
    """
 
    def __init__(self, mu0: np.ndarray, sigma0: np.ndarray, sigma: np.ndarray):
        """
        Parameters
        ----------
        mu0    : (d,) prior mean vector
        sigma0 : (d,d) prior covariance on the mean
        sigma  : (d,d) known observation covariance
        """
        self.mu0    = np.asarray(mu0, dtype=float)
        self.sigma0 = np.asarray(sigma0, dtype=float)
        self.sigma  = np.asarray(sigma,  dtype=float)
        self.d      = len(mu0)
        # Precompute inverses
        self._sigma0_inv = np.linalg.inv(self.sigma0)
        self._sigma_inv  = np.linalg.inv(self.sigma)
        # Sufficient statistics
        self._n   = 0
        self._sum = np.zeros(self.d)
 
    @property
    def _post_cov(self) -> np.ndarray:
        return np.linalg.inv(self._sigma0_inv + self._n * self._sigma_inv)
 
    @property
    def _post_mean(self) -> np.ndarray:
        return self._post_cov @ (self._sigma0_inv @ self.mu0
                                 + self._sigma_inv @ self._sum)
 
    def log_pred(self, x) -> float:
        pred_cov = self.sigma + self._post_cov
        return stats.multivariate_normal.logpdf(x, self._post_mean, pred_cov)
 
    def update(self, x) -> None:
        self._n   += 1
        self._sum += np.asarray(x, dtype=float)
 
    def reset(self) -> 'MultivariateGaussian':
        return MultivariateGaussian(self.mu0, self.sigma0, self.sigma)
 
    def __repr__(self):
        return f"MultivariateGaussian(d={self.d})"

