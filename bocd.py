# ---------------------------------------------------------------------------
# CORE BOCD ALGORITHM  (Section 2–3 of the paper)
# ---------------------------------------------------------------------------

from algorithm import * 

class BOCD:
    """Bayesian Online Changepoint Detection.
 
    Maintains the exact posterior over run lengths P(rₜ | x₁:ₜ)
    using the message-passing recursion from Adams & MacKay (2007).
 
    Core recursion (Eq. 3):
        P(rₜ, x₁:ₜ) = Σ_{r_{t-1}} P(rₜ | r_{t-1}) · π^r_t · P(r_{t-1}, x₁:ₜ₋₁)
 
    where:
        P(rₜ | r_{t-1}) = H     if rₜ = 0          (changepoint)
                        = 1 - H if rₜ = r_{t-1} + 1 (growth)
                        = 0     otherwise
        π^r_t           = P(xₜ | r_{t-1}, x^(r)_t)  (predictive)
 
    Parameters
    ----------
    model   : ProbabilisticModel
        The observation model (one of the 6 models above).
    hazard  : HazardFunction
        The hazard function H(τ). Default: ConstantHazard(lam=200).
    threshold : float
        Minimum probability mass to keep a hypothesis (pruning).
        Hypotheses below this are dropped. Default: 1e-4.
    """
 
    def __init__(self,
                 model: ProbabilisticModel,
                 hazard: Optional[HazardFunction] = None,
                 threshold: float = 1e-4):
        self.model_template = model
        self.hazard  = hazard if hazard is not None else ConstantHazard(200.0)
        self.threshold = threshold
 
        # --- Algorithm state ---
        # log_joint[i] = log P(r=i, x₁:t)   for each active hypothesis
        # models[i]    = model instance with sufficient stats for r=i
        # run_lengths  = the actual r values for each hypothesis
        self._log_joint:  list[float] = []
        self._models:     list[ProbabilisticModel] = []
        self._run_lengths: list[int]  = []
 
        # History for analysis and plotting
        self.run_length_dist: list[np.ndarray] = []  # P(rₜ | x₁:ₜ) at each t
        self.run_length_vals: list[np.ndarray] = []  # corresponding r values
        self.observations: list = []
        self.changepoint_scores: list[float] = []    # P(rₜ=0 | x₁:ₜ) at each t
        self.t = 0
 
    def update(self, x) -> np.ndarray:
        """Process one new observation and update the posterior.
 
        Parameters
        ----------
        x : scalar or array_like
            The new observation.
 
        Returns
        -------
        probs : np.ndarray
            The normalised posterior P(rₜ | x₁:ₜ).
            probs[i] = P(rₜ = run_lengths[i] | x₁:ₜ)
        """
        self.t += 1
        self.observations.append(x)
 
        # Step 0: At t=1, initialise with a single r=0 hypothesis
        if self.t == 1:
            self._log_joint   = [0.0]          # log P(r=0) = log 1 = 0
            self._models      = [self.model_template.reset()]
            self._run_lengths = [0]
 
        # ---------------------------------------------------------------
        # Step 1: Compute log π^r_t = log P(xₜ | r, x^(r)_t)
        #         for each current hypothesis
        # ---------------------------------------------------------------
        log_preds = np.array([m.log_pred(x) for m in self._models])
 
        # ---------------------------------------------------------------
        # Step 2 & 3: Compute new log joint for growth (r+1) and
        #             changepoint (r=0) hypotheses
        # ---------------------------------------------------------------
        log_H  = np.log(np.array([self.hazard(r) for r in self._run_lengths]))
        log_1mH = np.log(1.0 - np.array([self.hazard(r) for r in self._run_lengths]))
 
        # Growth: log P(r+1, x₁:t) = log P(r, x₁:t-1) + log π + log(1-H)
        new_log_joint_growth = (np.array(self._log_joint)
                                + log_preds + log_1mH)
 
        # Changepoint: log P(r=0, x₁:t) = logsumexp[log P(r, x₁:t-1) + log π + log H]
        new_log_joint_cp = np.logaddexp.reduce(
            np.array(self._log_joint) + log_preds + log_H
        )
 
        # ---------------------------------------------------------------
        # Step 4: Normalise to get P(rₜ | x₁:ₜ)
        # ---------------------------------------------------------------
        # Combine: r=0 (changepoint) and r=1,2,...  (growth)
        all_log_joints = np.concatenate([[new_log_joint_cp], new_log_joint_growth])
        all_run_lengths = [0] + [r + 1 for r in self._run_lengths]
 
        # Normalise in log-space (numerically stable)
        log_norm = np.logaddexp.reduce(all_log_joints)
        log_probs = all_log_joints - log_norm
        probs = np.exp(log_probs)
 
        # ---------------------------------------------------------------
        # Step 5: Update sufficient statistics for growth hypotheses
        #         Reset model for the changepoint (r=0) hypothesis
        # ---------------------------------------------------------------
        # For r=0 hypothesis: fresh model
        cp_model = self.model_template.reset()
 
        # For growth hypotheses: update each model with x
        updated_growth_models = []
        for m in self._models:
            m_new = m.reset()                     # copy prior structure
            # Copy sufficient statistics then update with x
            # (we re-create to keep immutability)
            m_copy = self._copy_model_with_update(m, x)
            updated_growth_models.append(m_copy)
 
        # ---------------------------------------------------------------
        # Step 6: Prune hypotheses with negligible probability
        # ---------------------------------------------------------------
        all_models = [cp_model] + updated_growth_models
        mask = probs > self.threshold
        # Always keep at least the top-3 hypotheses
        if mask.sum() < 3:
            top_k = np.argsort(probs)[-3:]
            mask[top_k] = True
 
        probs      = probs[mask]
        all_log_joints = all_log_joints[mask]
        all_run_lengths = [all_run_lengths[i] for i in range(len(mask)) if mask[i]]
        all_models      = [all_models[i]      for i in range(len(mask)) if mask[i]]
 
        # Re-normalise after pruning
        probs /= probs.sum()
        log_probs = np.log(np.clip(probs, 1e-300, None))
 
        # ---------------------------------------------------------------
        # Store state
        # ---------------------------------------------------------------
        self._log_joint   = list(log_probs)
        self._models      = all_models
        self._run_lengths = all_run_lengths
 
        # Record history
        self.run_length_dist.append(probs.copy())
        self.run_length_vals.append(np.array(all_run_lengths))
        # P(changepoint at t) = P(rₜ=0 | x₁:ₜ)
        cp_score = probs[all_run_lengths.index(0)] if 0 in all_run_lengths else 0.0
        self.changepoint_scores.append(float(cp_score))
 
        return probs
 
    def _copy_model_with_update(self, model: ProbabilisticModel, x) -> ProbabilisticModel:
        """Create a copy of model with sufficient stats updated by x."""
        import copy
        m = copy.deepcopy(model)
        m.update(x)
        return m
 
    def most_likely_run_length(self) -> int:
        """Return the MAP estimate of the current run length."""
        if not self._run_lengths:
            return 0
        idx = int(np.argmax(self._log_joint))
        return self._run_lengths[idx]
 
    def changepoint_probability(self) -> float:
        """Return P(rₜ = 0 | x₁:ₜ) — probability a changepoint just occurred."""
        return self.changepoint_scores[-1] if self.changepoint_scores else 0.0
 
    def run_batch(self, data) -> list[np.ndarray]:
        """Process an entire dataset at once.
 
        Parameters
        ----------
        data : array_like of shape (T,) or (T, d)
 
        Returns
        -------
        list of length T, each element = P(rₜ | x₁:ₜ)
        """
        return [self.update(x) for x in data]
 
    def reset(self) -> None:
        """Reset the detector to its initial state."""
        self._log_joint   = []
        self._models      = []
        self._run_lengths = []
        self.run_length_dist     = []
        self.run_length_vals     = []
        self.observations        = []
        self.changepoint_scores  = []
        self.t = 0
 
