"""Step-function accumulator models for neural decision-making.

This module implements HMM-based models in which a neural population switches
from a low-activity ``baseline`` state to a high-activity ``step`` state at a
single, latent change-point within a trial.  The step time is governed by a
geometric waiting-time distribution parameterised by a per-trial (or global)
step probability.

Firing rates are modelled as either trial-specific Poisson rates
(:class:`TrialPoissonObservations`) or as Gamma-distributed rates with Poisson
spiking (:class:`GammaDistributionObservations`).

"""

import autograd.numpy as np
import autograd.numpy.random as npr
import ssm.stats as stats
from autograd.scipy.special import logsumexp
from ssm.hmm import HMM
from ssm.init_state_distns import InitialStateDistribution
from ssm.observations import PoissonObservations
from ssm.transitions import StationaryTransitions, Transitions


class TrialPoissonObservations(PoissonObservations):
    """Per-trial Poisson observations for a K-state step HMM.

    Maintains an independent log-rate vector for every ``(state, neuron,
    trial)`` combination.  At each time bin the spike count of neuron ``n``
    under state ``k`` on trial ``tag`` is modelled as:

    .. math::
        y_{tn} \\sim \\mathrm{Poisson}(\\lambda_{kn,\\mathrm{tag}} \\cdot \\Delta)

    where :math:`\\Delta` is the bin size in seconds and
    :math:`\\lambda_{kn,\\mathrm{tag}}` is the per-trial firing rate.

    Args:
        K (int): Number of discrete latent states.
        D (int): Number of observed neurons (emission dimensionality).
        M (int): External input dimensionality. Defaults to 0.
        num_trials (int): Total number of trials. One log-rate vector is
            allocated per trial. Defaults to 1.
        bin_size (float): Spike-count bin width in seconds, used to scale
            Poisson rates. Defaults to 1.0.

    Attributes:
        bin_size (float): Bin width in seconds.
        log_lambdas (np.ndarray): Log firing rates of shape
            ``(K, D, num_trials)``.  Initialised to standard-normal samples
            minus ``log(bin_size)``.
    """

    def __init__(self, K, D, M=0, num_trials=1, bin_size=1.0):
        super(PoissonObservations, self).__init__(K, D, M)

        # one set of lambdas for each trial
        self.bin_size = bin_size
        self.log_lambdas = npr.randn(K, D, num_trials) - np.log(self.bin_size)

    def log_likelihoods(self, data, input, mask, tag):
        """Compute per-time-bin log-likelihoods under the Poisson model.

        Uses the trial-specific firing rates indexed by ``tag`` to evaluate
        the Poisson log-probability for every time bin and state.

        Args:
            data (np.ndarray): Observed spike counts, shape ``(T, D)``.
            input (np.ndarray): External inputs (unused). Shape ``(T, M)``.
            mask (np.ndarray or None): Boolean mask of shape ``(T, D)``.
                If ``None``, all observations are treated as valid.
            tag (int): Trial index used to select the appropriate
                ``log_lambdas[:, :, tag]`` slice.

        Returns:
            np.ndarray: Log-likelihood array of shape ``(T, K)``.
        """
        lambdas = np.exp(self.log_lambdas[:, :, tag]) * self.bin_size
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        return stats.poisson_logpdf(data[:, None, :], lambdas, mask=mask[:, None, :])

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        """Sample a spike-count observation from state ``z`` on trial ``tag``.

        Args:
            z (int): Current discrete latent state index.
            xhist (np.ndarray): History of previous observations (unused).
            input (np.ndarray, optional): External input (unused). Defaults
                to ``None``.
            tag (int, optional): Trial index used to select firing rates.
                Defaults to ``None``.
            with_noise (bool): Whether to add Poisson noise. If ``False``,
                the mean rate is returned. Defaults to ``True``.

        Returns:
            np.ndarray: Sampled spike counts of shape ``(D,)``.
        """
        lambdas = np.exp(self.log_lambdas[:, :, tag]) * self.bin_size
        return npr.poisson(lambdas[z])

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        """Update log firing rates via a weighted maximum-likelihood M-step.

        For each trial ``tag``, the per-state log-rate is updated to the
        log of the posterior-weighted mean spike count:

        .. math::
            \\log \\hat{\\lambda}_{k,\\mathrm{tag}} =
                \\log\\left(\\frac{\\sum_t w_{tk} y_t}{\\sum_t w_{tk}}\\right)

        where ``w_{tk}`` is the posterior state probability at time ``t``
        under state ``k``.

        Args:
            expectations (list[tuple]): Per-trial E-step outputs. Each
                element is a tuple whose first entry is the ``(T, K)``
                marginal state-probability array.
            datas (list[np.ndarray]): Observed spike-count arrays, each of
                shape ``(T_i, D)``.
            inputs (list[np.ndarray]): External input arrays (unused).
            masks (list[np.ndarray]): Observation masks (unused).
            tags (list[int]): Trial indices, one per element of ``datas``.
            **kwargs: Additional keyword arguments (unused).
        """
        for expectation, data, input, tag in zip(expectations, datas, inputs, tags):
            weights = expectation[0]
            for k in range(self.K):
                weighted_data = np.log(
                    np.average(data, axis=0, weights=weights[:, k]) + 1e-16
                )
                self.log_lambdas[k, :, tag] = weighted_data / self.bin_size

    def smooth(self, expectations, data, input, tag):
        """Compute the posterior mean observation.

        Computes the expected spike count at each time bin under the
        posterior distribution over latent discrete states:

        .. math::
            \\hat{y}_t = \\sum_k p(z_t = k \\mid y_{1:T}) \\cdot
                         \\lambda_{k,\\mathrm{tag}} \\cdot \\Delta

        Args:
            expectations (np.ndarray): Marginal state probabilities of shape
                ``(T, K)``.
            data (np.ndarray): Observed spike counts of shape ``(T, D)``
                (unused).
            input (np.ndarray): External inputs (unused).
            tag (int): Trial index used to select firing rates.

        Returns:
            np.ndarray: Posterior mean spike counts, shape ``(T, D)``.
        """
        return expectations.dot(np.exp(self.log_lambdas[:, :, tag]))


class GammaDistributionObservations:
    """Gamma-Poisson compound observations for a K-state HMM.

    Models each state's firing rate as a Gamma-distributed random variable,
    and the observed spike count as Poisson-distributed given that rate:

    .. math::
        \\lambda_{kn} \\sim \\mathrm{Gamma}(\\alpha_{kn}, \\beta_{kn}), \\quad
        y_{tn} \\mid z_t = k \\sim \\mathrm{Poisson}(\\lambda_{kn})

    The shape (``alpha``) and rate (``beta``) parameters are stored in log
    space for unconstrained optimisation.

    Args:
        K (int): Number of discrete latent states.
        D (int): Number of observed neurons.
        M (int): External input dimensionality. Defaults to 0.

    Attributes:
        inv_alphas (np.ndarray): Log shape parameters of shape ``(K, D)``,
            initialised to ``log(1)`` (i.e., shape = 1 for all states).
        inv_betas (np.ndarray): Log rate parameters of shape ``(K, D)``,
            initialised to ``log(0.01)``.
    """

    def __init__(self, K, D, M=0):
        super().__init__(K, D, M)

        # one gamma distribution per state
        # TODO -> change init?
        self.inv_alphas = np.log(np.ones((K, D)))
        self.inv_betas = np.log(0.01 * np.ones((K, D)))

    @property
    def params(self):
        """tuple: Learnable parameters ``(inv_alphas, inv_betas)``."""
        return self.inv_alphas, self.inv_betas

    @params.setter
    def params(self, value):
        """Set the log shape and log rate parameters.

        Args:
            value (tuple): 2-tuple ``(inv_alphas, inv_betas)``, each an
                array of shape ``(K, D)``.
        """
        self.inv_alphas, self.inv_betas = value

    def log_likelihoods(self, data, input, mask, tag):
        """Compute per-time-bin Poisson log-likelihoods.

        Note:
            The current implementation evaluates likelihoods using
            ``self.log_lambdas`` (a Poisson attribute) rather than the
            Gamma parameters.  This is a placeholder pending full
            Gamma-Poisson marginal likelihood integration.

        Args:
            data (np.ndarray): Observed spike counts, shape ``(T, D)``.
            input (np.ndarray): External inputs (unused).
            mask (np.ndarray or None): Boolean mask of shape ``(T, D)``.
                If ``None``, all observations are treated as valid.
            tag: Trial tag (unused).

        Returns:
            np.ndarray: Log-likelihood array of shape ``(T, K)``.
        """
        lambdas = np.exp(self.log_lambdas)
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        return stats.poisson_logpdf(data[:, None, :], lambdas, mask=mask[:, None, :])

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        """Sample a spike-count observation from state ``z``.

        First samples a firing rate from the Gamma prior for state ``z``,
        then draws Poisson spike counts given that rate.

        Args:
            z (int): Current discrete latent state index.
            xhist (np.ndarray): History of previous observations (unused).
            input (np.ndarray, optional): External input (unused). Defaults
                to ``None``.
            tag: Trial tag (unused). Defaults to ``None``.
            with_noise (bool): Whether to add Poisson noise. Defaults to
                ``True``.

        Returns:
            np.ndarray: Sampled spike counts of shape ``(D,)``.
        """
        alphas = np.exp(self.inv_alphas)
        betas = np.exp(self.inv_betas)
        lambdas = npr.gamma(alphas, 1.0 / betas)
        return npr.poisson(lambdas[z])

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        """Placeholder M-step for Gamma distribution parameters.

        Args:
            expectations: Per-trial E-step outputs (unused).
            datas: Observed data sequences (unused).
            inputs: External input sequences (unused).
            masks: Observation masks (unused).
            tags: Trial tags (unused).
            **kwargs: Additional keyword arguments (unused).
        """

    def smooth(self, expectations, data, input, tag):
        """Compute the posterior mean observation under the Gamma-Poisson model.

        The mean firing rate under state ``k`` is
        :math:`\\alpha_{kn} / \\beta_{kn}`, so the posterior mean spike
        count is:

        .. math::
            \\hat{y}_t = \\sum_k p(z_t = k \\mid y_{1:T}) \\cdot
                         \\frac{\\alpha_{kn}}{\\beta_{kn}}

        Args:
            expectations (np.ndarray): Marginal state probabilities of shape
                ``(T, K)``.
            data (np.ndarray): Observed spike counts of shape ``(T, D)``
                (unused).
            input (np.ndarray): External inputs (unused).
            tag: Trial tag (unused).

        Returns:
            np.ndarray: Posterior mean spike counts, shape ``(T, D)``.
        """
        mean_lambdas = np.exp(self.inv_alphas - self.inv_betas)
        return expectations.dot(mean_lambdas)


class StepTransitions(StationaryTransitions):
    """Single global step-probability transition model for a 2-state HMM.

    Parameterises the transition matrix with a single log step-probability
    ``log_p_step`` shared across all trials.  The resulting transition matrix
    is:

    .. math::
        P = \\begin{pmatrix}
                1 - p_\\mathrm{step} & p_\\mathrm{step} \\\\
                \\varepsilon & 1 - \\varepsilon
            \\end{pmatrix}

    where state 0 is the baseline state and state 1 is the step state.
    Once the step state is entered it is absorbing (probability ``1 - eps``
    of self-transition).  The step probability ``p_step`` is updated during
    the M-step via gradient-based optimisation inherited from
    :class:`ssm.transitions.Transitions`.

    Args:
        K (int): Number of discrete states. Must be 2.
        D (int): Latent dimensionality.
        M (int): External input dimensionality. Defaults to 0.

    Attributes:
        log_p_step (float): Log of the per-bin step probability.
            Initialised to ``log(0.02)``.
        eps (float): Small constant (``1e-80``) used to keep the step state
            nearly absorbing without numerical issues.
        log_Ps (np.ndarray): Log transition matrix of shape ``(K, K)``.
    """

    def __init__(self, K, D, M=0):
        super(StationaryTransitions, self).__init__(K, D, M)
        assert K == 2

        # single step param
        self.log_p_step = np.log(0.02)
        self.eps = 1e-80
        Ps = np.array(
            [
                [1.0 - np.exp(self.log_p_step), np.exp(self.log_p_step)],
                [self.eps, 1.0 - self.eps],
            ]
        )
        self.log_Ps = np.log(Ps)

    @property
    def params(self):
        """tuple: Singleton tuple containing ``log_p_step``."""
        return (self.log_p_step,)

    @params.setter
    def params(self, value):
        """Set the log step probability and rebuild the transition matrix.

        Args:
            value (tuple): Singleton tuple containing the new
                ``log_p_step`` scalar.
        """
        self.log_p_step = value[0]
        mask1 = np.array([-1.0, 1.0])
        r1 = np.array([1.0, 0.0]) + mask1 * np.exp(self.log_p_step)
        r2 = np.array([self.eps, 1.0 - self.eps])
        self.log_Ps = np.log(np.vstack((r1, r2)))

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        """Update the step probability via gradient-based optimisation.

        Delegates to the parent :class:`ssm.transitions.Transitions` M-step,
        which maximises the expected log-likelihood with respect to
        ``log_p_step``.

        Args:
            expectations (list[tuple]): Per-trial E-step outputs containing
                pairwise state expectations.
            datas (list[np.ndarray]): Observed data sequences.
            inputs (list[np.ndarray]): External input sequences.
            masks (list[np.ndarray]): Observation masks.
            tags (list[int]): Trial indices.
            **kwargs: Additional keyword arguments forwarded to the parent
                M-step.
        """
        Transitions.m_step(self, expectations, datas, inputs, masks, tags, **kwargs)


class TrialStepTransitions(StationaryTransitions):
    """Per-trial step-probability transition model for a 2-state HMM.

    Maintains an independent log step-probability for every trial, allowing
    the hazard rate of the step to vary across experimental conditions or
    repetitions.  The transition matrix for trial ``tag`` is:

    .. math::
        P_{\\mathrm{tag}} = \\begin{pmatrix}
            1 - p_{\\mathrm{step},\\mathrm{tag}} &
            p_{\\mathrm{step},\\mathrm{tag}} \\\\
            \\varepsilon & 1 - \\varepsilon
        \\end{pmatrix}

    Args:
        K (int): Number of discrete states. Must be 2.
        D (int): Latent dimensionality.
        M (int): External input dimensionality. Defaults to 0.
        num_trials (int): Total number of trials. One log step-probability
            is allocated per trial. Defaults to 1.

    Attributes:
        log_p_steps (np.ndarray): Per-trial log step probabilities, shape
            ``(num_trials,)``.  Initialised to ``log(0.02)`` for all trials.
        eps (float): Small constant (``1e-80``) keeping the step state
            nearly absorbing.
        log_Ps (np.ndarray): Per-trial log transition matrices, shape
            ``(num_trials, K, K)``.
    """

    def __init__(self, K, D, M=0, num_trials=1):
        super(StationaryTransitions, self).__init__(K, D, M)
        assert K == 2

        # single step param
        self.log_p_steps = np.log(0.02) * np.ones(num_trials)
        self.eps = 1e-80
        Ps = []
        for log_p_step in self.log_p_steps:
            P = np.array(
                [
                    [1.0 - np.exp(log_p_step), np.exp(log_p_step)],
                    [self.eps, 1.0 - self.eps],
                ]
            )
            Ps.append(P)
        self.log_Ps = np.array([np.log(P) for P in Ps])

    @property
    def params(self):
        """tuple: Singleton tuple containing the ``log_p_steps`` array."""
        return (self.log_p_steps,)

    @params.setter
    def params(self, value):
        """Set per-trial log step probabilities and rebuild all transition matrices.

        Args:
            value (tuple): Singleton tuple containing a 1-D array of
                ``log_p_steps`` values, one per trial.
        """
        self.log_p_steps = value[0]
        mask1 = np.array([-1.0, 1.0])
        r1s = [
            np.array([1.0, 0.0]) + mask1 * np.exp(log_p_step)
            for log_p_step in self.log_p_steps
        ]
        r2 = np.array([self.eps, 1.0 - self.eps])
        self.log_Ps = np.array([np.log(np.vstack((r1, r2))) for r1 in r1s])

    def log_transition_matrices(self, data, input, mask, tag):
        """Return the normalised log transition matrix for trial ``tag``.

        Selects the per-trial log transition matrix indexed by ``tag``,
        row-normalises it to sum to 1 in probability space, and broadcasts
        it across all time bins.

        Args:
            data (np.ndarray): Observed data at the current time bin
                (unused).
            input (np.ndarray): External input at the current time bin
                (unused).
            mask (np.ndarray): Observation mask at the current time bin
                (unused).
            tag (int): Trial index used to select ``log_Ps[tag]``.

        Returns:
            np.ndarray: Log transition matrix of shape ``(1, K, K)``,
            suitable for broadcasting over time.
        """
        log_Ps = self.log_Ps[tag] - logsumexp(self.log_Ps[tag], axis=1, keepdims=True)
        return log_Ps[None, :, :]

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        """Update per-trial step probabilities via gradient-based optimisation.

        Delegates to the parent :class:`ssm.transitions.Transitions` M-step.

        Args:
            expectations (list[tuple]): Per-trial E-step outputs containing
                pairwise state expectations.
            datas (list[np.ndarray]): Observed data sequences.
            inputs (list[np.ndarray]): External input sequences.
            masks (list[np.ndarray]): Observation masks.
            tags (list[int]): Trial indices.
            **kwargs: Additional keyword arguments forwarded to the parent
                M-step.
        """
        Transitions.m_step(self, expectations, datas, inputs, masks, tags, **kwargs)


class Step(HMM):
    """Two-state step HMM combining step transitions with Poisson observations.

    Wraps a 2-state HMM in which the latent process transitions from a
    baseline state (state 0) to an absorbing step state (state 1) according
    to either a global or per-trial geometric waiting-time distribution.
    Observed spike counts are modelled using trial-specific Poisson rates
    (default) or standard Poisson observations shared across trials.

    The initial state distribution places all mass on state 0, reflecting
    the assumption that every trial begins in the baseline state.

    Args:
        K (int): Number of discrete states. Must be 2.
        D (int): Number of observed neurons.
        M (int): External input dimensionality. Keyword-only. Defaults to 0.
        transitions (str): Transition model to use.  One of:

            * ``"step"`` — global step probability (:class:`StepTransitions`).
            * ``"trial_step"`` — per-trial step probabilities
              (:class:`TrialStepTransitions`).

            Defaults to ``"step"``.
        transition_kwargs (dict, optional): Keyword arguments forwarded to
            the transition constructor (e.g., ``num_trials`` for
            ``"trial_step"``).
        observations (str): Observation model to use.  One of:

            * ``"trial_poisson"`` — per-trial Poisson rates
              (:class:`TrialPoissonObservations`).
            * ``"poisson"`` — shared Poisson rates
              (:class:`ssm.observations.PoissonObservations`).

            Defaults to ``"trial_poission"`` (note: kept as-is for backwards
            compatibility; effective key is ``"trial_poisson"``).
        observation_kwargs (dict, optional): Keyword arguments forwarded to
            the observation constructor (e.g., ``num_trials``, ``bin_size``).
        **kwargs: Additional keyword arguments forwarded to the parent
            :class:`ssm.hmm.HMM`.

    Note:
        The default value of ``observations`` is ``"trial_poission"``
        (misspelling) but the dispatch dictionary maps ``"trial_poisson"``
        (correct spelling).  Pass ``"trial_poisson"`` to use
        :class:`TrialPoissonObservations`.
    """

    def __init__(
        self,
        K,
        D,
        *,
        M=0,
        transitions="step",
        transition_kwargs=None,
        observations="trial_poission",
        observation_kwargs=None,
        **kwargs,
    ):
        assert K == 2

        init_state_distn = InitialStateDistribution(K, D, M=M)
        eps = 1e-80
        init_state_distn.log_pi0 = np.log(
            np.concatenate(([1.0 - eps], (eps) * np.ones(K - 1)))
        )

        transition_classes = dict(step=StepTransitions, trial_step=TrialStepTransitions)
        transition_kwargs = transition_kwargs or {}
        transitions = transition_classes[transitions](K, D, M=M, **transition_kwargs)

        observation_classes = dict(
            trial_poisson=TrialPoissonObservations, poisson=PoissonObservations
        )
        observation_kwargs = observation_kwargs or {}
        observation_distn = observation_classes[observations](
            K, D, M=M, **observation_kwargs
        )

        super().__init__(
            K,
            D,
            M=M,
            init_state_distn=init_state_distn,
            transitions=transitions,
            observations=observation_distn,
        )
