"""Ramping accumulator models for neural decision-making.

This module implements a family of switching linear dynamical system (SLDS)
models in which a 1-D latent variable *ramps* toward an absorbing decision
boundary driven by coherence-dependent input.  The core idea is that a
neural population integrates sensory evidence (encoded as a per-coherence
ramp rate ``beta``) until the accumulated signal crosses a threshold,
triggering a state transition from the ``ramp`` state to a ``bound`` state.
"""

import copy

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import logsumexp
from ssm.emissions import GaussianEmissions, PoissonEmissions
from ssm.hmm import HMM
from ssm.init_state_distns import InitialStateDistribution
from ssm.lds import SLDS
from ssm.observations import AutoRegressiveDiagonalNoiseObservations
from ssm.optimizers import (
    bfgs,
    lbfgs,
)
from ssm.preprocessing import (
    interpolate_data,
)
from ssm.transitions import RecurrentOnlyTransitions, RecurrentTransitions, Transitions
from ssm.util import ensure_args_are_lists, one_hot


class RampingSoftTransitions(RecurrentOnlyTransitions):
    """Soft recurrent-only transitions for a 2-state ramp/bound model.

    Implements state transitions in which the probability of entering the
    absorbing ``bound`` state increases monotonically with the current value
    of the latent accumulator.  The transition log-probabilities are kept
    fixed (no M-step update), making this a *soft*, non-learned boundary.

    The transition log-probability matrix encodes:

    * State 0 (ramp): neutral baseline log-probability (``0``).
    * State 1 (bound): large positive log-probability scaled by the latent
      value via ``R``, so the bound state becomes overwhelmingly likely as
      the accumulator approaches 1.

    Args:
        K (int): Number of discrete states. Must be 2.
        D (int): Dimensionality of the continuous latent state. Must be 1.
        M (int): Dimensionality of the external input. Defaults to 0.
        scale (float): Sharpness of the soft boundary. Larger values produce
            a harder, more step-like transition. Defaults to 200.

    Attributes:
        Ws (np.ndarray): Input weight matrix of shape ``(K, M)``, fixed at
            zero (no input-driven transitions).
        Rs (np.ndarray): Recurrence weight matrix of shape ``(K, 1)``; the
            bound state receives weight ``scale`` so its log-probability grows
            linearly with the latent value.
        r (np.ndarray): Bias vector of shape ``(K,)``; the bound state has a
            large negative bias ``-scale`` so it is only preferred near the
            boundary.
    """

    def __init__(self, K, D, M=0, scale=200):
        assert K == 2
        assert D == 1
        super().__init__(K, D, M)

        # Parameters linking past observations to state distribution
        self.Ws = np.zeros((K, M))
        self.Rs = np.array([0, scale]).reshape((K, 1))
        self.r = np.array([0, -scale])

    @property
    def params(self):
        """tuple: Empty tuple; all parameters are fixed and not learned."""
        return ()

    @params.setter
    def params(self, value):
        """No-op setter; parameters are fixed."""

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        """No-op initialiser; parameters require no data-driven initialisation.

        Args:
            datas: Unused.
            inputs: Unused.
            masks: Unused.
            tags: Unused.
        """

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        """No-op M-step; transition parameters are held fixed.

        Args:
            expectations: Unused.
            datas: Unused.
            inputs: Unused.
            masks: Unused.
            tags: Unused.
            **kwargs: Unused.
        """


class RampingTransitions(RecurrentTransitions):
    """Recurrent transitions for a 2-state ramp/bound model.

    Extends :class:`ssm.transitions.RecurrentTransitions` with fixed
    parameters that encode a hard upper absorbing boundary.  The
    log-probability matrix is initialised so that:

    * State 0 (ramp) → State 0 (ramp): neutral (log-prob ``0``).
    * State 0 (ramp) → State 1 (bound): large negative bias unless the
      latent value is near 1, where ``R[1] * x`` overcomes the bias.
    * State 1 (bound) → State 1 (bound): self-reinforcing (log-prob ``scale``).

    No parameters are updated during fitting.

    Args:
        K (int): Number of discrete states. Must be 2.
        D (int): Dimensionality of the continuous latent state. Must be 1.
        M (int): Dimensionality of the external input. Defaults to 0.
        scale (float): Controls the sharpness of the boundary.  Defaults to
            200.

    Attributes:
        log_Ps (np.ndarray): Fixed log-probability matrix of shape ``(K, K)``.
        Ws (np.ndarray): Input weight matrix of shape ``(K, M)``, fixed at
            zero.
        Rs (np.ndarray): Recurrence weights of shape ``(K, D)``; the bound
            state receives weight ``scale``.
    """

    def __init__(self, K, D, M=0, scale=200):
        assert K == 2
        assert D == 1
        super().__init__(K, D, M)

        # Parameters linking past observations to state distribution
        self.log_Ps = np.array([[0, -scale], [-scale, scale]])
        self.Ws = np.zeros((K, M))
        self.Rs = np.array([0, scale]).reshape((K, D))

    @property
    def params(self):
        """tuple: Empty tuple; all parameters are fixed."""
        return ()

    @params.setter
    def params(self, value):
        """No-op setter; parameters are fixed."""

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        """No-op initialiser.

        Args:
            datas: Unused.
            inputs: Unused.
            masks: Unused.
            tags: Unused.
        """

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        """No-op M-step; parameters are held fixed.

        Args:
            expectations: Unused.
            datas: Unused.
            inputs: Unused.
            masks: Unused.
            tags: Unused.
            **kwargs: Unused.
        """


class RampingLowerBoundTransitions(RecurrentTransitions):
    """3-state recurrent transitions with a learnable lower absorbing boundary.

    Augments the standard 2-state (ramp/upper-bound) model with a third state
    that acts as a lower absorbing boundary.  The location of the lower
    boundary is parameterised by ``lb_loc`` and ``lb_scale``, which are
    updated during the M-step via BFGS optimisation.

    State definitions:
        * State 0: Ramp state — the accumulator evolves according to the
          drift-diffusion dynamics.
        * State 1: Upper bound — absorbing state triggered when the
          accumulator reaches the upper boundary.
        * State 2: Lower bound — absorbing state triggered when the
          accumulator falls below the lower boundary.

    Args:
        K (int): Number of discrete states. Must be 3.
        D (int): Dimensionality of the continuous latent state. Must be 1.
        M (int): Dimensionality of the external input. Defaults to 0.
        scale (float): Sharpness parameter for the upper boundary transition.
            Defaults to 200.

    Attributes:
        log_Ps (np.ndarray): Log-probability matrix of shape ``(K, K)``.
        scale (float): Upper-boundary sharpness.
        lb_loc (float): Location (threshold) of the lower boundary in
            latent-state units.
        lb_scale (float): Sharpness of the lower boundary transition.
        Ws (np.ndarray): Input weight matrix of shape ``(K, M)``, fixed at
            zero.
        Rs (np.ndarray): Recurrence weights of shape ``(K, 1)`` encoding the
            sensitivity of each state to the current latent value.
    """

    def __init__(self, K, D, M=0, scale=200.0):
        assert K == 3
        assert D == 1
        super().__init__(K, D, M)

        # Parameters linking past observations to state distribution
        self.log_Ps = -scale * np.ones((K, K)) + np.diag(
            np.concatenate(([scale], 2.0 * scale * np.ones(K - 1)))
        )
        self.scale = scale
        self.lb_loc = 0.0
        self.lb_scale = 10.0
        self.log_Ps[0, 2] = self.lb_loc * self.lb_scale
        self.Ws = np.zeros((K, M))
        self.Rs = np.array([0, scale, -self.lb_scale]).reshape((3, 1))

    @property
    def params(self):
        """tuple: Learnable parameters ``(lb_loc, lb_scale)``."""
        return self.lb_loc, self.lb_scale

    @params.setter
    def params(self, value):
        """Set learnable parameters and rebuild derived matrices.

        Updates ``log_Ps`` and ``Rs`` to be consistent with the new
        ``lb_loc`` and ``lb_scale`` values.

        Args:
            value (tuple): A 2-tuple ``(lb_loc, lb_scale)``.
        """
        self.lb_loc, self.lb_scale = value
        mask = np.vstack((np.array([1.0, 1.0, 0.0]), np.ones((self.K - 1, self.K))))
        log_Ps = -self.scale * np.ones((self.K, self.K)) + np.diag(
            np.concatenate(([self.scale], 2.0 * self.scale * np.ones(self.K - 1)))
        )
        self.log_Ps = mask * log_Ps + (
            1.0 - mask
        ) * self.lb_loc * self.lb_scale * np.ones((self.K, self.K))
        self.Rs = np.array([0.0, self.scale, -self.lb_scale]).reshape((3, 1))

    def log_prior(self):
        """Compute the Gaussian log-prior on the lower-boundary location.

        Places a zero-mean Gaussian prior with variance 0.5 on ``lb_loc`` to
        regularise the lower boundary toward the centre of the latent space.

        Returns:
            float: Log-prior value.
        """
        loc_mean = 0.0
        loc_var = 0.5
        return np.sum(-0.5 * (self.lb_loc - loc_mean) ** 2 / loc_var)

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        """No-op initialiser.

        Args:
            datas: Unused.
            inputs: Unused.
            masks: Unused.
            tags: Unused.
        """

    def m_step(
        self, expectations, datas, inputs, masks, tags, optimizer="bfgs", **kwargs
    ):
        """Update lower-boundary parameters via gradient-based optimisation.

        Delegates to the parent :class:`ssm.transitions.Transitions` M-step,
        which maximises the expected log-likelihood with respect to the
        learnable parameters ``(lb_loc, lb_scale)``.

        Args:
            expectations (list): Per-trial posterior state expectations from
                the E-step.
            datas (list): Observed data sequences.
            inputs (list): External input sequences.
            masks (list): Observation masks.
            tags (list): Trial tags.
            optimizer (str): Gradient-based optimiser to use. Defaults to
                ``"bfgs"``.
            **kwargs: Additional keyword arguments forwarded to the parent
                M-step.
        """
        Transitions.m_step(
            self,
            expectations,
            datas,
            inputs,
            masks,
            tags,
            optimizer=optimizer,
            **kwargs,
        )


class RampingObservations(AutoRegressiveDiagonalNoiseObservations):
    """Auto-regressive observations for the ramping accumulator dynamics.

    Models the 1-D latent accumulator as a first-order auto-regressive process
    with unit auto-regressive coefficient and coherence-dependent drift:

    .. math::
        x_t = x_{t-1} + \\beta^\\top u_t + \\epsilon_t, \\quad
        \\epsilon_t \\sim \\mathcal{N}(0, \\sigma^2)

    where :math:`u_t` is a one-hot coherence input and :math:`\\beta` is the
    vector of per-coherence ramp rates.  Bound states have zero input weight
    (``V_k = 0`` for :math:`k > 0`) and negligible noise, effectively
    freezing the accumulator once it reaches a boundary.

    Args:
        K (int): Number of discrete states.
        D (int): Dimensionality of the continuous latent state. Must be 1.
        M (int): Dimensionality of the coherence input (one-hot). Defaults
            to 5.
        lags (int): Number of autoregressive lags. Unused; always 1.
        beta (np.ndarray, optional): Initial ramp rates of shape ``(M,)``.
            Defaults to a linear grid from ``-0.02`` to ``0.02``.
        log_sigma_scale (float): Log of the diffusion variance for the ramp
            state. Defaults to ``log(1e-3)``.
        x0 (float): Mean initial accumulator value at trial onset. Defaults
            to 0.4.

    Attributes:
        beta (np.ndarray): Per-coherence ramp rates, shape ``(M,)``.
        log_sigma_scale (float): Log diffusion variance for the ramp state.
        x0 (float): Mean initial value of the accumulator.
        base_var (float): Fixed noise variance for bound states (``1e-4``).
    """

    def __init__(
        self, K, D=1, M=5, lags=1, beta=None, log_sigma_scale=np.log(1e-3), x0=0.4
    ):
        assert D == 1
        super().__init__(K, D, M)

        # The only free parameters are the ramp rate...
        if beta is None:
            beta = np.linspace(-0.02, 0.02, M)
        self.beta = beta
        self.log_sigma_scale = log_sigma_scale  # log variance
        self.x0 = x0  # mu init

        # and the noise variances
        self.base_var = 1e-4
        mask1 = np.vstack(
            (
                np.ones(
                    D,
                ),
                np.zeros((K - 1, D)),
            )
        )
        mask2 = np.vstack((np.zeros(D), np.ones((K - 1, D))))
        self._log_sigmasq = self.log_sigma_scale * mask1 + np.log(self.base_var) * mask2

        # set init params
        self.mu_init = self.x0 * np.ones((K, 1))
        self._log_sigmasq_init = (
            self.log_sigma_scale * mask1 + np.log(self.base_var) * mask2
        )

        # They only differ in their input
        self.Vs[0] = self.beta  # ramp
        for k in range(1, K):
            self.Vs[k] = 0  # bound

        # Set the remaining parameters to fixed values
        self._As = np.ones((K, 1, 1))
        self.bs = np.zeros((K, 1))

    @property
    def params(self):
        """tuple: Learnable parameters ``(beta, log_sigma_scale, x0)``."""
        return self.beta, self.log_sigma_scale, self.x0

    @params.setter
    def params(self, value):
        """Set learnable parameters and rebuild all derived arrays.

        Rebuilds the input weight tensor ``Vs``, the log-variance arrays
        ``_log_sigmasq`` and ``_log_sigmasq_init``, and the initial mean
        ``mu_init`` to be consistent with the new parameters.

        Args:
            value (tuple): A 3-tuple ``(beta, log_sigma_scale, x0)``.
        """
        self.beta, self.log_sigma_scale, self.x0 = value
        mask = np.reshape(np.concatenate(([1], np.zeros(self.K - 1))), (self.K, 1, 1))
        self.Vs = mask * self.beta
        mask1 = np.vstack(
            (
                np.ones(
                    self.D,
                ),
                np.zeros((self.K - 1, self.D)),
            )
        )
        mask2 = np.vstack((np.zeros(self.D), np.ones((self.K - 1, self.D))))
        self._log_sigmasq = self.log_sigma_scale * mask1 + np.log(self.base_var) * mask2
        self.mu_init = self.x0 * np.ones((self.K, 1))
        self._log_sigmasq_init = (
            self.log_sigma_scale * mask1 + np.log(self.base_var) * mask2
        )

    # def log_prior(self):
    #     ...existing code...

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        """No-op initialiser; parameters are set analytically in the M-step.

        Args:
            datas: Unused.
            inputs: Unused.
            masks: Unused.
            tags: Unused.
        """

    def m_step(
        self,
        expectations,
        datas,
        inputs,
        masks,
        tags,
        continuous_expectations=None,
        **kwargs,
    ):
        """Compute the M-step for the ramping observation parameters.

        Performs an analytic maximum-a-posteriori update of ``beta``,
        ``log_sigma_scale``, and ``x0`` following the Gaussian auto-regressive
        M-step derivation.  A conjugate Inverse-Gamma prior (parameterised as
        InvWishart in 1-D) is placed on the diffusion variance to regularise
        small-sample estimates.

        The ramp rate ``beta`` is solved analytically via:

        .. math::
            \\beta = (\\mathbb{E}[u_t u_t^\\top])^{-1}
                     (\\mathbb{E}[y_t u_t^\\top] - \\mathbb{E}[x_{t-1} u_t^\\top])

        and the diffusion variance is updated as a MAP estimate under an
        Inverse-Gamma prior with shape ``alpha = 1.1`` and rate
        ``beta_prior = 1e-3``.

        Args:
            expectations (list): Per-trial posterior expectations
                ``(E[z_t], E[z_t z_{t+1}^T], ...)`` from the E-step.
            datas (list): Observed neural data sequences. Used as a fallback
                for ``x_means`` when ``continuous_expectations`` is ``None``.
            inputs (list): External coherence input sequences of shape
                ``(T, M)``.
            masks (list): Observation masks.
            tags (list): Trial tags.
            continuous_expectations (list, optional): Posterior mean and
                covariance of the continuous latent state from the SLDS
                E-step.  When provided, sufficient statistics are computed
                from these rather than from ``datas``. Defaults to ``None``.
            **kwargs: Additional keyword arguments (unused).
        """

        K, D, M, lags = self.K, self.D, self.M, 1

        # Copy priors from log_prior. 1D InvWishart(\nu, \Psi) is InvGamma(\nu/2, \Psi/2)
        nu0 = 2.0 * 1.1  # 2 times \alpha
        Psi0 = 2.0 * 1e-3  # 2 times \beta

        # Collect sufficient statistics
        if continuous_expectations is None:
            ExuxuTs, ExuyTs, EyyTs, Ens = self._get_sufficient_statistics(
                expectations, datas, inputs
            )
            x_means = datas
        else:
            ExuxuTs, ExuyTs, EyyTs, Ens = self._extend_given_sufficient_statistics(
                expectations, continuous_expectations, inputs
            )
            x_means = [exp[1] for exp in continuous_expectations]

        # remove bias block
        ExuxuTs = ExuxuTs[:, :-1, :-1]
        ExuyTs = ExuyTs[:, :-1, :]

        # V_d = E[u_t u_t^T]^{-1} * (E[y_t u_t^T]  - E[x_{t-1} u_t^T])
        Euu_d = ExuxuTs[0, 1:, 1:]
        Exu_d = ExuxuTs[0, 0, 1:]
        Eyu_d = ExuyTs[0, 1:, 0]
        Euu_d = Euu_d + 1e-6 * np.eye(self.M)  # temp add for numerical stability
        beta = np.linalg.solve(Euu_d, Eyu_d - Exu_d).T
        W = np.concatenate((np.array([1.0]), beta))
        x0 = np.mean([x[0] for x in x_means])

        # Solve for the MAP estimate of the covariance
        # should include estimate of initial state - ignore for now (TODO)
        sqerr = EyyTs[0, :, :] - 2 * W @ ExuyTs[0, :, :] + W @ ExuxuTs[0, :, :] @ W.T
        nu = nu0 + Ens[0]
        log_sigma_scale = np.log((sqerr + Psi0) / (nu + self.D + 1))

        params = beta, log_sigma_scale, x0
        self.params = params

        return


class RampingPoissonEmissions(PoissonEmissions):
    """Poisson spike-count emissions with a softplus link for ramping models.

    Maps the 1-D latent accumulator to per-neuron spike counts via a linear
    loading matrix ``C`` followed by a softplus nonlinearity:

    .. math::
        \\lambda_t = \\log(1 + \\exp(C x_t)) \\cdot \\Delta

    where :math:`\\Delta` is the bin size in seconds.  The input weight
    matrix ``Fs`` and offset ``ds`` are zeroed and held fixed so that only
    the latent state drives the firing rate.

    A Gamma log-prior is placed on the columns of ``C`` to encourage
    positive, sparse loadings.

    Args:
        N (int): Number of observed neurons.
        K (int): Number of discrete states.
        D (int): Dimensionality of the continuous latent state.
        M (int): Dimensionality of the external input. Defaults to 0.
        single_subspace (bool): Whether all discrete states share the same
            emission matrix. Defaults to ``True``.
        link (str): Link function. Defaults to ``"softplus"``.
        bin_size (float): Spike-count bin size in seconds. Defaults to
            ``0.01``.

    Attributes:
        Fs (np.ndarray): Input weight matrix, fixed at zero.
        ds (np.ndarray): Offset vector, fixed at zero.
    """

    def __init__(
        self, N, K, D, M=0, single_subspace=True, link="softplus", bin_size=0.01
    ):
        super().__init__(
            N, K, D, M=M, single_subspace=single_subspace, link=link, bin_size=bin_size
        )
        # Make sure the input matrix Fs is set to zero and never updated
        self.Fs *= 0
        self.ds *= 0

    # Construct an emissions model
    @property
    def params(self):
        """np.ndarray: Loading matrix ``Cs`` of shape ``(1, N, D)``."""
        return self.Cs

    @params.setter
    def params(self, value):
        """Set the loading matrix.

        Args:
            value (np.ndarray): New loading matrix of shape ``(1, N, D)``.
        """
        self.Cs = value

    def log_prior(self):
        """Compute the Gamma log-prior on the loading matrix.

        Places an ``a=2, b=0.05`` Gamma prior on each element of ``Cs[0]``
        to encourage positive, non-degenerate loadings:

        .. math::
            \\log p(C) = \\sum_{n,d} (a - 1) \\log C_{nd} - b \\cdot C_{nd}

        Returns:
            float: Log-prior value.
        """
        a = 2.0
        b = 0.05
        return np.sum((a - 1.0) * np.log(self.Cs[0]) - b * self.Cs[0])

    def initialize(self, datas, inputs=None, masks=None, tags=None, choices=None):
        """Initialise the loading matrix from the spike-count data.

        Estimates the population firing gain ``C`` and the initial latent
        value ``x0`` from trial-end spike counts, stratified by choice
        direction.  The initialisation heuristic computes the maximum
        end-of-trial firing rate across choice categories and uses this to
        set the scale of ``Cs``.

        Args:
            datas (list): List of spike-count arrays, each of shape
                ``(T_i, N)``.
            inputs (list, optional): External input sequences. Unused.
            masks (list, optional): Observation masks. Unused.
            tags (list, optional): Trial tags. Unused.
            choices (array-like): Binary choice label (0 or 1) for each
                trial, used to stratify end-of-trial firing rates.

        Returns:
            tuple: A 2-tuple ``(C, x0)`` where:

            * **C** (*float*): Estimated population firing gain.
            * **x0** (*float*): Estimated initial latent value.
        """

        def initialize_ramp_choice(ys, choices, bin_size):
            """Estimate C and x0 from end-of-trial firing rates by choice.

            Args:
                ys (list): Spike count arrays per trial.
                choices (np.ndarray): Binary choice labels, shape ``(T,)``.
                bin_size (float): Bin size in seconds.

            Returns:
                tuple: ``(C, x0)`` estimates.
            """
            choice_0 = np.where(choices == 0)[0]
            choice_1 = np.where(choices == 1)[0]
            y_end = np.array([y[-5:] for y in ys])
            C0 = np.mean(y_end[choice_0]) / bin_size
            C1 = np.mean(y_end[choice_1]) / bin_size
            C = max(C0, C1)
            y0_mean = np.mean([y[:3] for y in ys])
            x0 = y0_mean / C / bin_size
            return C, x0

        choices = np.array(choices)
        C, x0 = initialize_ramp_choice(datas, choices, self.bin_size)

        self.Cs[0] = C.reshape((self.N, self.D))

        return C, x0


class RampingPoissonSigmoidEmissions(PoissonEmissions):
    """Poisson spike-count emissions with a sigmoid link for ramping models.

    Alternative to :class:`RampingPoissonEmissions` for populations in which
    firing rates saturate as the accumulator approaches the boundary.  Uses a
    sigmoid link function, and the loading matrix, offsets, and sigmoid
    inverse-temperature parameters are all jointly optimised during
    initialisation via BFGS.

    Args:
        N (int): Number of observed neurons.
        K (int): Number of discrete states.
        D (int): Dimensionality of the continuous latent state.
        M (int): Dimensionality of the external input. Defaults to 0.
        single_subspace (bool): Whether all discrete states share the same
            emission matrix. Defaults to ``True``.
        link (str): Link function. Defaults to ``"sigmoid"``.
        bin_size (float): Spike-count bin size in seconds. Defaults to
            ``0.01``.

    Attributes:
        Fs (np.ndarray): Input weight matrix, fixed at zero.
        ds (np.ndarray): Offset vector, fixed at zero.
    """

    def __init__(
        self, N, K, D, M=0, single_subspace=True, link="sigmoid", bin_size=0.01
    ):
        super().__init__(
            N, K, D, M=M, single_subspace=single_subspace, link=link, bin_size=bin_size
        )
        # Make sure the input matrix Fs is set to zero and never updated
        self.Fs *= 0
        self.ds *= 0

    @property
    def params(self):
        """tuple: Learnable parameters ``(Cs, ds, inv_Ls, inv_bs)``."""
        return self.Cs, self.ds, self.inv_Ls, self.inv_bs

    @params.setter
    def params(self, value):
        """Set all learnable emission parameters.

        Args:
            value (tuple): 4-tuple ``(Cs, ds, inv_Ls, inv_bs)``.
        """
        self.Cs, self.ds, self.inv_Ls, self.inv_bs = value

    def initialize(
        self,
        base_model,
        datas,
        inputs=None,
        masks=None,
        tags=None,
        emission_optimizer="bfgs",
        num_optimizer_iters=1000,
    ):
        """Initialise emission parameters by maximising the emission likelihood.

        Generates synthetic latent trajectories from ``base_model`` and then
        optimises the emission parameters to maximise the expected log-
        likelihood of the observed spike counts given those trajectories.

        Args:
            base_model (ObservedRamping): A pre-configured
                :class:`ObservedRamping` HMM used to generate reference latent
                trajectories for the initialisation objective.
            datas (list): Observed spike-count arrays, each of shape
                ``(T_i, N)``.
            inputs (list, optional): External input sequences. Defaults to
                ``None``.
            masks (list, optional): Observation masks. Missing observations
                are interpolated before optimisation. Defaults to ``None``.
            tags (list, optional): Trial tags. Defaults to ``None``.
            emission_optimizer (str): Optimiser for the emission likelihood.
                Must be one of ``"bfgs"`` or ``"lbfgs"``. Defaults to
                ``"bfgs"``.
            num_optimizer_iters (int): Maximum number of optimiser iterations.
                Defaults to 1000.
        """
        print("Initializing Emissions parameters...")
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        Td = sum([data.shape[0] for data in datas])
        xs = [
            base_model.sample(T=data.shape[0], input=input)[1]
            for data, input in zip(datas, inputs)
        ]

        def _objective(params, itr):
            self.params = params
            obj = 0
            obj += self.log_prior()
            for data, input, mask, tag, x in zip(datas, inputs, masks, tags, xs):
                obj += np.sum(self.log_likelihoods(data, input, mask, tag, x))
            return -obj / Td

        # Optimize emissions log-likelihood
        optimizer = dict(bfgs=bfgs, lbfgs=lbfgs)[emission_optimizer]
        self.params = optimizer(
            _objective, self.params, num_iters=num_optimizer_iters, full_output=False
        )


class RampingGaussianEmissions(GaussianEmissions):
    """Gaussian emissions for ramping models with continuous neural signals.

    Suitable when the observed data are continuous (e.g., smoothed firing
    rates, local field potentials) rather than spike counts.  The input
    weight matrix ``Fs`` is zeroed and held fixed.

    Args:
        N (int): Number of observed dimensions.
        K (int): Number of discrete states.
        D (int): Dimensionality of the continuous latent state.
        M (int): Dimensionality of the external input. Defaults to 0.
        single_subspace (bool): Whether all discrete states share the same
            emission matrix. Defaults to ``True``.

    Attributes:
        Fs (np.ndarray): Input weight matrix, fixed at zero.
    """

    def __init__(self, N, K, D, M=0, single_subspace=True):
        super().__init__(N, K, D, M=M, single_subspace=single_subspace)
        # Make sure the input matrix Fs is set to zero and never updated
        self.Fs *= 0

    @property
    def params(self):
        """tuple: Learnable parameters ``(_Cs, ds, inv_etas)``."""
        return self._Cs, self.ds, self.inv_etas

    @params.setter
    def params(self, value):
        """Set all learnable emission parameters.

        Args:
            value (tuple): 3-tuple ``(_Cs, ds, inv_etas)``.
        """
        self._Cs, self.ds, self.inv_etas = value

    @ensure_args_are_lists
    def initialize(
        self,
        datas,
        inputs=None,
        masks=None,
        tags=None,
        num_em_iters=10,
        num_tr_iters=50,
    ):
        """Placeholder initialiser for Gaussian emission parameters.

        Args:
            datas (list): Observed continuous neural signals.
            inputs (list, optional): External inputs. Defaults to ``None``.
            masks (list, optional): Observation masks. Defaults to ``None``.
            tags (list, optional): Trial tags. Defaults to ``None``.
            num_em_iters (int): Number of EM iterations (unused). Defaults to
                10.
            num_tr_iters (int): Number of transition-refinement iterations
                (unused). Defaults to 50.
        """
        # here, init params using data
        # estimate boundary rate
        # use that to estimate initial x0
        # then init w2, betas


class RampingInitialStateDistribution(InitialStateDistribution):
    """Concentrated initial state distribution for ramping models.

    Places nearly all probability mass (99.9%) on state 0 (the ramp state)
    at trial onset, reflecting the assumption that the accumulator always
    begins in the pre-decision ramping regime rather than at a boundary.

    The remaining 0.1% is distributed uniformly across the remaining ``K-1``
    states for numerical stability.

    Args:
        K (int): Number of discrete states.
        D (int): Dimensionality of the continuous latent state.
        M (int): Dimensionality of the external input. Defaults to 0.

    Attributes:
        K (int): Number of discrete states.
        D (int): Latent dimensionality.
        M (int): Input dimensionality.
        log_pi0 (np.ndarray): Log unnormalised initial state probabilities,
            shape ``(K,)``.
    """

    def __init__(self, K, D, M=0):
        self.K, self.D, self.M = K, D, M
        self.log_pi0 = -np.log(K) * np.ones(K)

    @property
    def params(self):
        """tuple: Singleton tuple containing ``log_pi0``."""
        return (self.log_pi0,)

    @params.setter
    def params(self, value):
        """Set the log initial state probabilities.

        Args:
            value (tuple): Singleton tuple containing a log-probability array
                of shape ``(K,)``.
        """
        self.log_pi0 = value[0]

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        """No-op initialiser; the initial distribution is set analytically.

        Args:
            datas: Unused.
            inputs: Unused.
            masks: Unused.
            tags: Unused.
        """

    def permute(self, perm):
        """Permute the discrete latent states.

        Args:
            perm (array-like): Permutation indices of length ``K``.
        """
        self.log_pi0 = self.log_pi0[perm]

    @property
    def initial_state_distn(self):
        """np.ndarray: Normalised initial state probabilities, shape ``(K,)``."""
        return np.exp(self.log_pi0 - logsumexp(self.log_pi0))

    @property
    def log_initial_state_distn(self):
        """np.ndarray: Log normalised initial state probabilities, shape ``(K,)``."""
        return self.log_pi0 - logsumexp(self.log_pi0)

    def log_prior(self):
        """Flat log-prior on the initial state distribution.

        Returns:
            int: Always 0 (uninformative prior).
        """
        return 0

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        """No-op M-step; the initial distribution is held fixed.

        Args:
            expectations: Unused.
            datas: Unused.
            inputs: Unused.
            masks: Unused.
            tags: Unused.
            **kwargs: Unused.
        """
        self.log_pi0 = self.log_pi0


class ObservedRamping(HMM):
    """HMM wrapper for observed (emission-free) ramping accumulator models.

    Combines :class:`RampingInitialStateDistribution`,
    :class:`RampingTransitions` (or one of its variants), and
    :class:`RampingObservations` into a standard HMM in which the observed
    data *are* the latent accumulator values.  Primarily used as a base model
    during the initialisation of :class:`Ramping` emission parameters.

    Args:
        K (int): Number of discrete states.
        D (int): Dimensionality of the continuous latent state. Must be 1.
        M (int): Dimensionality of the external input. Keyword-only.
        transitions (str or Transitions): Transition model to use. One of
            ``"ramp"``, ``"ramplower"``, or ``"rampsoft"``, or a
            pre-constructed :class:`~ssm.transitions.Transitions` instance.
            Defaults to ``"ramp"``.
        transition_kwargs (dict, optional): Keyword arguments forwarded to
            the transition constructor when ``transitions`` is a string.
        observations (str): Observation model to use. Currently only
            ``"ramp"`` is supported. Defaults to ``"ramp"``.
        observation_kwargs (dict, optional): Keyword arguments forwarded to
            the observation constructor.
        **kwargs: Additional keyword arguments forwarded to the parent
            :class:`ssm.hmm.HMM`.
    """

    def __init__(
        self,
        K,
        D,
        *,
        M,
        transitions="ramp",
        transition_kwargs=None,
        observations="ramp",
        observation_kwargs=None,
        **kwargs,
    ):

        init_state_distn = RampingInitialStateDistribution(K, D, M=M)
        init_state_distn.log_pi0 = np.log(
            np.concatenate(([0.999], (0.001 / (K - 1)) * np.ones(K - 1)))
        )

        if transitions == "ramplower":
            assert K == 3
        transition_classes = dict(
            ramp=RampingTransitions,
            ramplower=RampingLowerBoundTransitions,
            rampsoft=RampingSoftTransitions,
        )
        transition_kwargs = transition_kwargs or {}
        if isinstance(transitions, str):
            transitions = transition_classes[transitions](
                K, D, M=M, **transition_kwargs
            )

        observation_kwargs = observation_kwargs or {}
        if observations == "ramp":
            observations = RampingObservations(K, D, M=M, **observation_kwargs)

        super().__init__(
            K,
            D,
            M=M,
            init_state_distn=init_state_distn,
            transitions=transitions,
            observations=observations,
        )


class Ramping(SLDS):
    """Full SLDS ramping accumulator model with neural population emissions.

    Implements a switching linear dynamical system (SLDS) in which a 1-D
    latent accumulator ramps toward an absorbing boundary at a rate
    determined by the coherence of the sensory stimulus.  Spike counts (or
    continuous signals) emitted by a neural population are modelled as noisy
    linear projections of the accumulator through a nonlinear link function.

    The model has three components:

    1. **Initial state**: :class:`RampingInitialStateDistribution` —
       concentrates mass on the ramp state at trial onset.
    2. **Transitions**: :class:`RampingTransitions` (default) or one of its
       variants — encodes the absorbing boundary structure.
    3. **Dynamics**: :class:`RampingObservations` — coherence-driven
       drift-diffusion dynamics.
    4. **Emissions**: :class:`RampingPoissonEmissions` (default),
       :class:`RampingPoissonSigmoidEmissions`, or
       :class:`RampingGaussianEmissions`.

    Args:
        N (int): Number of observed neurons (or emission dimensions).
        K (int): Number of discrete states. Defaults to 2.
        D (int): Dimensionality of the continuous latent state. Must be 1.
        M (int): Dimensionality of the coherence input. Keyword-only.
            Defaults to 5.
        transitions (str): Transition model. One of ``"ramp"``,
            ``"ramplower"``, or ``"rampsoft"``. Defaults to ``"ramp"``.
        transition_kwargs (dict, optional): Keyword arguments forwarded to
            the transition constructor.
        dynamics_kwargs (dict, optional): Keyword arguments forwarded to
            :class:`RampingObservations`.
        emissions (str): Emission model. One of ``"poisson"``,
            ``"poisson_sigmoid"``, or ``"gaussian"``. Defaults to
            ``"poisson"``.
        emission_kwargs (dict, optional): Keyword arguments forwarded to
            the emission constructor.
        single_subspace (bool): Whether all discrete states share a single
            emission subspace. Defaults to ``True``.
        **kwargs: Additional keyword arguments (unused).
    """

    def __init__(
        self,
        N,
        K=2,
        D=1,
        *,
        M=5,
        transitions="ramp",
        transition_kwargs=None,
        dynamics_kwargs=None,
        emissions="poisson",
        emission_kwargs=None,
        single_subspace=True,
        **kwargs,
    ):

        init_state_distn = RampingInitialStateDistribution(K, D, M=M)
        init_state_distn.log_pi0 = np.log(
            np.concatenate(([0.999], (0.001 / (K - 1)) * np.ones(K - 1)))
        )

        if transitions == "ramplower":
            assert K == 3
        transition_classes = dict(
            ramp=RampingTransitions,
            ramplower=RampingLowerBoundTransitions,
            rampsoft=RampingSoftTransitions,
        )
        transition_kwargs = transition_kwargs or {}
        transitions = transition_classes[transitions](K, D, M=M, **transition_kwargs)

        dynamics_kwargs = dynamics_kwargs or {}
        dynamics = RampingObservations(K, D, M=M, **dynamics_kwargs)

        emission_classes = dict(
            poisson=RampingPoissonEmissions,
            poisson_sigmoid=RampingPoissonSigmoidEmissions,
            gaussian=RampingGaussianEmissions,
        )
        emission_kwargs = emission_kwargs or {}
        emissions = emission_classes[emissions](
            N, K, D, M=M, single_subspace=single_subspace, **emission_kwargs
        )

        super().__init__(
            N,
            K=K,
            D=D,
            M=M,
            init_state_distn=init_state_distn,
            transitions=transitions,
            dynamics=dynamics,
            emissions=emissions,
        )

    @ensure_args_are_lists
    def initialize(
        self,
        datas,
        inputs=None,
        masks=None,
        tags=None,
        num_optimizer_iters=1000,
        num_em_iters=25,
        betas=None,
        accum_log_sigmasq=None,
        **kwargs,
    ):
        """Initialise model parameters from observed data.

        Constructs a shallow copy of the ramping dynamics as an
        :class:`ObservedRamping` HMM and uses it to initialise the emission
        parameters via the emission model's own ``initialize`` method.

        Args:
            datas (list): Observed spike-count (or continuous) arrays, each
                of shape ``(T_i, N)``.
            inputs (list, optional): External coherence input sequences,
                each of shape ``(T_i, M)``. Defaults to ``None``.
            masks (list, optional): Boolean observation masks. Defaults to
                ``None``.
            tags (list, optional): Trial tags. Defaults to ``None``.
            num_optimizer_iters (int): Maximum iterations for emission
                parameter optimisation. Defaults to 1000.
            num_em_iters (int): Number of EM iterations for ARHMM
                pre-training (unused in current implementation). Defaults
                to 25.
            betas (np.ndarray, optional): Initial ramp rates (unused;
                reserved for future use). Defaults to ``None``.
            accum_log_sigmasq (float, optional): Initial log diffusion
                variance (unused; reserved for future use). Defaults to
                ``None``.
            **kwargs: Additional keyword arguments (printed for debugging).
        """
        print(f"{kwargs}=")

        # First initialize the observation model
        # self.base_model = ObservedRamping(self.K, self.D, M=self.M,
        #                                transitions=self.transitions_label,
        #                                transition_kwargs=self.transition_kwargs,
        #                                observation_kwargs=self.dynamics_kwargs)
        # self.base_model.observations.Vs = self.dynamics.Vs
        # self.base_model.observations.As = self.dynamics.As
        # self.base_model.observations.Sigmas = self.dynamics.Sigmas
        self.base_model = ObservedRamping(
            self.K,
            self.D,
            M=self.M,
            transitions=copy.deepcopy(self.transitions),
            observations=copy.deepcopy(self.dynamics),
        )
        self.emissions.initialize(self.base_model, datas, inputs, masks, tags)


def simulate_ramping(
    beta=np.linspace(-0.02, 0.02, 5), w2=3e-3, x0=0.5, C=40, T=100, bin_size=0.01
):
    """Simulate synthetic spike-count trials from a ramping accumulator model.

    Generates ``T`` trials of a 1-D drift-diffusion accumulator with coherence-
    dependent drift rate and an absorbing upper boundary at 1.  Spike counts
    are drawn from a Poisson distribution with rate proportional to
    :math:`\\log(1 + \\exp(C x_t))`.

    The ``NC = 5`` coherence levels are evenly distributed across trials.
    Trial lengths are drawn uniformly from ``[50, 100)`` time bins.

    Args:
        beta (np.ndarray): Ramp rates for each coherence level, shape
            ``(NC,)``. Defaults to a linear grid from ``-0.02`` to ``0.02``
            with 5 levels.
        w2 (float): Diffusion variance per time step. Defaults to ``3e-3``.
        x0 (float): Mean initial accumulator value. Defaults to ``0.5``.
        C (float): Poisson gain (firing-rate scale factor). Defaults to
            ``40``.
        T (int): Total number of trials to simulate. Defaults to ``100``.
        bin_size (float): Spike-count bin size in seconds. Defaults to
            ``0.01``.

    Returns:
        tuple: A 6-tuple ``(ys, xs, zs, us, tr_lengths, trial_cohs)`` where:

        * **ys** (*list*): Spike-count arrays, each of shape
          ``(T_i, 1)``.
        * **xs** (*list*): Latent accumulator trajectories, each of shape
          ``(T_i, 1)``.
        * **zs** (*list*): Discrete state sequences (0 = ramp, 1 = bound),
          each of shape ``(T_i, 1)``.
        * **us** (*list*): One-hot coherence input matrices, each of shape
          ``(T_i, NC)``.
        * **tr_lengths** (*np.ndarray*): Trial lengths in time bins, shape
          ``(T,)``.
        * **trial_cohs** (*np.ndarray*): Coherence index for each trial,
          shape ``(T,)``.
    """

    NC = 5  # number of trial types
    cohs = np.arange(NC)
    trial_cohs = np.repeat(cohs, int(T / NC))
    tr_lengths = np.random.randint(50, size=(T)) + 50
    us = []
    xs = []
    zs = []
    ys = []
    for t in range(T):
        tr_coh = trial_cohs[t]
        betac = beta[tr_coh]

        tr_length = tr_lengths[t]
        x = np.zeros(tr_length)
        z = np.zeros(tr_length)
        x[0] = x0 + np.sqrt(w2) * npr.randn()
        z[0] = 0
        for i in np.arange(1, tr_length):
            if x[i - 1] >= 1.0:
                x[i] = 1.0
                z[i] = 1
            else:
                x[i] = np.min((1.0, x[i - 1] + betac + np.sqrt(w2) * npr.randn()))
                if x[i] >= 1.0:
                    z[i] = 1
                else:
                    z[i] = 0

        y = npr.poisson(np.log1p(np.exp(C * x)) * bin_size)

        u = np.tile(one_hot(tr_coh, 5), (tr_length, 1))
        us.append(u)
        xs.append(x.reshape((tr_length, 1)))
        zs.append(z.reshape((tr_length, 1)))
        ys.append(y.reshape((tr_length, 1)))

    return ys, xs, zs, us, tr_lengths, trial_cohs
