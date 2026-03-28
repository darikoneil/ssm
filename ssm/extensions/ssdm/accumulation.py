"""Accumulation and diffusion-style latent-state models built on top of ``ssm``.

This module defines a family of HMM/SLDS components for evidence accumulation
problems, including DDM-like boundaries, race models, collapsing boundaries,
and several observation/emission parameterizations.

Shape conventions used throughout:
- ``T``: number of time steps in a single trial.
- ``N``: number of observed channels (e.g., neurons).
- ``K``: number of discrete states.
- ``D``: latent continuous dimensionality.
- ``M``: input dimensionality.

Most classes override pieces of the ``ssm`` API. In several transition classes,
parameters are intentionally fixed (``params`` returns empty tuple) and
``m_step``/``initialize`` are no-ops to preserve task-specific dynamics.
"""

import copy

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import value_and_grad
from autograd.scipy.special import logsumexp
from ssm import hmm
from ssm.emissions import GaussianEmissions, PoissonEmissions
from ssm.hmm import HMM
from ssm.init_state_distns import InitialStateDistribution
from ssm.lds import SLDS
from ssm.observations import AutoRegressiveDiagonalNoiseObservations, Observations
from ssm.optimizers import (
    bfgs,
    lbfgs,
    sgd_step,
)
from ssm.preprocessing import (
    factor_analysis_with_imputation,
    interpolate_data,
)
from ssm.transitions import RecurrentOnlyTransitions, RecurrentTransitions, Transitions
from ssm.util import ensure_args_are_lists
from ssmdm.misc import smooth
from tqdm import tqdm
from tqdm.auto import trange


class AccumulationRaceTransitions(RecurrentTransitions):
    """Hard race-model transitions for ``K = D + 1`` discrete states.

    State semantics:
    - ``0``: accumulation state.
    - ``1..D``: boundary/commitment states; state ``d`` activates when latent
      dimension ``x_d`` crosses the decision threshold.

    Transition logits depend on the previous continuous state through ``Rs``
    with a large ``scale`` to approximate hard thresholding.
    """

    def __init__(self, K, D, M=0, scale=200):
        """Create fixed race transitions.

        Args:
            K: Number of discrete states; must equal ``D + 1``.
            D: Number of accumulation dimensions.
            M: Input dimensionality.
            scale: Boundary sharpness in transition logits.
        """
        assert K == D + 1
        assert D >= 1
        super().__init__(K, D, M)

        # "Race" transitions with D+1 states
        # Transition to state d when x_d > 1.0
        # State 0 is the accumulation state
        # scale determines sharpness of the threshold
        # Transitions out of boundary states occur w/ very low probability
        top_row = np.concatenate(([0.0], -scale * np.ones(D)))
        rest_rows = np.hstack(
            (
                -scale * np.ones((D, 1)),
                -scale * np.ones((D, D)) + np.diag(2.0 * scale * np.ones(D)),
            )
        )
        self.log_Ps = np.vstack((top_row, rest_rows))
        self.Ws = np.zeros((K, M))
        self.Rs = np.vstack((np.zeros(D), scale * np.eye(D)))

    @property
    def params(self):
        """Return trainable parameters.

        Returns:
            Empty tuple because this transition model is fixed.
        """
        return ()

    @params.setter
    def params(self, value):
        """Ignore parameter updates for fixed transitions."""

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        """Initialize transition parameters.

        Args:
            datas: Observation sequences.
            inputs: Optional input sequences.
            masks: Optional observation masks.
            tags: Optional metadata tags.

        Returns:
            None. This method intentionally performs no initialization.
        """

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        """Update transition parameters.

        Args:
            expectations: Posterior expectations from E-step.
            datas: Observation sequences.
            inputs: Input sequences.
            masks: Observation masks.
            tags: Metadata tags.
            **kwargs: Additional optimizer arguments.

        Returns:
            None. This method intentionally does not update parameters.
        """


class AccumulationRaceSoftTransitions(RecurrentOnlyTransitions):
    """Soft race-model transitions using only recurrent latent effects.

    Similar to :class:`AccumulationRaceTransitions`, but allows smoother
    transitions out of boundary states by using ``RecurrentOnlyTransitions``.
    """

    def __init__(self, K, D, M=0, scale=100):
        """Create soft race transitions.

        Args:
            K: Number of discrete states; must equal ``D + 1``.
            D: Number of accumulation dimensions.
            M: Input dimensionality.
            scale: Boundary sharpness in recurrent logits.
        """
        assert K == D + 1
        assert D >= 1
        super().__init__(K, D, M)

        # Like Race Transitions but soft boundaries
        # Transitions depend on previous x only
        # Transition to state d when x_d > 1.0
        self.Ws = np.zeros((K, M))
        self.Rs = np.vstack((np.zeros(D), scale * np.eye(D)))
        self.r = np.concatenate(([0], -scale * np.ones(D)))

    @property
    def params(self):
        """Return trainable parameters (none for this fixed form)."""
        return ()

    @params.setter
    def params(self, value):
        """Ignore external parameter assignment."""

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        """Initialize transition parameters.

        Args:
            datas: Observation sequences.
            inputs: Optional input sequences.
            masks: Optional observation masks.
            tags: Optional metadata tags.

        Returns:
            None. This method intentionally performs no initialization.
        """

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        """Update transition parameters.

        Args:
            expectations: Posterior expectations from E-step.
            datas: Observation sequences.
            inputs: Input sequences.
            masks: Observation masks.
            tags: Metadata tags.
            **kwargs: Additional optimizer arguments.

        Returns:
            None. This method intentionally does not update parameters.
        """


class DDMTransitions(RecurrentTransitions):
    """Hard-boundary drift-diffusion transitions with three states.

    Discrete states are ``{accumulate, upper-boundary, lower-boundary}`` and
    are driven by a 1D latent state.
    """

    def __init__(self, K, D, M=0, scale=200):
        """Create fixed DDM transition logits.

        Args:
            K: Number of states; must be 3.
            D: Latent dimensionality; must be 1.
            M: Input dimensionality.
            scale: Boundary sharpness.
        """
        assert K == 3
        assert D == 1
        # assert M == 1
        super().__init__(K, D, M)

        # DDM has one accumulation state and boundary states at +/- 1
        self.log_Ps = -scale * np.ones((K, K)) + np.diag(
            np.concatenate(([scale], 2.0 * scale * np.ones(K - 1)))
        )
        self.Ws = np.zeros((K, M))
        self.Rs = np.array([0, scale, -scale]).reshape((3, 1))

    @property
    def params(self):
        """Return empty parameters for fixed transition dynamics."""
        return ()

    @params.setter
    def params(self, value):
        """Ignore parameter setting for fixed DDM transitions."""

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        """Initialize transition parameters.

        Args:
            datas: Observation sequences.
            inputs: Optional input sequences.
            masks: Optional observation masks.
            tags: Optional metadata tags.

        Returns:
            None. This method intentionally performs no initialization.
        """

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        """Update transition parameters.

        Args:
            expectations: Posterior expectations from E-step.
            datas: Observation sequences.
            inputs: Input sequences.
            masks: Observation masks.
            tags: Metadata tags.
            **kwargs: Additional optimizer arguments.

        Returns:
            None. This method intentionally does not update parameters.
        """


class DDMSoftTransitions(RecurrentOnlyTransitions):
    """Soft DDM transitions allowing escape from boundary states."""

    def __init__(self, K, D, M=0, scale=200):
        """Create soft-boundary DDM transitions.

        Args:
            K: Number of states; must be 3.
            D: Latent dimensionality; must be 1.
            M: Input dimensionality.
            scale: Boundary sharpness in recurrent term.
        """
        assert K == 3
        assert D == 1
        super().__init__(K, D, M)

        # DDM where transitions out of boundary state can occur
        self.Ws = np.zeros((3, M))
        self.Rs = np.array([0, scale, -scale]).reshape((3, 1))
        self.r = np.array([0, -scale, -scale])

    @property
    def params(self):
        """Return empty parameters for fixed transition structure."""
        return ()

    @params.setter
    def params(self, value):
        """Ignore parameter updates."""

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        """Initialize transition parameters.

        Args:
            datas: Observation sequences.
            inputs: Optional input sequences.
            masks: Optional observation masks.
            tags: Optional metadata tags.

        Returns:
            None. This method intentionally performs no initialization.
        """

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        """Update transition parameters.

        Args:
            expectations: Posterior expectations from E-step.
            datas: Observation sequences.
            inputs: Input sequences.
            masks: Observation masks.
            tags: Metadata tags.
            **kwargs: Additional optimizer arguments.

        Returns:
            None. This method intentionally does not update parameters.
        """


class DDMCollapsingTransitions(RecurrentTransitions):
    """DDM transitions with linearly input-driven collapsing boundaries."""

    def __init__(self, K, D, M=0, scale=200):
        """Create collapsing-boundary DDM transitions.

        Args:
            K: Number of states; must be 3.
            D: Latent dimensionality; must be 1.
            M: Input dimensionality.
            scale: Baseline boundary sharpness.
        """
        assert K == 3
        assert D == 1
        # assert M == 1
        super().__init__(K, D, M)

        # Parameters linking past observations to state distribution
        self.log_Ps = -scale * np.ones((K, K)) + np.diag(
            np.concatenate(([scale], 2.0 * scale * np.ones(K - 1)))
        )
        self.bound_scale = 0.008  # 0.01
        self.Ws = self.bound_scale * scale * np.eye(K, M)
        self.Ws[0][0] = 0.0
        self.Rs = np.array([0, scale, -scale]).reshape((3, 1))

    @property
    def params(self):
        """Return empty parameters for fixed transition form."""
        return ()

    @params.setter
    def params(self, value):
        """Ignore parameter assignment."""

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        """Initialize transition parameters.

        Args:
            datas: Observation sequences.
            inputs: Optional input sequences.
            masks: Optional observation masks.
            tags: Optional metadata tags.

        Returns:
            None. This method intentionally performs no initialization.
        """

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        """Update transition parameters.

        Args:
            expectations: Posterior expectations from E-step.
            datas: Observation sequences.
            inputs: Input sequences.
            masks: Observation masks.
            tags: Metadata tags.
            **kwargs: Additional optimizer arguments.

        Returns:
            None. This method intentionally does not update parameters.
        """


class DDMNonlinearCollapsingTransitions(RecurrentTransitions):
    """DDM transitions with nonlinear (time-dependent) boundary collapse."""

    def __init__(self, K, D, M=0, scale=200):
        """Create nonlinear collapsing-boundary transitions.

        Args:
            K: Number of states; must be 3.
            D: Latent dimensionality; must be 1.
            M: Input dimensionality.
            scale: Baseline boundary sharpness.
        """
        assert K == 3
        assert D == 1
        super().__init__(K, D, M)

        # Parameters linking past observations to state distribution
        self.log_Ps = -scale * np.ones((K, K)) + np.diag(
            np.concatenate(([scale], 2.0 * scale * np.ones(K - 1)))
        )
        self.Ws = scale * np.eye(K, M)
        self.Ws[0][0] = 0.0
        self.Rs = np.array([0, scale, -scale]).reshape((3, 1))

        self.ap = np.log(0.75)  # 0.5 - np.exp(ap) is value the boundary collapses to
        self.lamb = 50.0

    @property
    def params(self):
        """Return trainable transition parameters.

        Returns:
            tuple: Empty tuple in the current fixed-parameter configuration.
        """
        #     return (self.ap, self.lamb)
        return ()

    @params.setter
    def params(self, value):
        """Set transition parameters.

        Args:
            value: Ignored in the current fixed-parameter configuration.
        """
        #     self.ap, self.lamb = value

    def log_transition_matrices(self, data, input, mask, tag):
        """Compute normalized log transition matrices for each time step.

        Args:
            data: Continuous latent sequence with shape ``(T, D)``.
            input: Input sequence with shape ``(T, M)``.
            mask: Unused mask argument kept for API compatibility.
            tag: Unused metadata argument kept for API compatibility.

        Returns:
            np.ndarray: Array of shape ``(T - 1, K, K)`` containing log
            transition probabilities.
        """

        def bound_func(t, a, ap, lamb, k):
            return a - (1 - np.exp(-((t / lamb) ** k))) * (0.0 * a + np.exp(ap))

        T, D = data.shape
        # Previous state effect
        log_Ps = np.tile(self.log_Ps[None, :, :], (T - 1, 1, 1))
        # Input effect
        boundary_input = 1.0 - bound_func(input[1:], 1.0, self.ap, self.lamb, 2)
        log_Ps = log_Ps + np.dot(boundary_input, self.Ws.T)[:, None, :]
        # Past observations effect
        log_Ps = log_Ps + np.dot(data[:-1], self.Rs.T)[:, None, :]
        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        """Initialize transition parameters.

        Args:
            datas: Observation sequences.
            inputs: Optional input sequences.
            masks: Optional observation masks.
            tags: Optional metadata tags.

        Returns:
            None. This method intentionally performs no initialization.
        """

    def m_step(
        self, expectations, datas, inputs, masks, tags, optimizer="lbfgs", **kwargs
    ):
        """Run generic transition M-step via :class:`ssm.transitions.Transitions`.

        Args:
            expectations: Posterior expectations from E-step.
            datas: Observation sequences.
            inputs: Input sequences.
            masks: Observation masks.
            tags: Metadata tags.
            optimizer: Optimizer name passed to the base implementation.
            **kwargs: Additional optimizer arguments.

        Returns:
            None. Updates transition parameters through the base method.
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


class AccumulationObservations(AutoRegressiveDiagonalNoiseObservations):
    """Task-structured AR dynamics used as SLDS continuous-state dynamics.

    This class parameterizes accumulation dynamics in state 0 and fixed
    near-identity boundary dynamics in the remaining states.
    """

    def __init__(
        self, K, D, M, lags=1, learn_A=True, learn_V=False, learn_sig_init=False
    ):
        """Create accumulation dynamics with optional learning controls.

        Args:
            K: Number of discrete states.
            D: Continuous latent dimensionality.
            M: Input dimensionality.
            lags: Included for API compatibility; implementation uses lag 1.
            learn_A: If ``True``, learn accumulation-state autoregressive gains.
            learn_V: If ``True``, learn extra input covariate weights.
            learn_sig_init: If ``True``, learn initial-state noise parameters.
        """
        super().__init__(K, D, M)

        # diagonal dynamics for each state
        # only learn dynamics for accumulation state
        self.learn_A = learn_A
        self._a_diag = np.ones((D, 1))
        if self.learn_A:
            mask1 = np.vstack(
                (np.eye(D)[None, :, :], np.zeros((K - 1, D, D)))
            )  # for accum state
            mask2 = np.vstack((np.zeros((1, D, D)), np.tile(np.eye(D), (K - 1, 1, 1))))
            self._As = self._a_diag * mask1 + mask2
        else:
            self._As = np.tile(np.eye(D), (K, 1, 1))

        # set input Accumulation params, one for each dimension
        # first D inputs are accumulated in different dimensions
        # rest of M-D inputs are applied to each dimension
        self._betas = 0.1 * np.ones(
            D,
        )
        self.learn_V = learn_V
        self._V = 0.0 * np.ones((D, M - D))  # additional covariates, if they exist
        self.Vs[0] = np.hstack((self._betas * np.eye(D, D), self._V))
        for d in range(1, K):
            self.Vs[d] *= np.zeros((D, M))

        # set noise variances
        self.accum_log_sigmasq = np.log(1e-3) * np.ones(
            D,
        )
        mask1 = np.vstack(
            (
                np.ones(
                    D,
                ),
                np.zeros((K - 1, D)),
            )
        )
        mask2 = np.vstack((np.zeros(D), np.ones((K - 1, D))))
        self.bound_variance = 1e-4
        self._log_sigmasq = (
            self.accum_log_sigmasq * mask1 + np.log(self.bound_variance) * mask2
        )
        self.learn_sig_init = learn_sig_init
        if self.learn_sig_init:
            self.accum_log_sigmasq_init = np.log(1e-3) * np.ones(
                D,
            )
            self._log_sigmasq_init = (
                self.accum_log_sigmasq_init * mask1
                + np.log(self.bound_variance) * mask2
            )
        else:
            self._log_sigmasq_init = (
                self.accum_log_sigmasq + np.log(2)
            ) * mask1 + np.log(self.bound_variance) * mask2

        # Set the remaining parameters to fixed values
        self.bs = np.zeros((K, D))
        acc_mu_init = np.zeros((1, D))
        self.mu_init = np.vstack((acc_mu_init, np.ones((K - 1, D))))

    @property
    def params(self):
        """Pack learnable parameters into the expected tuple format."""
        params = self._betas, self.accum_log_sigmasq
        params = params + (self._a_diag,) if self.learn_A else params
        params = params + (self._V,) if self.learn_V else params
        params = (
            params + (self.accum_log_sigmasq_init) if self.learn_sig_init else params
        )
        return params

    @params.setter
    def params(self, value):
        """Unpack parameter tuple and rebuild derived tensors (A, V, Sigma)."""
        self._betas, self.accum_log_sigmasq = value[:2]
        if self.learn_A:
            self._a_diag = value[2]
        if self.learn_V:
            self._V = value[-1]
        # TODO fix above
        if self.learn_sig_init:
            self.accum_log_sigmasq_init = value[-1]

        K, D, M = self.K, self.D, self.M

        # update V
        mask0 = np.hstack((np.eye(D), np.ones((D, M - D))))  # state K = 0
        mask = np.vstack((mask0[None, :, :], np.zeros((K - 1, D, M))))

        # self.Vs = self._betas * mask
        self.Vs = np.hstack((np.diag(self._betas), self._V)) * mask

        # update sigmas
        mask1 = np.vstack(
            (
                np.ones(
                    D,
                ),
                np.zeros((K - 1, D)),
            )
        )
        mask2 = np.vstack((np.zeros(D), np.ones((K - 1, D))))
        self._log_sigmasq = (
            self.accum_log_sigmasq * mask1 + np.log(self.bound_variance) * mask2
        )
        # self._log_sigmasq_init = (self.accum_log_sigmasq + np.log(2) )* mask1 + np.log(self.bound_variance) * mask2
        if self.learn_sig_init:
            self.accum_log_sigmasq_init = np.log(1e-3) * np.ones(
                D,
            )
            self._log_sigmasq_init = (
                self.accum_log_sigmasq_init * mask1
                + np.log(self.bound_variance) * mask2
            )
        else:
            self._log_sigmasq_init = (
                self.accum_log_sigmasq + np.log(2)
            ) * mask1 + np.log(self.bound_variance) * mask2

        # update A
        # if self.learn_A:
        mask1 = np.vstack(
            (np.eye(D)[None, :, :], np.zeros((K - 1, D, D)))
        )  # for accum state
        mask2 = np.vstack((np.zeros((1, D, D)), np.tile(np.eye(D), (K - 1, 1, 1))))
        self._As = self._a_diag * mask1 + mask2

    # @property
    # def betas(self):
    #     return self._betas
    #
    # @betas.setter
    # def betas(self, value):
    #     assert value.shape == (self.D,)
    #     self._betas = value
    #     mask = np.vstack((np.eye(self.D)[None,:,:], np.zeros((self.K-1,self.D,self.D))))
    #     self.Vs = self._betas * mask

    def log_prior(self):
        """Return log-prior contribution for accumulation-state process noise.

        Uses an inverse-gamma-style prior on variances (in log-space).
        """
        alpha = 1.1  # or 0.02
        beta = 1e-3  # or 0.02
        dyn_vars = np.exp(self.accum_log_sigmasq)
        var_prior = np.sum(-(alpha + 1) * np.log(dyn_vars) - np.divide(beta, dyn_vars))
        return var_prior

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        """Initialize observation parameters.

        Args:
            datas: Observation sequences.
            inputs: Optional input sequences.
            masks: Optional observation masks.
            tags: Optional metadata tags.

        Returns:
            None. This method intentionally performs no initialization.
        """

    # def m_step(self, expectations, datas, inputs, masks, tags,
    # continuous_expectations=None, **kwargs):
    # Observations.m_step(self, expectations, datas, inputs, masks, tags, **kwargs)

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
        """Update accumulation dynamics parameters from expected sufficient statistics.

        Args:
            expectations: Discrete-state posterior expectations from E-step.
            datas: Observed latent trajectories, each with shape ``(T, D)``.
            inputs: Input trajectories, each with shape ``(T, M)``.
            masks: Unused mask inputs for compatibility.
            tags: Unused trial metadata for compatibility.
            continuous_expectations: Optional precomputed continuous expectations.
            **kwargs: Extra keyword arguments accepted for API compatibility.

        Returns:
            None. Updates ``self.params`` in place.
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
        else:
            ExuxuTs, ExuyTs, EyyTs, Ens = self._extend_given_sufficient_statistics(
                expectations, continuous_expectations, inputs
            )

        # remove bias block
        ExuxuTs = ExuxuTs[:, :-1, :-1]
        ExuyTs = ExuyTs[:, :-1, :]

        # initialize new parameters
        a_diag = np.zeros_like(self._a_diag)
        betas = np.zeros_like(self._betas)
        accum_log_sigmasq = np.zeros_like(self.accum_log_sigmasq)
        # V = np.zeros_like(self._V)

        # this only works if input and latent dimensions are same
        assert self.D == self.M
        assert self.learn_V is False

        # Solve for each dimension separately and for the first state only.
        # Other states have no dynamics parameters.
        for d in range(D):
            # get relevant dimensions of expections
            ExuxuTs_d = np.array(
                [
                    [ExuxuTs[0, d, d], ExuxuTs[0, d, d + D]],
                    [ExuxuTs[0, d + D, d], ExuxuTs[0, d + D, d + D]],
                ]
            )

            ExuyTs_d = ExuyTs[0, [d, d + D], d]

            if self.learn_A:
                W = np.linalg.solve(ExuxuTs_d, ExuyTs_d).T
                a_diag[d] = W[0]
                betas[d] = W[1]

            else:
                # V_d = E[u_t^2]^{-1} * (E[y_t u_t]  - E[x_{t-1} u_t])
                Euu_d = ExuxuTs[0, d + D, d + D]
                Exu_d = ExuxuTs[0, d, d + D]
                Eyu_d = ExuyTs[0, [d + D], d]
                betas[d] = 1.0 / Euu_d * (Eyu_d - Exu_d)
                W = np.array([1.0, betas[d]])

            # Solve for the MAP estimate of the covariance
            sqerr = EyyTs[0, d, d] - 2 * W @ ExuyTs_d + W @ ExuxuTs_d @ W.T
            nu = nu0 + Ens[0]
            accum_log_sigmasq[d] = np.log((sqerr + Psi0) / (nu + d + 1))

        # set based on variance of sig2
        params = betas, accum_log_sigmasq
        params = params + (a_diag,) if self.learn_A else params

        if self.learn_sig_init:
            accum_log_sigmasq_init = np.zeros(self.D)
            for d in range(self.D):
                x0_d = np.array([data[0, d] for data in datas])
                sqerr_d = np.sum(x0_d**2)
                accum_log_sigmasq_init[d] = np.log(sqerr_d)
            params = params + (self.accum_log_sigmasq_init)

        self.params = params

        return


class AccumulationInputNoiseObservations(AutoRegressiveDiagonalNoiseObservations):
    """Variant of accumulation dynamics with input-driven noise usage pattern."""

    def __init__(
        self, K, D, M, lags=1, learn_A=True, learn_V=False, learn_sig_init=False
    ):
        """Create input-noise accumulation dynamics.

        Args:
            K: Number of discrete states.
            D: Continuous latent dimensionality.
            M: Input dimensionality.
            lags: Included for API compatibility; implementation uses lag 1.
            learn_A: If ``True``, learn accumulation-state autoregressive gains.
            learn_V: If ``True``, learn additional covariate weights.
            learn_sig_init: If ``True``, learn initial-state noise parameters.
        """
        super().__init__(K, D, M)

        # diagonal dynamics for each state
        # only learn dynamics for accumulation state
        self.learn_A = learn_A
        self._a_diag = np.ones((D, 1))
        if self.learn_A:
            mask1 = np.vstack(
                (np.eye(D)[None, :, :], np.zeros((K - 1, D, D)))
            )  # for accum state
            mask2 = np.vstack((np.zeros((1, D, D)), np.tile(np.eye(D), (K - 1, 1, 1))))
            self._As = self._a_diag * mask1 + mask2
        else:
            self._As = np.tile(np.eye(D), (K, 1, 1))

        # set input Accumulation params, one for each dimension
        # first D inputs are accumulated in different dimensions
        # rest of M-D inputs are applied to each dimension
        self._betas = 0.1 * np.ones(
            D,
        )
        self.learn_V = learn_V
        self._V = 0.0 * np.ones((D, M - D))  # additional covariates, if they exist
        self.Vs[0] = np.hstack((self._betas * np.eye(D, D), self._V))
        for d in range(1, K):
            self.Vs[d] *= np.zeros((D, M))

        # set noise variances
        self.accum_log_sigmasq = np.log(1e-3) * np.ones(
            D,
        )
        mask1 = np.vstack(
            (
                np.ones(
                    D,
                ),
                np.zeros((K - 1, D)),
            )
        )
        mask2 = np.vstack((np.zeros(D), np.ones((K - 1, D))))
        self.bound_variance = 1e-4
        self._log_sigmasq = (
            self.accum_log_sigmasq * mask1 + np.log(self.bound_variance) * mask2
        )
        self.learn_sig_init = learn_sig_init
        if self.learn_sig_init:
            self.accum_log_sigmasq_init = np.log(1e-3) * np.ones(
                D,
            )
            self._log_sigmasq_init = (
                self.accum_log_sigmasq_init * mask1
                + np.log(self.bound_variance) * mask2
            )
        else:
            self._log_sigmasq_init = (
                self.accum_log_sigmasq + np.log(2)
            ) * mask1 + np.log(self.bound_variance) * mask2

        # Set the remaining parameters to fixed values
        self.bs = np.zeros((K, D))
        acc_mu_init = np.zeros((1, D))
        self.mu_init = np.vstack((acc_mu_init, np.ones((K - 1, D))))

    @property
    def params(self):
        """Pack learnable parameters.

        Returns:
            tuple: Tuple containing the currently learnable parameter blocks.
        """
        params = self._betas, self.accum_log_sigmasq
        params = params + (self._a_diag,) if self.learn_A else params
        params = params + (self._V,) if self.learn_V else params
        params = (
            params + (self.accum_log_sigmasq_init) if self.learn_sig_init else params
        )
        return params

    @params.setter
    def params(self, value):
        """Unpack parameter tuple and rebuild dependent matrices.

        Args:
            value: Parameter tuple in the same order as returned by ``params``.
        """
        self._betas, self.accum_log_sigmasq = value[:2]
        if self.learn_A:
            self._a_diag = value[2]
        if self.learn_V:
            self._V = value[-1]
        # TODO fix above
        if self.learn_sig_init:
            self.accum_log_sigmasq_init = value[-1]

        K, D, M = self.K, self.D, self.M

        # update V
        mask0 = np.hstack((np.eye(D), np.ones((D, M - D))))  # state K = 0
        mask = np.vstack((mask0[None, :, :], np.zeros((K - 1, D, M))))

        # self.Vs = self._betas * mask
        self.Vs = np.hstack((np.diag(self._betas), self._V)) * mask

        # update sigmas
        mask1 = np.vstack(
            (
                np.ones(
                    D,
                ),
                np.zeros((K - 1, D)),
            )
        )
        mask2 = np.vstack((np.zeros(D), np.ones((K - 1, D))))
        self._log_sigmasq = (
            self.accum_log_sigmasq * mask1 + np.log(self.bound_variance) * mask2
        )
        # self._log_sigmasq_init = (self.accum_log_sigmasq + np.log(2) )* mask1 + np.log(self.bound_variance) * mask2
        if self.learn_sig_init:
            self.accum_log_sigmasq_init = np.log(1e-3) * np.ones(
                D,
            )
            self._log_sigmasq_init = (
                self.accum_log_sigmasq_init * mask1
                + np.log(self.bound_variance) * mask2
            )
        else:
            self._log_sigmasq_init = (
                self.accum_log_sigmasq + np.log(2)
            ) * mask1 + np.log(self.bound_variance) * mask2

        # update A
        # if self.learn_A:
        mask1 = np.vstack(
            (np.eye(D)[None, :, :], np.zeros((K - 1, D, D)))
        )  # for accum state
        mask2 = np.vstack((np.zeros((1, D, D)), np.tile(np.eye(D), (K - 1, 1, 1))))
        self._As = self._a_diag * mask1 + mask2

    def log_prior(self):
        """Return log-prior term for process variances.

        Returns:
            float: Log prior contribution.
        """
        alpha = 1.1  # or 0.02
        beta = 1e-3  # or 0.02
        dyn_vars = np.exp(self.accum_log_sigmasq)
        var_prior = np.sum(-(alpha + 1) * np.log(dyn_vars) - np.divide(beta, dyn_vars))
        return var_prior

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        """Initialize observation parameters.

        Args:
            datas: Observation sequences.
            inputs: Optional input sequences.
            masks: Optional observation masks.
            tags: Optional metadata tags.

        Returns:
            None. This method intentionally performs no initialization.
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
        """Estimate state-0 AR/input/noise parameters from sufficient stats.

        Behavior mirrors :meth:`AccumulationObservations.m_step`.

        Args:
            expectations: Discrete-state posterior expectations from E-step.
            datas: Observed latent trajectories, each with shape ``(T, D)``.
            inputs: Input trajectories, each with shape ``(T, M)``.
            masks: Unused mask inputs for compatibility.
            tags: Unused trial metadata for compatibility.
            continuous_expectations: Optional precomputed continuous expectations.
            **kwargs: Extra keyword arguments accepted for API compatibility.

        Returns:
            None. Updates ``self.params`` in place.
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
        else:
            ExuxuTs, ExuyTs, EyyTs, Ens = self._extend_given_sufficient_statistics(
                expectations, continuous_expectations, inputs
            )

        # remove bias block
        ExuxuTs = ExuxuTs[:, :-1, :-1]
        ExuyTs = ExuyTs[:, :-1, :]

        # initialize new parameters
        a_diag = np.zeros_like(self._a_diag)
        betas = np.zeros_like(self._betas)
        accum_log_sigmasq = np.zeros_like(self.accum_log_sigmasq)
        # V = np.zeros_like(self._V)

        # this only works if input and latent dimensions are same
        assert self.D == self.M
        assert self.learn_V is False

        # Solve for each dimension separately and for the first state only.
        # Other states have no dynamics parameters.
        for d in range(D):
            # get relevant dimensions of expections
            ExuxuTs_d = np.array(
                [
                    [ExuxuTs[0, d, d], ExuxuTs[0, d, d + D]],
                    [ExuxuTs[0, d + D, d], ExuxuTs[0, d + D, d + D]],
                ]
            )

            ExuyTs_d = ExuyTs[0, [d, d + D], d]

            if self.learn_A:
                W = np.linalg.solve(ExuxuTs_d, ExuyTs_d).T
                a_diag[d] = W[0]
                betas[d] = W[1]

            else:
                # V_d = E[u_t^2]^{-1} * (E[y_t u_t]  - E[x_{t-1} u_t])
                Euu_d = ExuxuTs[0, d + D, d + D]
                Exu_d = ExuxuTs[0, d, d + D]
                Eyu_d = ExuyTs[0, [d + D], d]
                betas[d] = 1.0 / Euu_d * (Eyu_d - Exu_d)
                W = np.array([1.0, betas[d]])

            # Solve for the MAP estimate of the covariance
            sqerr = EyyTs[0, d, d] - 2 * W @ ExuyTs_d + W @ ExuxuTs_d @ W.T
            nu = nu0 + Ens[0]
            accum_log_sigmasq[d] = np.log((sqerr + Psi0) / (nu + d + 1))

        # set based on variance of sig2
        params = betas, accum_log_sigmasq
        params = params + (a_diag,) if self.learn_A else params

        if self.learn_sig_init:
            accum_log_sigmasq_init = np.zeros(self.D)
            for d in range(self.D):
                x0_d = np.array([data[0, d] for data in datas])
                sqerr_d = np.sum(x0_d**2)
                accum_log_sigmasq_init[d] = np.log(sqerr_d)
            params = params + (self.accum_log_sigmasq_init)

        self.params = params

        return


# class AccumulationObservations(AutoRegressiveDiagonalNoiseObservations):
# TODO: Add in new dynamics here for additional dimensions.
# Compose?
class EmbeddedAccumulationObservations(AutoRegressiveDiagonalNoiseObservations):
    """Accumulation dynamics split into accumulation and embedded subspaces.

    The latent space is partitioned into ``D_acc`` task-relevant accumulation
    dimensions and ``D_emb`` additional embedding dimensions.
    """

    def __init__(self, K, D, M, lags=1, D_emb=None, D_acc=None):
        """Create embedded accumulation dynamics.

        Args:
            K: Number of discrete states.
            D: Total latent dimensionality.
            M: Input dimensionality.
            lags: Included for compatibility; lag-1 dynamics are used.
            D_emb: Number of embedding dimensions.
            D_acc: Number of accumulation dimensions.
        """
        super().__init__(K, D, M)

        assert D_emb + D_acc == D, "accum + emb dims must equal total dims"
        self.D_acc = D_acc
        self.D_emb = D_emb

        # diagonal dynamics for each state
        # only learn accumulation dynamics for accumulation state
        self._a_diag = np.ones((D, 1))
        mask1 = np.vstack(
            (np.eye(D)[None, :, :], np.zeros((K - 1, D, D)))
        )  # for accum state
        mask2 = np.vstack((np.zeros((1, D, D)), np.tile(np.eye(D), (K - 1, 1, 1))))
        self._As = self._a_diag * mask1 + mask2

        # set input Accumulation params, one for each dimension
        # first D inputs are accumulated in different dimensions
        # rest of M-D inputs are applied to each dimension
        ### This is original, for first D_acc dims
        # self._betas = 0.1*np.ones(D_acc,)
        # r1 = self._betas*np.eye(D_acc, D_acc)
        # r2 = np.zeros((self.D_emb, M)) # is D_emb rows by input # of columns
        # self.Vs[0] = np.vstack((r1, r2))
        ### This is new, for all dims
        self._betas = 0.1 * np.ones(
            M,
        )
        self.Vs[0] = self._betas * np.eye(D, M)

        for d in range(1, K):
            self.Vs[d] *= np.zeros((D, M))

        # set noise variances
        self.accum_log_sigmasq = np.log(1e-3) * np.ones(
            D,
        )
        mask1 = np.vstack(
            (
                np.ones(
                    D,
                ),
                np.zeros((K - 1, D)),
            )
        )
        mask2 = np.vstack((np.zeros(D), np.ones((K - 1, D))))
        self.bound_variance = 1e-4
        self._log_sigmasq = (
            self.accum_log_sigmasq * mask1 + np.log(self.bound_variance) * mask2
        )
        self._log_sigmasq_init = (self.accum_log_sigmasq + np.log(2)) * mask1 + np.log(
            self.bound_variance
        ) * mask2

        # Set the remaining parameters to fixed values
        self.bs = np.zeros((K, D))
        acc_mu_init = np.zeros((1, D))
        self.mu_init = np.vstack((acc_mu_init, np.ones((K - 1, D))))

    @property
    def params(self):
        """Pack learnable parameters for optimization."""
        params = self._betas, self.accum_log_sigmasq, self._a_diag
        return params

    @params.setter
    def params(self, value):
        """Unpack parameters and rebuild A/V/sigma tensors."""
        self._betas, self.accum_log_sigmasq, self._a_diag = value

        K, D, M = self.K, self.D, self.M

        # Update V
        ### This is original, for first D_acc dims
        # k1 = np.vstack((self._betas * np.eye(self.D_acc), np.zeros((self.D_emb,self.D_acc)))) # state K = 0
        # self.Vs = np.vstack((k1[None,:,:], np.zeros((K-1,D,M))))
        ### This is new, for all dims
        k1 = self._betas * np.eye(self.D, self.M)
        self.Vs = np.vstack((k1[None, :, :], np.zeros((K - 1, D, M))))

        # update sigmas
        mask1 = np.vstack(
            (
                np.ones(
                    D,
                ),
                np.zeros((K - 1, D)),
            )
        )
        mask2 = np.vstack((np.zeros(D), np.ones((K - 1, D))))
        self._log_sigmasq = (
            self.accum_log_sigmasq * mask1 + np.log(self.bound_variance) * mask2
        )
        self._log_sigmasq_init = (self.accum_log_sigmasq + np.log(2)) * mask1 + np.log(
            self.bound_variance
        ) * mask2

        # update A
        # if self.learn_A:
        mask1 = np.vstack(
            (np.eye(D)[None, :, :], np.zeros((K - 1, D, D)))
        )  # for accum state
        mask2 = np.vstack((np.zeros((1, D, D)), np.tile(np.eye(D), (K - 1, 1, 1))))
        self._As = self._a_diag * mask1 + mask2

    def log_prior(self):
        """Return variance prior term used during optimization.

        Returns:
            float: Log prior contribution.
        """
        alpha = 1.1  # or 0.02
        beta = 1e-3  # or 0.02
        dyn_vars = np.exp(self.accum_log_sigmasq)
        var_prior = np.sum(-(alpha + 1) * np.log(dyn_vars) - np.divide(beta, dyn_vars))
        return var_prior

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        """Initialize observation parameters.

        Args:
            datas: Observation sequences.
            inputs: Optional input sequences.
            masks: Optional observation masks.
            tags: Optional metadata tags.

        Returns:
            None. This method intentionally performs no initialization.
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
        """Use base observation M-step implementation from ``ssm``.

        Args:
            expectations: Posterior expectations from E-step.
            datas: Observation sequences.
            inputs: Input sequences.
            masks: Mask sequences.
            tags: Metadata tags.
            continuous_expectations: Optional continuous expectations (unused).
            **kwargs: Additional optimizer arguments.

        Returns:
            None.
        """
        Observations.m_step(self, expectations, datas, inputs, masks, tags, **kwargs)


class EmbeddedAccumulationRaceTransitions(RecurrentTransitions):
    """Race transitions that ignore embedded dimensions in boundary logic."""

    def __init__(self, K, D, M=0, scale=200, D_acc=None, D_emb=None):
        """Create race transitions for mixed accumulation/embedding latents.

        Args:
            K: Number of states; must equal ``D_acc + 1``.
            D: Total latent dimensionality (``D_acc + D_emb``).
            M: Input dimensionality.
            scale: Boundary sharpness.
            D_acc: Number of accumulation dimensions used for boundaries.
            D_emb: Number of embedding dimensions ignored by boundaries.
        """
        assert D_acc + 1 == K
        assert D_acc >= 1
        assert D_acc + D_emb == D
        super().__init__(K, D, M)

        # "Race" transitions with D+1 states
        # Transition to state d when x_d > 1.0
        # State 0 is the accumulation state
        # scale determines sharpness of the threshold
        # Transitions out of boundary states occur w/ very low probability
        top_row = np.concatenate(([0.0], -scale * np.ones(D_acc)))
        rest_rows = np.hstack(
            (
                -scale * np.ones((D_acc, 1)),
                -scale * np.ones((D_acc, D_acc))
                + np.diag(2.0 * scale * np.ones(D_acc)),
            )
        )
        self.log_Ps = np.vstack((top_row, rest_rows))
        self.Ws = np.zeros((K, M))
        R1 = np.vstack((np.zeros(D_acc), scale * np.eye(D_acc)))
        R2 = np.zeros((K, D_emb))
        self.Rs = np.hstack((R1, R2))

    @property
    def params(self):
        """Return empty tuple; transition parameters are fixed."""
        return ()

    @params.setter
    def params(self, value):
        """Ignore parameter updates."""

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        """Initialize transition parameters.

        Args:
            datas: Observation sequences.
            inputs: Optional input sequences.
            masks: Optional observation masks.
            tags: Optional metadata tags.

        Returns:
            None. This method intentionally performs no initialization.
        """

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        """Update transition parameters.

        Args:
            expectations: Posterior expectations from E-step.
            datas: Observation sequences.
            inputs: Input sequences.
            masks: Observation masks.
            tags: Metadata tags.
            **kwargs: Additional optimizer arguments.

        Returns:
            None. This method intentionally does not update parameters.
        """


# class EmbeddedAccumulationPoissonEmissions():


class AccumulationGLMObservations(AutoRegressiveDiagonalNoiseObservations):
    """GLM-style accumulation dynamics with state-specific AR structure."""

    def __init__(self, K, D, M, lags=1):
        """Create GLM-like observation dynamics.

        Args:
            K: Number of discrete states.
            D: Latent dimensionality.
            M: Input dimensionality.
            lags: Included for API compatibility.
        """
        super().__init__(K, D, M)

        # diagonal dynamics for each state
        # only learn dynamics for accumulation state
        a_diag = np.ones((D, 1))
        self._a_diag = a_diag
        mask1 = np.vstack(
            (np.eye(D)[None, :, :], np.zeros((K - 1, D, D)))
        )  # for accum state
        mask2 = np.vstack((np.zeros((1, D, D)), np.tile(np.eye(D), (K - 1, 1, 1))))
        self._As = self._a_diag * mask1 + mask2

        # set input Accumulation params, one for each dimension
        # They only differ in their input
        self._V0 = np.zeros((D, M))
        self.Vs[0] = self._V0  # ramp
        for d in range(1, K):
            self.Vs[d] *= np.zeros((D, M))

        # set noise variances
        self.accum_log_sigmasq = np.log(1e-3) * np.ones(
            D,
        )
        mask1 = np.vstack(
            (
                np.ones(
                    D,
                ),
                np.zeros((K - 1, D)),
            )
        )
        mask2 = np.vstack((np.zeros(D), np.ones((K - 1, D))))
        self.bound_variance = 1e-4
        self._log_sigmasq = (
            self.accum_log_sigmasq * mask1 + np.log(self.bound_variance) * mask2
        )

        # Set the remaining parameters to fixed values
        self.bs = np.zeros((K, D))
        self.mu_init = np.zeros((K, D))
        self._log_sigmasq_init = np.log(0.001 * np.ones((K, D)))

    @property
    def params(self):
        """Pack learnable parameters."""
        params = self._V0, self.accum_log_sigmasq, self._a_diag
        return params

    @params.setter
    def params(self, value):
        """Unpack parameters and rebuild derived state-dependent matrices."""
        self._V0, self.accum_log_sigmasq, self._a_diag = value

        K, D, M = self.K, self.D, self.M

        # Update V
        self.Vs = np.vstack((self._V0 * np.ones((1, D, M)), np.zeros((K - 1, D, M))))

        # update sigma
        mask1 = np.vstack(
            (
                np.ones(
                    D,
                ),
                np.zeros((K - 1, D)),
            )
        )
        mask2 = np.vstack((np.zeros(D), np.ones((K - 1, D))))
        self._log_sigmasq = (
            self.accum_log_sigmasq * mask1 + np.log(self.bound_variance) * mask2
        )

        # update A
        mask1 = np.vstack(
            (np.eye(D)[None, :, :], np.zeros((K - 1, D, D)))
        )  # for accum state
        mask2 = np.vstack((np.zeros((1, D, D)), np.tile(np.eye(D), (K - 1, 1, 1))))
        self._As = self._a_diag * mask1 + mask2

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        """Initialize observation parameters.

        Args:
            datas: Observation sequences.
            inputs: Optional input sequences.
            masks: Optional observation masks.
            tags: Optional metadata tags.

        Returns:
            None. This method intentionally performs no initialization.
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
        """Use generic observation M-step from ``ssm``.

        Args:
            expectations: Posterior expectations from E-step.
            datas: Observation sequences.
            inputs: Input sequences.
            masks: Observation masks.
            tags: Metadata tags.
            continuous_expectations: Optional continuous expectations (unused).
            **kwargs: Additional optimizer arguments.

        Returns:
            None.
        """
        Observations.m_step(self, expectations, datas, inputs, masks, tags, **kwargs)


class Accumulation(HMM):
    """Discrete-state accumulation model composed from custom transitions/observations."""

    def __init__(
        self,
        K,
        D,
        *,
        M,
        transitions="race",
        transition_kwargs=None,
        observations="acc",
        observation_kwargs=None,
        **kwargs,
    ):
        """Construct an accumulation HMM.

        Args:
            K: Number of discrete states.
            D: Continuous latent dimensionality used by observations.
            M: Input dimensionality.
            transitions: Transition class label in ``transition_classes``.
            transition_kwargs: Optional kwargs forwarded to transition constructor.
            observations: Observation class label in ``observation_classes``.
            observation_kwargs: Optional kwargs forwarded to observation constructor.
            **kwargs: Unused extra kwargs for compatibility.
        """

        init_state_distn = AccumulationInitialStateDistribution(K, D, M=M)
        init_state_distn.log_pi0 = np.log(
            np.concatenate(([0.999], (0.001 / (K - 1)) * np.ones(K - 1)))
        )

        transition_classes = dict(
            racesoft=AccumulationRaceSoftTransitions,
            race=AccumulationRaceTransitions,
            ddmsoft=DDMSoftTransitions,
            ddm=DDMTransitions,
            ddmcollapsing=DDMCollapsingTransitions,
            ddmnlncollapsing=DDMNonlinearCollapsingTransitions,
        )
        transition_kwargs = transition_kwargs or {}
        transitions = transition_classes[transitions](K, D, M=M, **transition_kwargs)

        observation_classes = dict(
            acc=AccumulationObservations, accglm=AccumulationGLMObservations
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


class AccumulationGaussianEmissions(GaussianEmissions):
    """Gaussian emission model used by latent accumulation SLDS variants."""

    def __init__(self, N, K, D, M=0, single_subspace=True):
        """Create Gaussian emissions and freeze direct input effects.

        Args:
            N: Number of observed channels.
            K: Number of discrete states.
            D: Continuous latent dimensionality.
            M: Input dimensionality.
            single_subspace: Whether to share emission subspace across states.
        """
        super().__init__(N, K, D, M=M, single_subspace=single_subspace)
        # Make sure the input matrix Fs is set to zero and never updated
        self.Fs *= 0

    # Construct an emissions model
    @property
    def params(self):
        """Return trainable Gaussian emission parameters."""
        return self._Cs, self.ds, self.inv_etas

    @params.setter
    def params(self, value):
        """Set Gaussian emission parameters."""
        self._Cs, self.ds, self.inv_etas = value

    def initialize(
        self,
        base_model,
        datas,
        inputs=None,
        masks=None,
        tags=None,
        num_em_iters=50,
        num_tr_iters=50,
    ):
        """Initialize emission parameters from FA and model-consistent transforms.

        For 1D DDM models, uses percentile-conditioned trial ends to set an
        interpretable loading direction. Otherwise fits FA, then learns an affine
        transform of latent coordinates to better match AR-HMM priors.

        Args:
            base_model: ``Accumulation`` model used to score transformed latents.
            datas: Observation sequences.
            inputs: Input sequences.
            masks: Observation masks.
            tags: Optional metadata tags.
            num_em_iters: Number of FA EM iterations.
            num_tr_iters: Number of latent-transform optimization iterations.

        Returns:
            None. Updates emission parameters in place.
        """

        print("Initializing...")
        print(f"First with FA using {num_em_iters} steps of EM.")
        fa, xhats, Cov_xhats, lls = factor_analysis_with_imputation(
            self.D, datas, masks=masks, num_iters=num_em_iters
        )

        if (
            self.D == 1
            and base_model.transitions.__class__.__name__ == "DDMTransitions"
        ):
            d_init = np.mean([y[0:3] for y in datas], axis=(0, 1))
            u_sum = np.array([np.sum(u) for u in inputs])
            y_end = np.array([y[-3:] for y in datas])
            u_l, u_u = np.percentile(
                u_sum, [20, 80]
            )  # use 20th and 80th percentile input
            y_U = y_end[np.where(u_sum >= u_u)]
            y_L = y_end[np.where(u_sum <= u_l)]
            C_init = (1.0 / 2.0) * np.mean(
                (np.mean(y_U, axis=0) - np.mean(y_L, axis=0)), axis=0
            )

            self.Cs = C_init.reshape([1, self.N, self.D])
            self.ds = d_init.reshape([1, self.N])
            self.inv_etas = np.log(fa.sigmasq).reshape([1, self.N])

        else:
            # define objective
            Td = sum([x.shape[0] for x in xhats])

            def _objective(params, itr):
                new_datas = [np.dot(x, params[0].T) + params[1] for x in xhats]
                obj = base_model.log_likelihood(new_datas, inputs=inputs)
                return -obj / Td

            # initialize R and r
            R = 0.1 * np.random.randn(self.D, self.D)
            r = 0.01 * np.random.randn(self.D)
            params = [R, r]

            print(
                f"Next by transforming latents to match AR-HMM prior using {num_tr_iters} steps of max log likelihood."
            )
            state = None
            lls = [-_objective(params, 0) * Td]
            pbar = trange(num_tr_iters)
            pbar.set_description(f"Epoch {0} Itr {0} LP: {lls[-1]:.1f}")

            for itr in pbar:
                params, val, g, state = sgd_step(
                    value_and_grad(_objective), params, itr, state
                )
                lls.append(-val * Td)
                pbar.set_description(f"LP: {lls[-1]:.1f}")
                pbar.update(1)

            R = params[0]
            r = params[1]

            # scale x's to be max at 1.1
            for d in range(self.D):
                x_transformed = [(np.dot(x, R.T) + r)[:, d] for x in xhats]
                max_x = np.max(x_transformed)
                R[d, :] *= 1.1 / max_x
                r[d] *= 1.1 / max_x

            self.Cs = (fa.W @ np.linalg.inv(R)).reshape([1, self.N, self.D])
            self.ds = fa.mean - fa.W @ np.linalg.inv(R) @ r
            self.inv_etas = np.log(fa.sigmasq).reshape([1, self.N])


class AccumulationPoissonEmissions(PoissonEmissions):
    """Poisson emission model with smooth inversion helper for initialization."""

    def __init__(
        self, N, K, D, M=0, single_subspace=True, link="softplus", bin_size=1.0
    ):
        """Create Poisson emissions and freeze direct input effects.

        Args:
            N: Number of observed channels.
            K: Number of discrete states.
            D: Continuous latent dimensionality.
            M: Input dimensionality.
            single_subspace: Whether all states share one emission subspace.
            link: Inverse link used by :class:`ssm.emissions.PoissonEmissions`.
            bin_size: Observation bin width for count scaling/smoothing behavior.
        """
        super().__init__(
            N, K, D, M=M, single_subspace=single_subspace, link=link, bin_size=bin_size
        )
        # Make sure the input matrix Fs is set to zero and never updated
        self.Fs *= 0

    # Construct an emissions model
    @property
    def params(self):
        """Return trainable Poisson emission parameters."""
        return self._Cs, self.ds

    @params.setter
    def params(self, value):
        """Set Poisson emission parameters."""
        self._Cs, self.ds = value

    def invert(self, data, input=None, mask=None, tag=None, clip=np.array([0.0, 1.0])):
        """Build a smoothed latent initialization from count observations.

        Args:
            data: Count observations with shape ``(T, N)``.
            input: Optional input matrix ``(T, M)``.
            mask: Optional validity mask.
            tag: Optional trial metadata.
            clip: Unused legacy argument kept for compatibility.

        Returns:
            Approximate latent trajectory ``xhat`` with shape ``(T, D)``.
        """
        #         yhat = self.link(np.clip(data, .1, np.inf))
        if self.bin_size < 1:
            yhat = smooth(data, 20)
        else:
            yhat = smooth(data, 5)

        xhat = self.link(np.clip(yhat, 0.01, np.inf))
        xhat = self._invert(xhat, input=input, mask=mask, tag=tag)
        xhat = smooth(xhat, 10)

        if self.bin_size < 1:
            xhat = np.clip(xhat, -0.95, 0.95)

        # in all models, x starts in between boundaries at [-1,1]
        if np.abs(xhat[0]).any() > 1.0:
            xhat[0] = 0.05 * npr.randn(1, self.D)
        return xhat

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
        """Initialize Poisson emissions from model samples and optimize likelihood.

        Args:
            base_model: ``Accumulation`` model used to generate latent samples.
            datas: Observation sequences.
            inputs: Input sequences.
            masks: Observation masks.
            tags: Optional metadata tags.
            emission_optimizer: Name of optimizer to use (``"bfgs"`` or ``"lbfgs"``).
            num_optimizer_iters: Maximum optimizer iterations.

        Returns:
            None. Updates emission parameters in place.
        """
        print("Initializing Emissions parameters...")

        if (
            self.D == 1
            and base_model.transitions.__class__.__name__ == "DDMTransitions"
        ):
            # if self.D == 0:
            d_init = np.mean([y[0:3] for y in datas], axis=(0, 1))
            u_sum = np.array([np.sum(u) for u in inputs])
            y_end = np.array([y[-3:] for y in datas])
            u_l, u_u = np.percentile(
                u_sum, [20, 80]
            )  # use 20th and 80th percentile input
            y_U = y_end[np.where(u_sum >= u_u)]
            y_L = y_end[np.where(u_sum <= u_l)]
            C_init = (1.0 / 2.0) * np.mean(
                (np.mean(y_U, axis=0) - np.mean(y_L, axis=0)), axis=0
            )
            self.Cs = C_init.reshape([1, self.N, self.D]) / self.bin_size
            self.ds = d_init.reshape([1, self.N]) / self.bin_size

        else:
            datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]

            Td = sum([data.shape[0] for data in datas])
            xs = [
                base_model.sample(T=data.shape[0], input=input)[1]
                for data, input in zip(datas, inputs)
            ]

            def _objective(params, itr):
                self.params = params
                # self.Cs = params
                obj = 0
                obj += self.log_prior()
                for data, input, mask, tag, x in zip(datas, inputs, masks, tags, xs):
                    obj += np.sum(self.log_likelihoods(data, input, mask, tag, x))
                return -obj / Td

            # Optimize emissions log-likelihood
            optimizer = dict(bfgs=bfgs, lbfgs=lbfgs)[emission_optimizer]
            self.params = optimizer(
                _objective,
                self.params,
                num_iters=num_optimizer_iters,
                full_output=False,
            )


class RampStepPoissonEmissions(PoissonEmissions):
    """Poisson emissions with shared loading matrix across all discrete states."""

    def __init__(
        self, N, K, D, M=0, single_subspace=False, link="softplus", bin_size=1.0
    ):
        """Create state-tied Poisson emissions for ramp-step models."""
        super().__init__(
            N, K, D, M=M, single_subspace=single_subspace, link=link, bin_size=bin_size
        )
        # Make sure the input matrix Fs is set to zero and never updated
        self.Fs *= 0
        self.C = self._Cs[0]
        self._Cs = self.C * np.ones((K, N, D))

    # Construct an emissions model
    @property
    def params(self):
        """Return trainable shared loading matrix and offsets."""
        return self.C, self.ds

    @params.setter
    def params(self, value):
        """Set shared loading matrix and broadcast it to all states."""
        self.C, self.ds = value
        self._Cs = self.C * np.ones((self.K, self.N, self.D))

    def _invert(self, data, input=None, mask=None, tag=None):
        """Approximate inverse of linear emissions using a pseudoinverse.

        Args:
            data: Emission-space array with shape ``(T, N)``.
            input: Optional input sequence with shape ``(T, M)``.
            mask: Optional observation mask.
            tag: Optional metadata tag (unused).

        Returns:
            np.ndarray: Estimated latent sequence with shape ``(T, D)``.

        Notes:
            Assumes an approximately linear mapping ``y = Cx + d + noise``.
        """
        # assert self.single_subspace, "Can only invert with a single emission model"

        C, F, d = self.Cs[0], self.Fs[0], self.ds[0]
        C_pseudoinv = np.linalg.solve(C.T.dot(C), C.T).T

        # Account for the bias
        bias = input.dot(F.T) + d

        if not np.all(mask):
            data = interpolate_data(data, mask)
            # We would like to find the PCA coordinates in the face of missing data
            # To do so, alternate between running PCA and imputing the missing entries
            for itr in range(25):
                mu = (data - bias).dot(C_pseudoinv)
                data[:, ~mask[0]] = (mu.dot(C.T) + bias)[:, ~mask[0]]

        # Project data to get the mean
        return (data - bias).dot(C_pseudoinv)

    def invert(self, data, input=None, mask=None, tag=None, clip=np.array([0.0, 1.0])):
        """Generate smoothed clipped latent initialization from count data.

        Args:
            data: Count observations with shape ``(T, N)``.
            input: Optional input sequence with shape ``(T, M)``.
            mask: Optional observation mask.
            tag: Optional metadata tag.
            clip: Legacy argument retained for compatibility.

        Returns:
            np.ndarray: Smoothed latent initialization with shape ``(T, D)``.
        """
        yhat = smooth(data, 20)
        xhat = self.link(np.clip(yhat, 0.01, np.inf))
        xhat = self._invert(xhat, input=input, mask=mask, tag=tag)
        num_pad = 10
        xhat = smooth(np.concatenate((np.zeros((num_pad, self.D)), xhat)), 10)[
            num_pad:, :
        ]
        xhat = np.clip(xhat, -1.1, 1.1)
        if np.abs(xhat[0]).any() > 1.0:
            xhat[0] = 0.1 * npr.randn(1, self.D)
        return xhat

    def initialize(
        self,
        base_model,
        datas,
        inputs=None,
        masks=None,
        tags=None,
        emission_optimizer="bfgs",
        num_optimizer_iters=50,
    ):
        """Fit ramp-step emission parameters by maximizing expected log-likelihood.

        Args:
            base_model: ``Accumulation`` model used to generate latent samples.
            datas: Observation sequences.
            inputs: Input sequences.
            masks: Observation masks.
            tags: Optional metadata tags.
            emission_optimizer: Name of optimizer to use (``"bfgs"`` or ``"lbfgs"``).
            num_optimizer_iters: Maximum optimizer iterations.

        Returns:
            None. Updates emission parameters in place.
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


class LatentAccumulation(SLDS):
    """Switching LDS with accumulation-specific transitions, dynamics, and emissions."""

    def __init__(
        self,
        N,
        K,
        D,
        *,
        M,
        transitions="race",
        transition_kwargs=None,
        dynamics="acc",
        dynamics_kwargs=None,
        emissions="gaussian",
        emission_kwargs=None,
        single_subspace=True,
        **kwargs,
    ):
        """Construct a latent accumulation SLDS from named component families.

        Args:
            N: Number of observed channels.
            K: Number of discrete states.
            D: Continuous latent dimensionality.
            M: Input dimensionality.
            transitions: Transition family label.
            transition_kwargs: Optional transition constructor kwargs.
            dynamics: Continuous dynamics family label.
            dynamics_kwargs: Optional dynamics constructor kwargs.
            emissions: Emission family label.
            emission_kwargs: Optional emission constructor kwargs.
            single_subspace: Emission subspace sharing flag.
            **kwargs: Extra kwargs reserved for compatibility.
        """

        init_state_distn = AccumulationInitialStateDistribution(K, D, M=M)
        init_state_distn.log_pi0 = np.log(
            np.concatenate(([0.9999], (0.0001 / (K - 1)) * np.ones(K - 1)))
        )
        # init_state_distn.log_pi0 = np.log(np.concatenate(([1.0],(0.0/(K-1))*np.ones(K-1))))

        transition_classes = dict(
            racesoft=AccumulationRaceSoftTransitions,
            race=AccumulationRaceTransitions,
            ddmsoft=DDMSoftTransitions,
            ddm=DDMTransitions,
            ddmcollapsing=DDMCollapsingTransitions,
            ddmnlncollapsing=DDMNonlinearCollapsingTransitions,
            embedded_race=EmbeddedAccumulationRaceTransitions,
        )
        self.transitions_label = transitions
        self.transition_kwargs = transition_kwargs
        transition_kwargs = transition_kwargs or {}
        transitions = transition_classes[transitions](K, D, M=M, **transition_kwargs)

        self.dynamics_kwargs = dynamics_kwargs
        dynamics_kwargs = dynamics_kwargs or {}
        dynamics_classes = dict(
            acc=AccumulationObservations,
            embedded_acc=EmbeddedAccumulationObservations,
        )
        dynamics_kwargs = dynamics_kwargs or {}
        self.dynamics_kwargs = dynamics_kwargs
        dynamics = dynamics_classes[dynamics](K, D, M=M, **dynamics_kwargs)

        self.emissions_label = emissions
        emission_classes = dict(
            gaussian=AccumulationGaussianEmissions,
            poisson=AccumulationPoissonEmissions,
            rampstep=RampStepPoissonEmissions,
        )  # ,
        # calcium=AccumulationCalciumEmissions)
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
    ):
        """Initialize emissions and optionally warm-start latent dynamics.

        For Gaussian emissions, this method performs a short AR-HMM EM fit on
        inverted latents and copies those fitted initial-state, transition, and
        dynamics parameters into the SLDS.

        Args:
            datas: Observation sequences.
            inputs: Input sequences.
            masks: Observation masks.
            tags: Optional metadata tags.
            num_optimizer_iters: Reserved optimizer-iteration argument.
            num_em_iters: Number of warm-start AR-HMM EM iterations.
            betas: Optional external drift/input coefficients (currently unused).
            accum_log_sigmasq: Optional external process variances (currently unused).

        Returns:
            None. Updates model components in place.
        """

        # First initialize the observation model
        self.base_model = Accumulation(
            self.K,
            self.D,
            M=self.M,
            transitions=self.transitions_label,
            transition_kwargs=self.transition_kwargs,
            observation_kwargs=self.dynamics_kwargs,
        )
        self.base_model.observations.Vs = self.dynamics.Vs
        self.base_model.observations.As = self.dynamics.As
        self.base_model.observations.Sigmas = self.dynamics.Sigmas
        self.emissions.initialize(self.base_model, datas, inputs, masks, tags)

        if self.emissions_label == "gaussian":
            # Get the initialized variational mean for the data
            xs = [
                self.emissions.invert(data, input, mask, tag)
                for data, input, mask, tag in zip(datas, inputs, masks, tags)
            ]
            xmasks = [np.ones_like(x, dtype=bool) for x in xs]

            # Now run a few iterations of EM on a ARHMM with the variational mean
            print(f"Initializing with an ARHMM using {num_em_iters} steps of EM.")
            arhmm = hmm.HMM(
                self.K,
                self.D,
                M=self.M,
                init_state_distn=copy.deepcopy(self.init_state_distn),
                transitions=copy.deepcopy(self.transitions),
                observations=copy.deepcopy(self.dynamics),
            )

            arhmm.fit(
                xs,
                inputs=inputs,
                masks=xmasks,
                tags=tags,
                method="em",
                num_em_iters=num_em_iters,
            )

            self.init_state_distn = copy.deepcopy(arhmm.init_state_distn)
            self.transitions = copy.deepcopy(arhmm.transitions)
            self.dynamics = copy.deepcopy(arhmm.observations)

    @ensure_args_are_lists
    def monte_carlo_loglikelihood(
        self, datas, inputs=None, masks=None, tags=None, num_samples=100
    ):
        """Estimate ``log p(y | theta)`` using prior-sampled latent trajectories.

        Args:
            datas: List of observed trials.
            inputs: Optional list of trial inputs.
            masks: Optional list of masks.
            tags: Optional list of metadata tags.
            num_samples: Number of latent samples per trial.

        Returns:
            Tuple ``(ll, trial_lls, trial_sample_lls)`` with total log-likelihood,
            per-trial log-likelihood estimates, and raw per-sample trial scores.
        """
        trial_lls = []
        trial_sample_lls = []

        print("Estimating log-likelihood...")
        for data, input, mask, tag in zip(tqdm(datas), inputs, masks, tags):
            sample_lls = []
            samples = [
                self.sample(data.shape[0], input=input, tag=tag)
                for sample in range(num_samples)
            ]

            for sample in samples:
                z, x = sample[:2]
                sample_ll = np.sum(
                    self.emissions.log_likelihoods(data, input, mask, tag, x)
                )
                sample_lls.append(sample_ll)

                assert np.isfinite(sample_ll)

            trial_ll = logsumexp(sample_lls) - np.log(num_samples)
            trial_lls.append(trial_ll)
            trial_sample_lls.append(sample_lls)

        ll = np.sum(trial_lls)

        return ll, trial_lls, trial_sample_lls


class AccumulationInitialStateDistribution(InitialStateDistribution):
    """Initial discrete-state distribution with optional fixed prior mass."""

    def __init__(self, K, D, M=0):
        """Create an initial-state distribution over ``K`` states."""
        self.K, self.D, self.M = K, D, M
        self.log_pi0 = -np.log(K) * np.ones(K)

    @property
    def params(self):
        """Return model parameters as a tuple for ``ssm`` compatibility."""
        return (self.log_pi0,)

    @params.setter
    def params(self, value):
        """Set log-probabilities of the initial state distribution."""
        self.log_pi0 = value[0]

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        """Initialize initial-state parameters.

        Args:
            datas: Observation sequences.
            inputs: Optional input sequences.
            masks: Optional observation masks.
            tags: Optional metadata tags.

        Returns:
            None. This method intentionally performs no initialization.
        """

    def permute(self, perm):
        """Permute state labels consistently with model state relabeling.

        Args:
            perm: Permutation index array for states ``0..K-1``.
        """
        self.log_pi0 = self.log_pi0[perm]

    @property
    def initial_state_distn(self):
        """Return normalized initial state probabilities."""
        return np.exp(self.log_pi0 - logsumexp(self.log_pi0))

    @property
    def log_initial_state_distn(self):
        """Return normalized log-probabilities for initial states."""
        return self.log_pi0 - logsumexp(self.log_pi0)

    def log_prior(self):
        """Return log-prior over parameters (constant in this implementation)."""
        return 0

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        """Update initial-state distribution parameters.

        Args:
            expectations: Posterior expectations from E-step.
            datas: Observation sequences.
            inputs: Input sequences.
            masks: Observation masks.
            tags: Metadata tags.
            **kwargs: Additional optimizer arguments.

        Returns:
            None. This method intentionally keeps the prior fixed.
        """
        # do not update the parameters
        self.log_pi0 = self.log_pi0
