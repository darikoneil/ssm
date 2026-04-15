from warnings import warn

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import gammaln
from autograd import hessian

from ssm.util import ensure_args_are_lists, \
    logistic, logit, softplus, inv_softplus
from ssm.preprocessing import interpolate_data, pca_with_imputation
from ssm.optimizers import adam, bfgs, rmsprop, sgd, lbfgs
from ssm.stats import independent_studentst_logpdf, bernoulli_logpdf
from ssm.regression import fit_linear_regression

from ssm.emissions import Emissions, _LinearEmissions, _OrthogonalLinearEmissions, _IdentityEmissions

class _GammaEmissionsMixin(object):
    def __init__(self, N, K, D, M=0, single_subspace=True, link="softplus", **kwargs):
        super(_GammaEmissionsMixin, self).__init__(N, K, D, M, single_subspace=single_subspace, **kwargs)

        self.link_name = link
        mean_functions = dict(
            log=self._log_mean,
            softplus=self._softplus_mean
        )
        self.mean = mean_functions[link]
        link_functions = dict(
            log=self._log_link,
            softplus=self._softplus_link
        )
        self.link = link_functions[link]

        # Per-emission gamma shape parameter (must be positive)
        Keff = 1 if single_subspace else K
        self.inv_alphas = np.log(10.0) * np.ones((Keff, N))

    @property
    def alphas(self):
        return np.exp(self.inv_alphas)

    @property
    def params(self):
        return super(_GammaEmissionsMixin, self).params + (self.inv_alphas,)

    @params.setter
    def params(self, value):
        super(_GammaEmissionsMixin, self.__class__).params.fset(self, value[:-1])
        self.inv_alphas = value[-1]

    def permute(self, perm):
        super(_GammaEmissionsMixin, self).permute(perm)
        if not self.single_subspace:
            self.inv_alphas = self.inv_alphas[perm]
            
    def neg_hessian_log_emissions_prob(self, data, input, mask, tag, x, Ez):
    """
    Analytical negative Hessian of log p(y | x) for Gamma emissions.

    We parameterize with mean mu and shape alpha:
        y ~ Gamma(alpha, beta),  beta = alpha / mu

    log p(y | mu, alpha) = const - alpha * log(mu) - alpha * y / mu
    where mu = f(eta), eta = Cx + Fu + d.
    """
    if self.single_subspace is False:
        raise Exception("Multiple subspaces are not supported for this Emissions class.")

    mask = np.ones_like(data, dtype=bool) if mask is None else mask
    y = np.clip(data, 1e-16, np.inf)
    eta = self.forward(x, input, tag)[:, 0, :]
    mu = np.clip(self.mean(eta), 1e-16, np.inf)
    alpha = np.clip(self.alphas[0], 1e-16, np.inf)

    if self.link_name == "log":
        d2l_deta2 = -alpha * y / mu
    elif self.link_name == "softplus":
        dmu = logistic(eta)
        d2mu = dmu * (1.0 - dmu)
        dl_dmu = alpha * (y - mu) / (mu ** 2)
        d2l_dmu2 = alpha * (mu - 2.0 * y) / (mu ** 3)
        d2l_deta2 = d2l_dmu2 * (dmu ** 2) + dl_dmu * d2mu
    else:
        raise Exception("No Hessian calculation for link: {}".format(self.link_name))

    d2l_deta2 = d2l_deta2 * mask
    hess = np.einsum('tn, ni, nj ->tij', d2l_deta2, self.Cs[0], self.Cs[0])
    return -1.0 * hess
    def _log_mean(self, x):
        return np.exp(x)

    def _softplus_mean(self, x):
        return softplus(x)

    def _log_link(self, rate):
        return np.log(rate)

    def _softplus_link(self, rate):
        return inv_softplus(rate)

    def log_likelihoods(self, data, input, mask, tag, x):
        eps = 1e-16
        mus = self.mean(self.forward(x, input, tag))
        mus = np.clip(mus, eps, np.inf)
        ys = np.clip(data[:, None, :], eps, np.inf)

        alphas = self.alphas[None, :, :]
        betas = alphas / mus

        lls = alphas * np.log(betas) - gammaln(alphas) + (alphas - 1.0) * np.log(ys) - betas * ys
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        return np.sum(lls * mask[:, None, :], axis=2)

    def invert(self, data, input=None, mask=None, tag=None):
        yhat = self.link(np.clip(data, 1e-8, np.inf))
        return self._invert(yhat, input=input, mask=mask, tag=tag)

    def sample(self, z, x, input=None, tag=None):
        T = z.shape[0]
        z = np.zeros_like(z, dtype=int) if self.single_subspace else z
        mus = self.mean(self.forward(x, input, tag))
        alphas = self.alphas
        betas = alphas / np.clip(mus[np.arange(T), z, :], 1e-16, np.inf)
        return npr.gamma(shape=alphas[z], scale=1.0 / betas)

    def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None):
        mus = self.mean(self.forward(variational_mean, input, tag))
        return mus[:, 0, :] if self.single_subspace else np.sum(mus * expected_states[:, :, None], axis=1)


class GammaEmissions(_GammaEmissionsMixin, _LinearEmissions):
    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        yhats = [self.link(np.clip(d, 1e-8, np.inf)) for d in datas]
        self._initialize_with_pca(yhats, inputs=inputs, masks=masks, tags=tags)


class GammaOrthogonalEmissions(_GammaEmissionsMixin, _OrthogonalLinearEmissions):
    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        yhats = [self.link(np.clip(d, 1e-8, np.inf)) for d in datas]
        self._initialize_with_pca(yhats, inputs=inputs, masks=masks, tags=tags)
