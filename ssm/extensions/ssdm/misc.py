"""Miscellaneous utilities for neural population data simulation and analysis.

This module provides helper functions for generating synthetic click stimuli,
running factor analysis, smoothing time-series data, simulating accumulator
models, computing and plotting peri-stimulus time histograms (PSTHs), and
evaluating goodness-of-fit metrics such as R² and mean absolute deviation.
"""

import itertools

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
from scipy.stats import multivariate_normal


def generate_clicks(T=1.0, dt=0.01, rate_r=20, rate_l=20):
    """Generate binned right and left Poisson click trains.

    Simulates right and left auditory clicks as two independent Poisson
    processes over a trial of duration ``T`` seconds.  Click times are drawn
    uniformly within the trial window and then binned into discrete time steps
    of width ``dt``.

    Args:
        T (float): Trial duration in seconds. Defaults to 1.0.
        dt (float): Bin width in seconds. Defaults to 0.01.
        rate_r (float): Mean click rate for the right channel in Hz.
            Defaults to 20.
        rate_l (float): Mean click rate for the left channel in Hz.
            Defaults to 20.

    Returns:
        tuple: A 2-tuple ``(binned_r, binned_l)`` where:

        * **binned_r** (*np.ndarray*): Right-channel click counts per bin,
          shape ``(T/dt,)``.
        * **binned_l** (*np.ndarray*): Left-channel click counts per bin,
          shape ``(T/dt,)``.
    """
    # number of clicks
    num_r = npr.poisson(rate_r * T)
    num_l = npr.poisson(rate_l * T)

    # click times
    click_time_r = np.sort(npr.uniform(low=0.0, high=T, size=[num_r, 1]))
    click_time_l = np.sort(npr.uniform(low=0.0, high=T, size=[num_l, 1]))

    # binned outputs are arrays with dimensions Tx1
    binned_r = np.histogram(click_time_r, np.arange(0.0, T + dt, dt))[0]
    binned_l = np.histogram(click_time_l, np.arange(0.0, T + dt, dt))[0]

    return binned_r, binned_l


def generate_clicks_D(rates, T=1.0, dt=0.01):
    """Generate binned click trains for D independent Poisson processes.

    Generalises :func:`generate_clicks` to an arbitrary number of channels.
    Each channel is an independent Poisson process whose clicks are drawn
    uniformly within the trial window and then binned.

    Args:
        rates (list[float]): Mean click rates in Hz for each of the ``D``
            channels.
        T (float): Trial duration in seconds. Defaults to 1.0.
        dt (float): Bin width in seconds. Defaults to 0.01.

    Returns:
        list[np.ndarray]: A list of ``D`` arrays, each of shape ``(T/dt,)``,
        containing the binned click counts for the corresponding channel.
    """
    # number of clicks
    num_clicks = [npr.poisson(rate * T) for rate in rates]

    # click times
    click_times = [
        np.sort(npr.uniform(low=0.0, high=T, size=[num_click, 1]))
        for num_click in num_clicks
    ]

    # binned outputs are arrays with dimensions Tx1
    binned_clicks = [
        np.histogram(click_time, np.arange(0.0, T + dt, dt))[0]
        for click_time in click_times
    ]

    return binned_clicks


def factor_analysis(D, ys, num_iters=15):
    """Fit a probabilistic factor-analysis model using the EM algorithm.

    Estimates the loading matrix ``C`` and diagonal noise covariance ``Psi``
    of the generative model:

    .. math::
        y = C x + \\epsilon, \\quad x \\sim \\mathcal{N}(0, I_D), \\quad
        \\epsilon \\sim \\mathcal{N}(0, \\Psi)

    via expectation-maximisation.  All trials in ``ys`` are concatenated and
    mean-subtracted before fitting.

    Args:
        D (int): Number of latent dimensions.
        ys (list[np.ndarray]): List of observation arrays.  Each element has
            shape ``(T_i, N)`` where ``T_i`` is the number of time points in
            trial ``i`` and ``N`` is the observation dimensionality.
        num_iters (int): Number of EM iterations to run. Defaults to 15.

    Returns:
        tuple: A 4-tuple ``(Cfa, my_xhats, lls, psi)`` where:

        * **Cfa** (*np.ndarray*): Estimated loading matrix, shape ``(N, D)``.
        * **my_xhats** (*list[np.ndarray]*): Posterior mean latent trajectories,
          one array of shape ``(T_i, D)`` per trial.
        * **lls** (*list[float]*): Marginal log-likelihood at each EM iteration.
        * **psi** (*np.ndarray*): Estimated diagonal noise covariance matrix,
          shape ``(N, N)``.
    """
    # concatenate ys
    all_y = np.array(list(itertools.chain(*ys)))

    # observation dimensions
    Nobs, N = np.shape(all_y)

    # compute mean across column
    mu_y = np.mean(all_y, axis=0, keepdims=True)

    # subtract mean
    ys_zero = all_y - mu_y

    # initialize C, Psi
    Cfa = np.random.randn(N, D)
    psi = np.eye(N)

    # run EM
    pbar = trange(num_iters)
    lls = []
    for i in pbar:
        # E-step
        lamb = np.linalg.inv(Cfa.T @ np.linalg.inv(psi) @ Cfa + np.eye(D))
        mu_x = (lamb @ Cfa.T @ np.linalg.inv(psi) @ (ys_zero.T)).T
        mu_xxT = [lamb + np.outer(mu_x[i, :], mu_x[i, :]) for i in range(Nobs)]

        # M-step
        Cfa = (np.linalg.inv((mu_x.T @ mu_x) + Nobs * lamb) @ mu_x.T @ ys_zero).T
        np.fill_diagonal(
            psi,
            np.diag((1.0 / Nobs) * (ys_zero.T @ ys_zero - ys_zero.T @ mu_x @ Cfa.T)),
        )

        # add small elements to diagonal of psi for stability (TODO: add condition)
        np.fill_diagonal(psi, np.diag(psi + 1e-7 * np.eye(N)))

        # compute log likelihood
        log_py = np.sum(
            multivariate_normal.logpdf(all_y, mean=mu_y[0, :], cov=(Cfa @ Cfa.T + psi))
        )
        lls += [log_py]

        pbar.set_description(f"Itr {i} LP: {lls[-1]:.1f}")
        pbar.update(1)

    # get xhats
    my_xhats = [(lamb @ Cfa.T @ np.linalg.inv(psi) @ (y - mu_y[0, :]).T).T for y in ys]

    return Cfa, my_xhats, lls, psi


def smooth(xs, window_size=5):
    """Apply a symmetric sliding-window mean smoother to a 2-D array.

    For each time point ``t``, the smoothed value is the mean of all samples
    within a window of half-width ``win = (window_size - 1) / 2`` centred on
    ``t``.  The window is clipped at the array boundaries, so edge time points
    are averaged over fewer samples.

    Args:
        xs (np.ndarray): Input array of shape ``(T, N)`` where ``T`` is the
            number of time bins and ``N`` is the number of channels/neurons.
        window_size (int): Total window size in bins (number of bins on each
            side is ``(window_size - 1) / 2``). Defaults to 5.

    Returns:
        np.ndarray: Smoothed array of the same shape ``(T, N)`` as ``xs``.
    """
    T, N = np.shape(xs)
    x_smooth = np.zeros(np.shape(xs))

    # win is number of bins on each side
    win = int((window_size - 1) / 2)

    for t in range(T):
        smooth_window = np.arange(np.maximum(t - win, 0), np.minimum(t + win, T - 1))
        x_smooth[t, :] = np.mean(xs[smooth_window, :], axis=0)

    return x_smooth


def simulate_accumulator(model, inputs, num_repeats=1):
    """Simulate spike-count observations from a fitted accumulator model.

    For each trial input and each repeat, draws a sample trajectory and the
    corresponding spike counts from ``model``.

    Args:
        model: A fitted SLDS/HMM model exposing a ``sample(T, input)`` method
            that returns ``(z, x, y)``.
        inputs (list[np.ndarray]): List of ``N`` input arrays, each of shape
            ``(T_i, M)``.
        num_repeats (int): Number of times to repeat all trials. Defaults to
            1.

    Returns:
        list[np.ndarray]: Simulated spike-count arrays, one per
        ``(repeat, trial)`` combination.  Each array has shape ``(T_i, N_obs)``
        matching the corresponding input length.
    """
    N = len(inputs)
    ys = []
    for r in range(num_repeats):
        for n in range(N):
            T = inputs[n].shape[0]
            z, x, y = model.sample(T, inputs[n])
            ys.append(y)

    return ys


def compute_psths(ys, inputs, num_partitions=3, num_bins_smooth=3):
    """Compute input-conditioned peri-stimulus time histograms (PSTHs).

    Trials are sorted by the signed sum of their input (i.e., net rightward
    minus leftward evidence) and partitioned into coherence groups below zero,
    at zero, and above zero.  The mean spike count within each group is
    computed and optionally smoothed.

    Supports both 1-D inputs (single evidence channel) and 2-D inputs (two
    opposing channels), in which case the signed difference is used.

    Args:
        ys (list[np.ndarray]): Spike-count arrays, each of shape ``(T, N)``
            where ``T`` is the number of time bins and ``N`` is the number of
            neurons.  All trials are assumed to have the same length ``T``.
        inputs (list[np.ndarray]): Input arrays per trial.  Each element has
            shape ``(T, 1)`` or ``(T, 2)``.
        num_partitions (int): Number of coherence groups to create strictly
            above and strictly below zero. Defaults to 3.
        num_bins_smooth (int): Half-window size passed to :func:`smooth`.
            Set to 1 to disable smoothing. Defaults to 3.

    Returns:
        list[list]: A nested list of shape ``(N, 2*num_partitions + 1)``
        where the outer index is neuron and the inner index is coherence group
        (ordered from most negative to most positive input sum).  Each element
        is a 1-D array of mean firing rates in spikes/s (divided by
        ``bin_size = 0.01`` s).
    """
    # get time bins, number of neurons
    T, N = ys[0].shape

    if inputs[0].shape[1] == 1:
        u_sums = np.array([np.sum(u) for u in inputs])
    elif inputs[0].shape[1] == 2:
        u_sums = np.array([np.sum(u[:, 0] - u[:, 1]) for u in inputs])

    # get sorting index
    idx_sort = np.argsort(u_sums)
    u_sorted = u_sums[idx_sort]
    u_below_0 = np.where(u_sorted < 0)[0][-1]
    u_above_0 = np.where(u_sorted > 0)[0][0]
    u_0 = np.where(np.abs(u_sorted) < 1e-3)[0]

    idx_below_0 = np.array_split(idx_sort[:u_below_0], num_partitions)
    idx_0 = np.copy(idx_sort[u_0]) if u_0.shape[0] > 0 else np.array([])
    idx_above_0 = np.array_split(idx_sort[u_above_0:], num_partitions)

    # compute below-zero PSTHs (assumes same trial length)
    bin_size = 0.01
    all_idx = idx_below_0 + [idx_0] + idx_above_0
    y_psths = []
    for idx in all_idx:
        if idx.shape[0] > 0:
            y_idx = [ys[i] for i in idx]
            y_psths.append(mean_diff_dims(y_idx) / bin_size)
        else:
            y_psths.append(np.zeros((0, N)))

    # rearrange to be neuron by psth
    neuron_psths = [[psth[:, n] for psth in y_psths] for n in range(N)]
    if num_bins_smooth > 1:
        neuron_psths = [
            [smooth(row[:, None], num_bins_smooth) for row in psth]
            for psth in neuron_psths
        ]

    return neuron_psths


def mean_diff_dims(xs, min_counts=5):
    """Compute the element-wise mean of variable-length trial arrays.

    Aligns arrays of potentially different lengths along the time axis by
    accumulating sums and counts up to the length of each array, then divides
    to produce the mean.  Time points with fewer than ``min_counts``
    contributing trials are set to ``NaN``.

    Args:
        xs (list[np.ndarray]): List of arrays, each of shape ``(T_i, N)``,
            where ``T_i`` may differ across elements.
        min_counts (int): Minimum number of trials required at a given time
            bin for the mean to be considered valid.  Bins with fewer
            contributions are returned as ``NaN``. Defaults to 5.

    Returns:
        np.ndarray: Mean array of shape ``(T_max, N)`` where ``T_max`` is the
        longest trial length.  Entries with insufficient counts are ``NaN``.
    """
    N = xs[0].shape[1]
    T_max = max([x.shape[0] for x in xs])
    counts = np.zeros((T_max, N))
    means = np.zeros((T_max, N))
    for x in xs:
        Ti = x.shape[0]
        means[:Ti, :] += x
        counts[:Ti, :] += 1.0

    # min counts for returning mean
    out = np.divide(means, counts)
    out[counts < min_counts] = np.nan
    return out


def plot_psths(ys, inputs, num_row, num_col, fig=None, linestyle="-", ylim=None):
    """Plot per-neuron input-conditioned PSTHs in a subplot grid.

    Computes PSTHs via :func:`compute_psths` and renders one subplot per
    neuron, with each coherence group drawn in a distinct colour ranging from
    red (most negative) to blue (most positive).

    Args:
        ys (list[np.ndarray]): Spike-count arrays per trial, each of shape
            ``(T, N)``.
        inputs (list[np.ndarray]): Input arrays per trial.
        num_row (int): Number of rows in the subplot grid.
        num_col (int): Number of columns in the subplot grid.
        fig (matplotlib.figure.Figure, optional): Existing figure to draw
            into.  A new figure is created if ``None``. Defaults to ``None``.
        linestyle (str): Matplotlib line style string. Defaults to ``'-'``.
        ylim (tuple, optional): ``(ymin, ymax)`` axis limits applied to every
            subplot.  No limit is applied if ``None``. Defaults to ``None``.

    Returns:
        None
    """
    if fig is None:
        plt.figure()

    # get time bins, number of neurons
    T, N = ys[0].shape
    neuron_psths = compute_psths(ys, inputs)

    # plot
    colors = [
        [1.0, 0.0, 0.0],
        [1.0, 0.3, 0.3],
        [1.0, 0.6, 0.6],
        "k",
        [0.6, 0.6, 1.0],
        [0.3, 0.3, 1.0],
        [0.0, 0.0, 1.0],
    ]
    for n in range(N):
        plt.subplot(num_row, num_col, n + 1)
        for coh in range(len(neuron_psths[n])):
            plt.plot(
                neuron_psths[n][coh],
                color=colors[coh],
                linestyle=linestyle,
                alpha=0.9,
                linewidth=1,
            )
            if ylim is not None:
                plt.ylim(ylim)
    return


def plot_neuron_psth(
    neuron_psth, linestyle="-", ylim=None, flip_colors=False, colors=None
):
    """Plot coherence-conditioned PSTHs for a single neuron.

    Renders one line per coherence group using a red-to-blue colour scale
    (most negative → most positive input sum).

    Args:
        neuron_psth (list[np.ndarray]): List of PSTH arrays for a single
            neuron, one per coherence group.  Each element is a 1-D array of
            firing rates.
        linestyle (str): Matplotlib line style string. Defaults to ``'-'``.
        ylim (tuple, optional): ``(ymin, ymax)`` y-axis limits.  Not applied
            if ``None``. Defaults to ``None``.
        flip_colors (bool): If ``True``, reverses the colour order so that
            the most positive coherence is plotted in red (useful for neurons
            with inverted tuning). Defaults to ``False``.
        colors (list, optional): Custom list of Matplotlib colour
            specifications, one per coherence group.  If ``None``, a default
            red-to-blue palette is used. Defaults to ``None``.

    Returns:
        None
    """
    if colors is None:
        colors = [
            [1.0, 0.0, 0.0],
            [1.0, 0.3, 0.3],
            [1.0, 0.6, 0.6],
            "k",
            [0.6, 0.6, 1.0],
            [0.3, 0.3, 1.0],
            [0.0, 0.0, 1.0],
        ]
    if flip_colors:
        colors.reverse()
    for coh in range(len(neuron_psth)):
        plt.plot(neuron_psth[coh], color=colors[coh], linestyle=linestyle, alpha=0.9)
        if ylim is not None:
            plt.ylim(ylim)

    return


def compute_r2(true_psths, sim_psths):
    """Compute per-neuron coefficient of determination (R²) between PSTHs.

    R² is defined as:

    .. math::
        R^2_n = 1 - \\frac{\\sum_{j,t}(\\hat{y}_{njt} - y_{njt})^2}
                           {\\sum_{j,t}(\\bar{y}_{nt} - y_{njt})^2}

    where the sum is over coherence groups ``j`` and time bins ``t``, and
    :math:`\\bar{y}_{nt}` is the grand mean PSTH for neuron ``n`` (averaged
    over all coherence groups).  ``NaN`` values in either PSTH are excluded
    from both numerator and denominator.

    Args:
        true_psths (list[list[np.ndarray]]): Ground-truth PSTHs.  Outer list
            has length ``N`` (neurons); inner list has one array per coherence
            group.
        sim_psths (list[list[np.ndarray]]): Simulated PSTHs with the same
            structure as ``true_psths``.  Simulated arrays may be longer than
            the corresponding true arrays; they are truncated to match.

    Returns:
        np.ndarray: Per-neuron R² values, shape ``(N,)``.
    """
    assert len(true_psths) == len(sim_psths)
    N = len(true_psths)

    r2 = np.zeros(N)

    for i in range(N):
        true_psth = true_psths[i]
        sim_psth = sim_psths[i]
        true_psth_mean = [
            true_psth[coh]
            for coh in range(len(true_psth))
            if true_psth[coh].shape[0] > 0
        ]
        mean_PSTH = np.nanmean(np.vstack(true_psth_mean))

        r2_num = 0.0
        r2_den = 0.0

        NC = len(true_psth)
        for j in range(NC):
            T = true_psth[j].shape[0]
            if T > 0:
                r2_num += np.nansum((true_psth[j] - sim_psth[j][:T]) ** 2)
                r2_den += np.nansum((mean_PSTH - true_psth[j]) ** 2)

        r2[i] = 1 - r2_num / r2_den

    return r2


def compute_mad(true_psths, sim_psths):
    """Compute per-neuron summed mean absolute deviation (MAD) between PSTHs.

    For each neuron, the total absolute deviation is accumulated across all
    coherence groups and time bins:

    .. math::
        \\mathrm{MAD}_n = \\sum_{j,t} |\\hat{y}_{njt} - y_{njt}|

    ``NaN`` entries are ignored via :func:`numpy.nansum`.

    Args:
        true_psths (list[list[np.ndarray]]): Ground-truth PSTHs.  Outer list
            has length ``N`` (neurons); inner list has one 1-D array per
            coherence group.
        sim_psths (list[list[np.ndarray]]): Simulated PSTHs with the same
            structure as ``true_psths``.  Simulated arrays are truncated to
            the length of the corresponding true array if longer.

    Returns:
        np.ndarray: Per-neuron summed absolute deviation values, shape
        ``(N,)``.
    """
    assert len(true_psths) == len(sim_psths)
    N = len(true_psths)

    mads = np.zeros(N)

    for i in range(N):
        true_psth = true_psths[i]
        sim_psth = sim_psths[i]

        mads_n = 0.0

        NC = len(true_psth)
        for j in range(NC):
            T = true_psth[j].shape[0]
            if T > 0:
                mads_n += np.nansum(np.abs(true_psth[j] - sim_psth[j][:T]))

        mads[i] = mads_n

    return mads


def plot_multiple_psths(psth_list, neuron_idx=None):
    """Plot PSTHs for multiple models side-by-side in a neuron × model grid.

    Each column corresponds to one model and each row to one neuron.
    Coherence groups within each subplot are coloured using the default
    red-to-blue palette of :func:`plot_neuron_psth`.

    Args:
        psth_list (list[list[list[np.ndarray]]]): List of PSTH collections,
            one per model.  Each element is itself a list indexed by neuron,
            containing the coherence-group PSTH arrays for that neuron.
        neuron_idx (np.ndarray, optional): Indices of neurons to plot.  If
            ``None``, all neurons in ``psth_list[0]`` are plotted. Defaults
            to ``None``.

    Returns:
        None
    """
    num_models = len(psth_list)
    if neuron_idx is None:
        neuron_idx = np.arange(0, len(psth_list[0]))
    num_neurons = neuron_idx.shape[0]

    plt.figure()
    for i in range(num_models):
        for j in range(num_neurons):
            plt.subplot(num_neurons, num_models, (j) * num_models + i + 1)
            plot_neuron_psth(psth_list[i][neuron_idx[j]])

    return


def plot_psth_grid(psths, num_row=1, num_col=1, ylim=None, flip_colors=False):
    """Plot a grid of per-neuron PSTHs from a single model.

    Creates a new figure and arranges one subplot per neuron in a
    ``num_row × num_col`` grid.

    Args:
        psths (list[list[np.ndarray]]): PSTH collection for a single model,
            indexed by neuron.  Each element is a list of coherence-group PSTH
            arrays for that neuron.
        num_row (int): Number of subplot rows. Defaults to 1.
        num_col (int): Number of subplot columns. Defaults to 1.
        ylim (tuple, optional): ``(ymin, ymax)`` y-axis limits applied to
            every subplot.  Not applied if ``None``. Defaults to ``None``.
        flip_colors (bool or list[bool]): If a scalar ``True``, the colour
            order is reversed for all subplots.  If a list, each element
            controls the corresponding neuron's subplot independently.
            Defaults to ``False``.

    Returns:
        None
    """
    if np.shape(flip_colors) == ():
        flip_colors = [flip_colors for i in range(num_row * num_col)]
    num_neurons = len(psths)
    plt.figure()
    for i in range(num_neurons):
        plt.subplot(num_row, num_col, i + 1)
        plot_neuron_psth(psths[i], ylim=ylim, flip_colors=flip_colors[i])

    return
