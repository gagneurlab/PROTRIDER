import numpy as np
import scipy
import tqdm
from joblib import Parallel, delayed
import logging

__all__ = ["fit_residuals", "get_pvals", "adjust_pvals"]

logger = logging.getLogger(__name__)

def fit_residuals(res, dis='gaussian'):
    if dis == 'gaussian':
        sigma = np.nanstd(res, ddof=1, axis=0)
        mu = np.nanmean(res, axis=0)
        df0 = None
    else:
        mu, sigma, df0 = _fit_t(res)

    return mu, sigma, df0


def get_pvals(res, mu, sigma, df0=None, how='two-sided', dis='gaussian'):
    hows = ('two-sided', 'left', 'right')
    if not how in hows:
        raise ValueError(f'Method should be in <{hows}>')
    dists = ('gaussian', 't')
    if not dis in dists:
        raise ValueError(f'Distribution should be in <{dists}>')

    if dis == 'gaussian':
        pvals, z = _get_pv_norm(res, mu=mu, sigma=sigma, how=how, )
    else:
        assert df0 is not None, "df0 should be provided for t-distribution"
        pvals, z = get_pv_t(res, df0=df0, sigma=sigma, mu=mu, how=how)

    return pvals, z

def adjust_pvals(pvals, method='bh'):
    mask = ~np.isfinite(pvals)
    pvals_adj = _false_discovery_control(np.where(mask, 1, pvals), axis=1, method=method)
    pvals_adj[mask] = np.nan
    return pvals_adj


def _get_pv_norm(res, mu, sigma, how='two-sided'):
    mask = ~np.isfinite(res)
    # Compute z-scores
    z = (res - mu) / sigma  #

    hows = ('two-sided', 'left', 'right')
    if not how in hows:
        raise ValueError(f'Method should be in <{hows}>')

    if how in ('left', 'two-sided'):
        left_pvals = scipy.stats.norm.cdf(np.where(mask, 0, z))
        pvals = left_pvals
    if how in ('right', 'two-sided'):
        right_pvals = scipy.stats.norm.sf(np.where(mask, 0, z))
        pvals = right_pvals
    if how == 'two-sided':
        pvals = 2 * np.minimum(left_pvals, right_pvals)
    pvals[mask] = np.nan
    return pvals, z


def _get_pv_t_base(x, mu, sigma, df, how='two-sided'):
    """
    Fit a Student's t distribution to data in vector x.
    Returns the corresponding two-sided p-values and the (fitted) degree of freedom.
    The degree of freedom may or may not be provided.

    Parameters:
        x (array-like): Data to fit.
        df (float, optional): Degrees of freedom. If None, df is estimated.

    Returns:
        dict: A dictionary containing:
            - 'pv': Two-sided p-values.
            - 'df': Fitted degrees of freedom.
    """
    hows = ('two-sided', 'left', 'right')
    if not how in hows:
        raise ValueError(f'Method should be in <{hows}>')

    # For df smaller than 2 the variance is not defined. 
    # We don't use these extreme heavy-tail distribs in practice.
    if df is not None and df <= 2:
        raise ValueError("Degrees of freedom (df) must be greater than 2.")

    # Mask for NaN values
    x = np.asarray(x)
    mask = ~np.isfinite(x)  # np.isnan(x)
    pv = np.full_like(x, np.nan, dtype=np.float64)
    z = np.full_like(x, np.nan, dtype=np.float64)

    try:
        z[~mask] = (x[~mask] - mu) / sigma

        # Calculate p-values
        if how == 'left':
            pv[~mask] = scipy.stats.t.cdf(z[~mask], df)
        elif how == 'right':
            pv[~mask] = scipy.stats.t.sf(z[~mask], df)
        else:
            pv[~mask] = 2 * np.minimum(scipy.stats.t.cdf(z[~mask], df), scipy.stats.t.sf(z[~mask], df))

        return pv, z

    except Exception as e:
        logger.error(e)
        return pv, z


def _fit_t_base(x, max_df=None, df=None):
    # Mask for NaN values
    x = np.asarray(x)
    mask = ~np.isfinite(x)  # np.isnan(x)
    if df is None:
        # If ddof is not provided, we estimate it jointly with location and scale
        # This is known to be unstable.
        # It will often not converge (e.g. 10% of the cases)
        df, mu, sigma = scipy.stats.t.fit(x[~mask])
        if max_df is not None:
            df = df if df <= max_df else np.nan
    else:
        _, mu, sigma = scipy.stats.t.fit(x[~mask], fix_df=df)  # fitdistr(x[!mask], densfun = "t", df=df)

    return mu, sigma, df


def _fit_t(res, max_df=100000):
    """
    Determine the degree of freedom for each protein in the data matrix. Take the median of the
    converged fits as the default degree of freedom.
    Args:
        res:
        MAX_DF:
        how:

    Returns:

    """

    # Fitting a Student's  degree of freedom is unstable
    # Moreover it tends to over-fit to outliers.
    # Hence, we fit Student for as many columns as possible in a first pass
    # Then we get the median df as default df
    # If do not we get enough converged fits (which is unlikely), we take df=10 as default
    # and we fit again with using that common default df for all.

    def first_pass(j):
        _, _, _df = _fit_t_base(res[:, j], max_df=max_df)
        return j, _df

    # Initialize variables
    df = np.full(res.shape[1], np.nan, dtype=np.float64)  # Array to store degrees of freedom
    # First pass: Fit the degree of freedom for each column of the data matrix
    ## if pv is too large, replace with np.nan --> it means it did not converge
    results = Parallel(n_jobs=-1)(delayed(first_pass)(j) for j in tqdm.trange(res.shape[1]))
    for j, _df in results:
        df[j] = _df

    # Report the number of non-converged fits
    logger.info(f"1st pass did not converge for {np.sum(np.isnan(df))} out of {len(df)} samples.")

    # Determine the default degree of freedom
    if np.sum(~np.isnan(df)) >= 10:
        df0 = np.nanmedian(df)  # Use median df if enough fits converged
        #df0 = max(df0, 9)
    else:
        df0 = 10  # Fallback default df
    logger.info('Degrees of freedom after first pass %s', df0)

    def second_pass(j):
        _mu, _sigma, _ = _fit_t_base(res[:, j], df=df0, max_df=max_df)
        return j, _mu, _sigma

    # Initialize variables
    mu = np.full(res.shape[1], np.nan, dtype=np.float64)
    sigma = np.full(res.shape[1], np.nan, dtype=np.float64)
    # Second pass, fit distribution now using df
    results = Parallel(n_jobs=-1)(delayed(second_pass)(j) for j in tqdm.trange(res.shape[1]))
    for j, _mu, _sigma in results:
        mu[j] = _mu
        sigma[j] = _sigma

    return mu, sigma, df0


def get_pv_t(res, sigma, mu, df0, how='two-sided', max_df=100000):
    # Validate the optional distribution parameters
    if not isinstance(sigma, (int, float)):
        assert len(sigma) == res.shape[
            1], "sigma should be a scalar or a vector of the same length as the number of proteins"
    if not isinstance(mu, (int, float)):
        assert len(mu) == res.shape[1], "mu should be a scalar or a vector of the same length as the number of proteins"
    assert isinstance(df0, (int, float)), "df0 should be a scalar"

    def process_column_with_df(j):
        x = res[:, j]
        _mu = mu[j] if (mu is not None) and (len(mu) > 1) else mu
        _sigma = sigma[j] if (sigma is not None) and (len(sigma) > 1) else sigma
        pv, z = _get_pv_t_base(x, mu=_mu, sigma=_sigma, df=df0, how=how)
        return j, pv, z

    pv_t = np.full_like(res, np.nan, dtype=np.float64)  # Matrix to store p-values
    z_scores = np.full_like(res, np.nan, dtype=np.float64)  # Matrix to store z-scores
    results = Parallel(n_jobs=-1)(delayed(process_column_with_df)(j) for j in tqdm.trange(res.shape[1]))
    for j, pv, z in results:
        pv_t[:, j] = pv
        z_scores[:, j] = z

    return pv_t, z_scores


def _false_discovery_control(ps, *, axis=0, method='bh'):
    # Input Validation and Special Cases
    ps = np.asarray(ps)

    ps_in_range = (np.issubdtype(ps.dtype, np.number)
                   and np.all(ps == np.clip(ps, 0, 1)))
    if not ps_in_range:
        raise ValueError("`ps` must include only numbers between 0 and 1.")

    methods = {'bh', 'by'}
    if method.lower() not in methods:
        raise ValueError(f"Unrecognized `method` '{method}'."
                         f"Method must be one of {methods}.")
    method = method.lower()

    if axis is None:
        axis = 0
        ps = ps.ravel()

    axis = np.asarray(axis)[()]
    if not np.issubdtype(axis.dtype, np.integer) or axis.size != 1:
        raise ValueError("`axis` must be an integer or `None`")

    if ps.size <= 1 or ps.shape[axis] <= 1:
        return ps[()]

    ps = np.moveaxis(ps, axis, -1)
    m = ps.shape[-1]

    # Main Algorithm
    # Equivalent to the ideas of [1] and [2], except that this adjusts the
    # p-values as described in [3]. The results are similar to those produced
    # by R's p.adjust.

    # "Let [ps] be the ordered observed p-values..."
    order = np.argsort(ps, axis=-1)
    ps = np.take_along_axis(ps, order, axis=-1)  # this copies ps

    # Equation 1 of [1] rearranged to reject when p is less than specified q
    i = np.arange(1, m + 1)
    ps *= m / i

    # Theorem 1.3 of [2]
    if method == 'by':
        ps *= np.sum(1 / i)

    # accounts for rejecting all null hypotheses i for i < k, where k is
    # defined in Eq. 1 of either [1] or [2]. See [3]. Starting with the index j
    # of the second to last element, we replace element j with element j+1 if
    # the latter is smaller.
    np.minimum.accumulate(ps[..., ::-1], out=ps[..., ::-1], axis=-1)

    # Restore original order of axes and data
    np.put_along_axis(ps, order, values=ps.copy(), axis=-1)
    ps = np.moveaxis(ps, -1, axis)

    return np.clip(ps, 0, 1)
