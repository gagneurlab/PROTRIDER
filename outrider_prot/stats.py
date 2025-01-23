import pandas as pd
import numpy as np
import scipy
import tqdm

def get_pvals(res, how='two-sided', dis='gaussian', padjust=None):
    hows =  ('two-sided', 'left', 'right')
    if not how in hows:
        raise ValueError(f'Method should be in <{hows}>')
    dists = ('gaussian', 't')
    if not dis in dists:
        raise ValueError(f'Distribution should be in <{dists}>')

    if dis=='gaussian':
        pvals, z = get_pv_norm(res, how)
    else:
        pv_t, pvals, dfs, z = get_pv_t(res, how)

    if padjust is not None:
        mask = ~np.isfinite(pvals)
        pvals_adj = false_discovery_control(np.where(mask, 1, pvals), axis=1, method=padjust)
        pvals_adj[mask] = np.nan
    else:
        pvals_adj = None
        
    return pvals, z, pvals_adj

def get_pv_norm(res, how='two-sided'):

    mask = ~np.isfinite(res)
    # Compute z-scores
    sigma = np.nanstd(res, ddof=1, axis=0)
    z = (res - np.nanmean(res, axis=0)) / sigma # 
    
    hows =  ('two-sided', 'left', 'right')
    if not how in hows:
        raise ValueError(f'Method should be in <{hows}>')
    
    if how in ('left', 'two-sided'):
        left_pvals = pvals = scipy.stats.norm.cdf(np.where(mask, 0, z))
        pvals = left_pvals
    if how in ('right', 'two-sided'):
        right_pvals = scipy.stats.norm.sf(np.where(mask, 0, z))
        pvals = right_pvals
    if how=='two-sided':
        pvals = 2*np.minimum(left_pvals, right_pvals)
    pvals[mask] = np.nan
    return pvals, z


def get_pv_t_base(x, df=None, max_df=None, how='two-sided'):
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
    hows =  ('two-sided', 'left', 'right')
    if not how in hows:
        raise ValueError(f'Method should be in <{hows}>')
        
    # For df smaller than 2 the variance is not defined. 
    # We don't use these extreme heavy-tail distribs in practice.
    if df is not None and df <= 2:
        raise ValueError("Degrees of freedom (df) must be greater than 2.")
    
    # Mask for NaN values
    x = np.asarray(x)
    mask = ~np.isfinite(x) #np.isnan(x)
    pv = np.full_like(x, np.nan, dtype=np.float64)
    z = np.full_like(x, np.nan, dtype=np.float64)

    try:
        if df is None:
            # If ddof is not provided, we estimate it jointly with location and scale
            # This is known to be unstable.
            # It will often not converge (e.g. 10% of the cases)
            df, loc, scale = scipy.stats.t.fit(x[~mask]) #fitdistr(x[!mask], densfun = "t")
            #df, loc, scale = fit_student_t(x[~mask])
            if max_df is not None:
                df = df if df <= max_df else np.nan
        else:
            #loc, scale = fit_student_t_fixed_df(x[~mask], df=df) #fitdistr(x[!mask], densfun = "t", df=df)
            _, loc, scale = scipy.stats.t.fit(x[~mask], fix_df=df) #fitdistr(x[!mask], densfun = "t", df=df)
        
        z[~mask] = (x[~mask] - loc) / scale

        # Calculate p-values
        if how=='left':
            pv[~mask] = scipy.stats.t.cdf(z[~mask], df)
        elif how=='right':
            pv[~mask] = scipy.stats.t.sf(z[~mask], df)
        else:
            pv[~mask] = 2 * np.minimum(scipy.stats.t.cdf(z[~mask], df), scipy.stats.t.sf(z[~mask], df))
            
        return pv, df, z
    
    except Exception as e:
        print(e)
        return pv, np.nan, z


def get_pv_t(res, how='two-sided', MAX_DF=100000):
    # Fitting a Student's  degree of freedom is unstable
    # Moreover it tends to over-fit to outliers.
    # Hence, we fit Student for as many columns as possible in a first pass
    # Then we get the median df as default df
    # If do not we get enough converged fits (which is unlikely), we take df=10 as default
    # and we fit again with using that common default df for all.
    
    # Initialize variables
    pv_t = np.full_like(res, np.nan, dtype=np.float64)  # Matrix to store p-values
    dfs = np.full(res.shape[1], np.nan, dtype=np.float64)  # Array to store degrees of freedom
    
    # First pass: Fit the degree of freedom for each column of the data matrix
    ## if pv is too large, replace with np.nan --> it means it did not converge
    for j in tqdm.tqdm(range(res.shape[1])):
        x = res[:, j]
        pv, df, _ = get_pv_t_base(x, how=how)  # Call the previously defined function
        pv_t[:, j] = pv  # Store p-values (optional, can omit this step in final code)
        dfs[j] = df if df<=MAX_DF else 10#np.nan
    

    # Report the number of non-converged fits
    print(f"1st pass did not converge for {np.sum(np.isnan(dfs))} out of {len(dfs)} samples.")
    
    # Determine the default degree of freedom
    if np.sum(~np.isnan(dfs)) >= 10:
        df0 = np.nanmedian(dfs)  # Use median df if enough fits converged
    else:
        df0 = 10  # Fallback default df
    
    print('Degree of freedom after first pass', df0)

    # Second pass, fit distribution now using df
    pv_t_df0 = np.full_like(res, np.nan, dtype=np.float64)  # Matrix to store p-values
    z_scores = np.full_like(res, np.nan, dtype=np.float64)  # Matrix to store z-scores
    for j in tqdm.tqdm(range(res.shape[1])):
        x = res[:, j]
        pv, _, z = get_pv_t_base(x, df=df0, how=how)
        pv_t_df0[:, j] = pv  
        z_scores[:, j] = z

    return pv_t, pv_t_df0, dfs, z_scores

def false_discovery_control(ps, *, axis=0, method='bh'):
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
    i = np.arange(1, m+1)
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

        
    