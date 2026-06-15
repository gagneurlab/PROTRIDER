import torch

BIAS_CORRECTION = 1.51
EPSILON = 1e-8

def estimate_theta_robust_moments(
    x_true: torch.Tensor,
    theta_min: float,
    theta_max: float,
    mu_min: float = 0.01,
) -> torch.Tensor:
    """
    Robust method of moments estimator for negative binomial theta parameter.

    Args:
        x_true: Count matrix of shape (n_samples, n_genes)
        theta_max: Upper bound for theta estimates
        theta_min: Lower bound for theta estimates
        mu_min: Lower bound for mean estimates to avoid numerical issues

    Returns:
        Estimated theta values of shape (n_genes,)
    """

    n_samples, _ = x_true.shape
    trim_fraction = 1 / 8
    n_trim = int(n_samples * trim_fraction)

    if n_trim * 2 >= n_samples:
        gene_means = torch.mean(x_true, dim=0)
        gene_means = torch.clamp(gene_means, min=mu_min)
        gene_vars = torch.var(x_true, dim=0, unbiased=True)
        theta_estimates = gene_means**2 / torch.clamp(
            gene_vars - gene_means, min=EPSILON
        )
    else:
        sorted_x_true = torch.sort(x_true, dim=0)[0]
        trimmed_x_true = (
            sorted_x_true[n_trim : n_samples - n_trim, :]
            if n_trim > 0
            else sorted_x_true
        )
        gene_means = torch.mean(trimmed_x_true, dim=0)
        gene_means = torch.clamp(gene_means, min=mu_min)

        squared_residuals = (x_true - gene_means.unsqueeze(0)) ** 2
        sorted_residuals = torch.sort(squared_residuals, dim=0)[0]
        trimmed_residuals = (
            sorted_residuals[n_trim : n_samples - n_trim, :]
            if n_trim > 0
            else sorted_residuals
        )
        trimmed_variance = torch.mean(trimmed_residuals, dim=0)

        corrected_variance = BIAS_CORRECTION * trimmed_variance
        variance_excess = torch.clamp(
            corrected_variance - gene_means, min=EPSILON
        )
        theta_estimates = gene_means**2 / variance_excess

    theta_estimates = torch.where(
        theta_estimates <= 0,
        torch.tensor(
            theta_max,
            dtype=theta_estimates.dtype,
            device=theta_estimates.device,
        ),
        theta_estimates,
    )

    return torch.clamp(theta_estimates, min=theta_min, max=theta_max)
