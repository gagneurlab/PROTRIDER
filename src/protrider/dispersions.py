import torch
import torch.optim as optim
import torch.nn as nn

from .estimate_theta_robust_moments import estimate_theta_robust_moments

__all__ = ['Dispersion', 'NegativeBinomialDistribution']

class Dispersion:
    def __init__(self, distribution):
        self.distribution = distribution
        self.mu_scale = None
        self.theta = None

    def get_parameters(self):
        mu_scale = None if self.mu_scale is None else self.mu_scale.detach().cpu().numpy()
        theta = None if self.theta is None else self.theta.detach().cpu().numpy()
        return mu_scale, theta
    
    def set_dispersion(self, theta):
        self.theta = theta

    def clip_theta(self, lower=0.01, upper=1000):
        self.theta = torch.clip(self.theta, lower, upper)

    def fit(self, x_true, x_pred, max_iter=100, lower_bound=0.01, device=None):
        """
        x_true, x_pred: torch.Tensor, shape (genes, samples)
        """

        x_true = x_true.to(dtype=torch.float64, device=device)
        x_pred = x_pred.to(dtype=torch.float64, device=device)
        
        mu_scale_init, theta_init = self.distribution.init_fit(x_true, x_pred)
        
        # Reparameterization using exp, since LBFGS does not consider bounds:
        # where theta = exp(p_theta) + lower_bound
        # This ensures parameters >= lower_bound
        # Use min=1e-8 to avoid log(0)
        p_theta_init = torch.log(torch.clamp(theta_init - lower_bound, min=1e-8))
        p_mu_scale_init = torch.log(torch.clamp(mu_scale_init - lower_bound, min=1e-8))

        p_theta = nn.Parameter(p_theta_init.to(device))
        p_mu_scale = nn.Parameter(p_mu_scale_init.to(device))

        optimizer = optim.LBFGS(
            [p_theta, p_mu_scale],
            max_iter=max_iter,
            history_size=5,
            tolerance_change=2.2e-9,
            line_search_fn="strong_wolfe"
        )

        def closure():
            optimizer.zero_grad()
            
            # Transform parameters (shape [genes])
            theta = torch.exp(p_theta) + lower_bound
            mu_scale = torch.exp(p_mu_scale) + lower_bound
            
            # Unsqueeze to [genes, 1] for broadcasting against data [genes, samples]
            theta = theta.unsqueeze(1)
            mu_scale = mu_scale.unsqueeze(1)

            # Calculate loss and backpropagate
            mu = x_pred * mu_scale
            loss = self.distribution.loss(x_true, theta, mu)
            loss.backward()
            return loss

        optimizer.step(closure)

        self.theta = (torch.exp(p_theta) + lower_bound).detach().cpu()
        self.mu_scale = (torch.exp(p_mu_scale) + lower_bound).detach().cpu()


class NegativeBinomialDistribution:
    def init_train(self, x_true, theta_min=0.01, theta_max=1000.0):
        """Initialize theta and mu for training: theta robust moments"""
        theta = estimate_theta_robust_moments(x_true=x_true.T, theta_min=theta_min, theta_max=theta_max)
        mu_scale = None
        return mu_scale, theta

    def init_fit(self, x_true, size_factors, epsilon=1e-8, theta_min=0.01, theta_max=1000.0, mu_min=0.01):
        """
        Initialize theta and mu for fitting: theta is dispersion, mu_scale the mean
        x_true, size_factors: torch.Tensor, shape (genes, samples)
        """
        x_true = x_true.to(torch.float64)
        size_factors = size_factors.to(torch.float64)
        normalized = x_true / (size_factors + epsilon)
        
        # Calculate mean and var per gene (dim=1)
        mu_scale = normalized.mean(dim=1)
        var = normalized.var(dim=1)

        theta = mu_scale**2 / (var - mu_scale + epsilon)

        # Set fallback for genes where var <= mu_scale or theta is invalid
        fallback_mask = (var <= mu_scale) | (theta <= 0) | torch.isnan(theta)
        theta[fallback_mask] = 1.0
        
        theta = torch.clamp(theta, theta_min, theta_max)
        mu_scale = torch.clamp(mu_scale, min=mu_min)
        return mu_scale, theta

    def loss(self, x_true, theta, mu):
        term_lgamma = torch.lgamma(x_true + theta) - torch.lgamma(theta) - torch.lgamma(x_true + 1)
        term_t_log_t = torch.xlogy(theta, theta)
        term_x_log_m = torch.xlogy(x_true, mu)
        term_tx_log_tm = torch.xlogy(x_true + theta, theta + mu)
        
        log_prob = term_lgamma + term_t_log_t + term_x_log_m - term_tx_log_tm
        log_prob = torch.nan_to_num(log_prob, nan=0.0, posinf=0.0, neginf=-1e20)
        loss = -torch.sum(log_prob)
        return loss
