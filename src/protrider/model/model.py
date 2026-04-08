import copy
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import pandas as pd
from dataclasses import dataclass
import logging
import torch.nn.functional as F
from protrider.datasets import ProtriderSubset

logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    """Stores model information."""
    q: np.array
    learning_rate: np.array
    n_epochs: np.array
    test_loss: np.array
    train_losses: np.array

    def save(self, out_dir: str) -> None:
        """
        Save model information to CSV files.
        
        Args:
            out_dir: Output directory path
        """
        import pandas as pd
        from pathlib import Path
        
        logger.info('=== Saving model info ===')
        
        out_dir = Path(out_dir)

        train_losses = np.asarray(self.train_losses).flatten()
        if len(train_losses) > 0:
            train_losses_df = pd.DataFrame({
                'epoch': range(1, len(train_losses) + 1),
                'train_loss': train_losses,
            })
            out_p = out_dir / 'train_losses.csv'
            train_losses_df.to_csv(out_p, header=True, index=False)
            logger.info(f"Saved training losses to {out_p}")

        # Save additional info
        out_p = out_dir / 'additional_info.csv'
        df_info = pd.DataFrame({
            'q': [self.q.item() if hasattr(self.q, 'item') else self.q],
            'learning_rate': [self.learning_rate.item() if hasattr(self.learning_rate, 'item') else self.learning_rate],
            'n_epochs': [self.n_epochs.item() if hasattr(self.n_epochs, 'item') else self.n_epochs],
            'test_loss': [self.test_loss.item() if hasattr(self.test_loss, 'item') else self.test_loss],
        })
        df_info.to_csv(out_p, header=True, index=False)
        logger.info(f"Saved additional info to {out_p}")

    def plot_training_loss(self, out_dir: str = None, **kwargs):
        """
        Plot training loss history.
        
        Args:
            out_dir: Optional output directory for saving plots. If None, plot is returned but not saved.
            **kwargs: Additional arguments passed to the plotting function (plot_title, fontsize)
            
        Returns:
            plotnine plot object
            
        Example:
            >>> plot = model_info.plot_training_loss()  # Interactive use
            >>> plot.draw()
            >>> model_info.plot_training_loss(out_dir='output/')  # Save plot
        """
        from protrider import plots
        # Build train_losses DataFrame
        train_losses_df = pd.DataFrame({
            'epoch': range(1, len(self.train_losses) + 1),
            'train_loss': self.train_losses,
        })
        return plots.plot_training_loss(
            output_dir=out_dir,
            train_losses=train_losses_df,
            **kwargs
        )

class ConditionalEnDecoder(nn.Module):

    def __init__(self, in_dim, out_dim, is_encoder, h_dim=None, n_layers=1,
                 prot_means=None):
        super().__init__()
        self.prot_means = prot_means

        self.is_encoder = is_encoder
        self.n_layers = n_layers

        last_layer = None
        if n_layers == 1:
            # if the model is a decoder, then we want to have trainable bias
            last_layer = nn.Linear(in_dim,
                                   out_dim,
                                   bias=not is_encoder or prot_means is None)
            self.model = last_layer

        elif n_layers > 1:
            if h_dim is None:
                h_dim = out_dim if is_encoder else in_dim
            modules = []
            modules.append(nn.Linear(in_dim, h_dim))
            modules.append(nn.GELU())
            for _ in range(1, n_layers - 1):
                modules.append(nn.Linear(h_dim, h_dim))
                modules.append(nn.GELU())
            # if the model is a decoder, then we want to have trainable bias
            last_layer = nn.Linear(h_dim, out_dim, bias=not is_encoder or prot_means is None)
            modules.append(last_layer)
            self.model = nn.Sequential(*modules)

        # if the model is a decoder, then the bias should be initialized to the protein means
        if not is_encoder and prot_means is not None:
            last_layer.bias.data.copy_(prot_means).squeeze(0)

    def forward(self, x, cond=None):
        if self.is_encoder and (self.prot_means is not None):
            x = x - self.prot_means
        if cond is not None:
            x = torch.cat([x, cond], -1)
        return self.model(x)


### input: s x (g + cov)
### encoder: (g+cov) x h
### latent: s x h
### decoder:(h+cov) x g
### output: s x g

class ProtriderAutoencoder(nn.Module):
    def __init__(self, in_dim, latent_dim, n_layers=1, n_cov=0, h_dim=None,
                 prot_means=None, presence_absence=False):
        super().__init__()
        self.n_layers = n_layers
        self.presence_absence = presence_absence
        self.encoder = ConditionalEnDecoder(in_dim=in_dim + n_cov,
                                            out_dim=latent_dim, h_dim=h_dim, n_layers=n_layers,
                                            is_encoder=True, prot_means=prot_means)

        self.decoder = ConditionalEnDecoder(in_dim=latent_dim + n_cov,
                                            out_dim=in_dim,
                                            h_dim=h_dim, n_layers=n_layers,
                                            is_encoder=False, prot_means=prot_means)

    def forward(self, x, mask, cond=None):
        if self.presence_absence:
            presence = (~mask).double()
            x = torch.stack([x, presence])
            cond = torch.stack([cond, cond])

        z = self.encoder(x, cond=cond)
        out = self.decoder(z, cond=cond)
        return out

    def initialize_wPCA(self, Vt_q, prot_means, n_cov=0):
        if self.n_layers == 1:
            enc_layer = self.encoder.model
            dec_layer = self.decoder.model
        else:
            enc_layer = self.encoder.model[0]
            dec_layer = self.decoder.model[-1]

        if enc_layer.weight.data.shape[0] != Vt_q.shape[0] or dec_layer.weight.data.shape[1] != Vt_q.shape[0] + n_cov:
            logger.warning(f'PCA initialization skipped: layer dimensions do not match latent dim. '
                           f'This happens when h_dim is explicitly set.')
            return

        device = enc_layer.weight.device
        Vt_q = torch.from_numpy(Vt_q).to(device) # (q, n_prots)

        ## ENCODER weights: (q, n_prots + n_cov), bias: (q)
        cov_enc_init = enc_layer.weight.data[:, 0:n_cov]
        enc_layer.weight.data.copy_(
            torch.cat([Vt_q.to(device),
                       cov_enc_init.to(device)], axis=1)
        )

        enc_layer.bias.data.copy_(-(Vt_q @ torch.from_numpy(prot_means).to(device).T).flatten())

        ## DECODER weights: (n_prots, q + n_cov), bias: (n_prot)
        dec_layer.bias.data.copy_(torch.from_numpy(prot_means).squeeze(0))
        cov_dec_init = dec_layer.weight.data[:, 0:n_cov]
        dec_layer.weight.data.copy_(
            torch.cat([Vt_q.T.to(device),
                       cov_dec_init.to(device)], axis=1)
        )      

def mse_masked(x_hat, x, mask):
    mse_loss = nn.MSELoss(reduction="none")
    loss = mse_loss(x_hat, x)
    masked_loss = torch.where(mask, torch.nan, loss)
    mse_loss_val = masked_loss.nanmean()
    return mse_loss_val


class MSEBCELoss(nn.Module):
    def __init__(self, presence_absence=False, lambda_bce=1.):
        super().__init__()
        self.presence_absence = presence_absence
        self.lambda_bce = lambda_bce

    def forward(self, x_hat, x, mask, detached=False):
        if self.presence_absence:
            presence = (~mask).double()
            # n = x_hat.shape[1] // 2
            presence_hat = x_hat[1]  # Predicted presence (0–1)
            x_hat = x_hat[0]  # Predicted intensities

        mse_loss = mse_masked(x_hat, x, mask)
        if detached:
            mse_loss = mse_loss.detach().cpu().numpy()

        if self.presence_absence:
            bce_loss = F.binary_cross_entropy(torch.sigmoid(presence_hat), presence)
            if detached:
                bce_loss = bce_loss.detach().cpu().numpy()
            loss = mse_loss + self.lambda_bce * bce_loss
        else:
            bce_loss = None
            loss = mse_loss

        return loss, mse_loss, bce_loss


def train(dataset, model, criterion, n_epochs=100, learning_rate=1e-3, batch_size=None, wandb=None, patience=50, min_delta=1e-4):
    # start data;pader
    if batch_size is None:
        batch_size = dataset.X.shape[0]
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience // 4)
    train_losses = []
    min_train_loss = float('inf')
    early_stopping_counter = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    early_stopping_epoch = 0
    for epoch in tqdm(range(n_epochs)):
        running_loss, running_mse_loss, running_bce_loss = _train_iteration(data_loader, model, criterion, optimizer)
        scheduler.step(running_loss)
        logger.debug('[%d] loss: %.6f, mse loss: %.6f, bce loss: %.6f' % (epoch + 1, running_loss,
                                                                          running_mse_loss, running_bce_loss))
        train_losses.append(running_loss)
        if wandb is not None:
            wandb.log({
                'train/loss': running_loss,
                'train/mse_loss': running_mse_loss,
                'train/bce_loss': running_bce_loss
            })
        if min_train_loss - running_loss > min_delta:
            min_train_loss = running_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            early_stopping_counter = 0
            early_stopping_epoch = epoch + 1
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                logger.info(f"\tEarly stopping at epoch {epoch + 1}")
                break

    logger.info('\tRestoring model weights from epoch %s', early_stopping_epoch)
    model.load_state_dict(best_model_wts)
    return running_loss, running_mse_loss, running_bce_loss, train_losses


def _train_iteration(data_loader, model, criterion, optimizer):
    running_loss = 0.0
    running_mse_loss = 0.0
    running_bce_loss = 0.0

    n_batches = 0
    for batch_idx, data in enumerate(data_loader):
        x, mask, cov, prot_means = data

        # restore grads and compute model out
        optimizer.zero_grad()
        x_hat = model(x, mask, cond=cov)

        loss, mse_loss, bce_loss = criterion(x_hat, x, mask)

        # Adjust learning weights
        loss.backward()
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        running_mse_loss += mse_loss.item()
        running_bce_loss += bce_loss.item() if bce_loss is not None else 0
        n_batches += 1

    return running_loss / n_batches, running_mse_loss / n_batches, running_bce_loss / n_batches
