import copy

import torch
import torch.nn as nn
from tqdm import tqdm
import math
import numpy as np


class ConditionalEnDecoder(nn.Module):

    def __init__(self, in_dim, out_dim, h_dim=None, n_layers=1, is_encoder=True):
        super().__init__()
        self.n_layers = n_layers
        if n_layers == 1:
            self.model = nn.Linear(in_dim, out_dim, bias=True)

        elif n_layers > 1:
            modules = []
            modules.append(nn.Linear(in_dim, h_dim, bias=False))
            modules.append(nn.ReLU())
            for _ in range(1, n_layers - 1):
                modules.append(nn.Linear(h_dim, h_dim, bias=False))
                modules.append(nn.ReLU())
            modules.append(nn.Linear(h_dim, out_dim, bias=not is_encoder))
            self.model = nn.Sequential(*modules)

    def forward(self, x, prot_means=None, cond=None):
        if (prot_means is not None) & (self.n_layers > 1):
            # print('substracting prot means')
            x = x - prot_means
        if cond is not None:
            x = torch.cat([x, cond], 1)
        return self.model(x)


### input: s x (g + cov)
### encoder: (g+cov) x h
### latent: s x h
### decoder:(h+cov) x g
### output: s x g

class ProtriderAutoencoder(nn.Module):
    def __init__(self, in_dim, latent_dim, n_layers=1, n_cov=0, h_dim=None ):
        super().__init__()
        self.n_layers = n_layers
        self.encoder = ConditionalEnDecoder(in_dim=in_dim + n_cov,
                                            out_dim=latent_dim,
                                            h_dim=h_dim, n_layers=n_layers,
                                            is_encoder=True)

        self.decoder = ConditionalEnDecoder(in_dim=latent_dim + n_cov,
                                            out_dim=in_dim,
                                            h_dim=h_dim, n_layers=n_layers,
                                            is_encoder=False
                                            )

    def forward(self, x, prot_means=None, cond=None):
        return self.decoder(self.encoder(x, cond=cond, prot_means=prot_means),
                            cond=cond, prot_means=None)

    def _initialize_wPCA(self, Vt_q, prot_means, n_cov=0):
        if self.n_layers > 1:
            print('[Warning] Initialization only possible for n_layers=1. Going back to random init...')
            self.decoder.model[-1].bias.data.copy_(torch.from_numpy(prot_means).squeeze(0))
            return
        stdv = 1. / math.sqrt(n_cov + 1)

        self.encoder.model.weight.data.copy_(torch.from_numpy(Vt_q))
        b = torch.cat([torch.from_numpy(prot_means),
                       torch.FloatTensor(1, n_cov).uniform_(-stdv, stdv)  # alternatively just set to zero
                       ], axis=1)  # prot_means
        self.encoder.model.bias.data.copy_(-(torch.from_numpy(Vt_q) @ b.T).flatten())

        ## Init cov with uniform distribution in range stdv

        cov_dec_init = self.decoder.model.weight.data.uniform_(-stdv, stdv)[:, 0:n_cov]
        self.decoder.model.weight.data.copy_(torch.cat([torch.from_numpy(Vt_q.T)[:prot_means.shape[1]].to(self.decoder.model.weight.data.device),
                                                        cov_dec_init], axis=1))
        self.decoder.model.bias.data.copy_( torch.from_numpy(prot_means).squeeze(0))


def mse_masked(x_hat, x, mask):
    mse_loss = nn.MSELoss(reduction="none")
    loss = mse_loss(x_hat, x)
    masked_loss = torch.where(mask, torch.nan, loss)
    mse_loss_val = masked_loss.nanmean()
    return mse_loss_val


def train_val(train_subset, val_subset, model, n_epochs=100, learning_rate=1e-3, val_every_nepochs=1, batch_size=None,
              patience=100, min_delta=0.001):
    # start data;pader
    if batch_size is None:
        batch_size = train_subset.X.shape[0]
    data_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    min_val_loss = np.inf
    best_model_wts = copy.deepcopy(model.state_dict())
    early_stopping_counter = 0
    early_stopping_epoch = 0

    train_losses = []
    val_losses = []
    for epoch in tqdm(range(n_epochs)):
        train_loss = _train_iteration(data_loader, model, optimizer)

        if epoch % val_every_nepochs == 0:
            train_losses.append(train_loss)
            x_hat_val = model(val_subset.X, prot_means=val_subset.prot_means_torch, cond=val_subset.cov_one_hot)
            val_loss = mse_masked(val_subset.X, x_hat_val, val_subset.torch_mask).detach().cpu().numpy()
            val_losses.append(val_loss)
            print('[%d] train loss: %.6f' % (epoch + 1, train_loss))
            print('[%d] validation loss: %.6f' % (epoch + 1, val_loss))

            if min_val_loss - val_loss > min_delta:
                min_val_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                early_stopping_counter = 0
                early_stopping_epoch = epoch + 1
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    print(f"\tEarly stopping at epoch {epoch + 1}")
                    break

    print('\tRestoring model weights from epoch', early_stopping_epoch)
    model.load_state_dict(best_model_wts)
    # make losses a 2d array
    return np.array(train_losses), np.array(val_losses)


def train(dataset, model,
          n_epochs=100, learning_rate=1e-3,
          batch_size=None, verbose=False):
    # start data;pader
    if batch_size is None:
        batch_size = dataset.X.shape[0]
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in tqdm(range(n_epochs)):
        running_loss = _train_iteration(data_loader, model, optimizer)
        if verbose:
            print('[%d] loss: %.6f' % (epoch + 1, running_loss))
    return running_loss


def _train_iteration(data_loader, model, optimizer):
    running_loss = 0.0
    n_batches = 0
    for batch_idx, data in enumerate(data_loader):
        x, mask, cov, prot_means = data

        # restore grads and compute model out
        optimizer.zero_grad()
        x_hat = model(x, prot_means=prot_means, cond=cov)

        # Compute the loss and its gradients
        loss = mse_masked(x_hat, x, mask)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        n_batches += 1

    return running_loss / n_batches
