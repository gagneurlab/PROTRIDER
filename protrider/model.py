import copy

import torch
import torch.nn as nn
from tqdm import tqdm
import math
import numpy as np
import logging
import torch.nn.functional as F
logger = logging.getLogger(__name__)


class ConditionalEnDecoder(nn.Module):

    def __init__(self, in_dim, out_dim, is_encoder, h_dim=None, n_layers=1, 
                 prot_means=None, presence_absence=False):
        super().__init__()
        if presence_absence & (prot_means is not None):
            self.prot_means = torch.cat([prot_means, 
                                     torch.zeros(prot_means.shape).to(prot_means.device)]) # for substraction in forward
        else: 
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
            modules = []
            modules.append(nn.Linear(in_dim, h_dim, bias=False))
            modules.append(nn.ReLU())
            for _ in range(1, n_layers - 1):
                modules.append(nn.Linear(h_dim, h_dim, bias=False))
                modules.append(nn.ReLU())
            # if the model is a decoder, then we want to have trainable bias
            last_layer = nn.Linear(h_dim, out_dim, bias=not is_encoder or prot_means is None)
            modules.append(last_layer)
            self.model = nn.Sequential(*modules)

        # if the model is a decoder, then the bias should be initialized to the protein means
        if not is_encoder and prot_means is not None:
            if presence_absence:
                prot_means = torch.cat([prot_means, last_layer.bias.data[out_dim//2:].to(prot_means.device)])
            last_layer.bias.data.copy_(prot_means).squeeze(0)

    def forward(self, x, cond=None):
        if self.is_encoder and (self.prot_means is not None):
            x = x - self.prot_means  
        if cond is not None:
            x = torch.cat([x, cond], 1)
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
        self.encoder = ConditionalEnDecoder(in_dim=in_dim + n_cov,
                                            out_dim=latent_dim, h_dim=h_dim, n_layers=n_layers,
                                            is_encoder=True, prot_means=prot_means,
                                            presence_absence=presence_absence
                                           )

        self.decoder = ConditionalEnDecoder(in_dim=latent_dim + n_cov,
                                            out_dim=in_dim,
                                            h_dim=h_dim, n_layers=n_layers,
                                            is_encoder=False, prot_means=prot_means,
                                            presence_absence=presence_absence
                                           )

    def forward(self, x, cond=None):
        z = self.encoder(x, cond=cond)
        out = self.decoder(z, cond=cond)
        return out
       

    def initialize_wPCA(self, Vt_q, prot_means, n_cov=0, presence_absence=False):
        if self.n_layers > 1:
            logger.warning('Initialization only possible for n_layers=1. Going back to random init...')
            return
        
        n_prots = prot_means.shape[1] # (1, n_prots)
        presence_absence = 2*n_prots==self.decoder.model.bias.shape[0]
        
        device = self.encoder.model.weight.device
        Vt_q = torch.from_numpy(Vt_q).to(device)
        stdv = 1. / math.sqrt(n_cov + 1)
        
        ## ENCODER
        self.encoder.model.weight.data.copy_(Vt_q)
        enc_bias = self.encoder.model.bias.data
        if presence_absence:
            b = torch.cat([torch.from_numpy(prot_means).to(device),
                           torch.ones((1, n_prots)).to(device), ### check alternative init
                           torch.FloatTensor(1, n_cov).uniform_(-stdv, stdv).to(device)  # alternatively just set to zero
                           ], axis=1)  # prot_means
        else:
            b = torch.cat([torch.from_numpy(prot_means).to(device),
                           torch.FloatTensor(1, n_cov).uniform_(-stdv, stdv).to(device)  # alternatively just set to zero
                           ], axis=1)
        self.encoder.model.bias.data.copy_(-(Vt_q @ b.T).flatten())


        ## DECODER
        ## weights: (n_prots or n_prots*2, q+cov), bias: (n_prot*2)

        # Bias 
        prot_means = torch.from_numpy(prot_means)
        
        if presence_absence:
            prot_means = torch.cat([prot_means.to(device), 
                                    self.decoder.model.bias.data[n_prots:].unsqueeze(0).to(device)],
                                   axis=1)
        self.decoder.model.bias.data.copy_(prot_means.squeeze(0))

        #Init cov with uniform distribution in range stdv
        cov_dec_init = self.decoder.model.weight.data.uniform_(-stdv, stdv)[:, 0:n_cov]
        self.decoder.model.weight.data.copy_(
            torch.cat([Vt_q.T[:(2*n_prots if presence_absence else n_prots)].to(device),
                       cov_dec_init.to(device)], axis=1)
        )
        

def mse_masked(x_hat, x, mask):
    mse_loss = nn.MSELoss(reduction="none")
    loss = mse_loss(x_hat, x)
    masked_loss = torch.where(mask, torch.nan, loss)
    mse_loss_val = masked_loss.nanmean()
    return mse_loss_val

def mse_bce_loss(x_hat, x, mask, lambda_bce, presence_absence, detached=False):

    if presence_absence:
        presence = (~mask).double()
        n = x_hat.shape[1] // 2
        presence_hat = x_hat[:, n:]       # Predicted presence (0–1)
        x_hat = x_hat[:, :n]              # Predicted intensities

    mse_loss = mse_masked(x_hat, x, mask)
    if detached:
        mse_loss = mse_loss.detach().cpu().numpy()
    
    if presence_absence:
        bce_loss = bce_loss = F.binary_cross_entropy(torch.sigmoid(presence_hat), presence)
        if detached:
            bce_loss = bce_loss.detach().cpu().numpy()
        loss = mse_loss + lambda_bce*bce_loss
    else:
        bce_loss = None
        loss = mse_loss

    
    return loss, mse_loss, bce_loss
    

def train_val(train_subset, val_subset, model, n_epochs=100, learning_rate=1e-3, val_every_nepochs=1, batch_size=None,
              patience=100, min_delta=0.001, presence_absence=False, lambda_bce=1.):
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
        train_loss, train_mse_loss, train_bce_loss = _train_iteration(data_loader, model, 
                                                                      optimizer, presence_absence, lambda_bce)

        if epoch % val_every_nepochs == 0:
            train_losses.append(train_loss)

            if presence_absence:
                presence = (~val_subset.torck_mask).double()
                x_in = torch.hstack([val_subset.X, presence])
            else: 
                x_in = val_subset.X
            
            x_hat_val = model(x_in, cond=val_subset.cov_one_hot)
            val_loss, val_mse_loss, val_bce_loss = mse_bce_loss(x_hat_val, val_subset.X, 
                                                                val_subset.torch_mask, 
                                                                lambda_bce, presence_absence).detach().cpu().numpy()
            
            val_losses.append(val_loss)
            logger.debug('[%d] train loss: %.6f' % (epoch + 1, train_loss))
            logger.debug('[%d] validation loss: %.6f' % (epoch + 1, val_loss))

            if min_val_loss - val_loss > min_delta:
                min_val_loss = val_loss
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
    # make losses a 2d array
    return np.array(train_losses), np.array(val_losses)


def train(dataset, model,
          n_epochs=100, learning_rate=1e-3,
          batch_size=None, presence_absence=False,lambda_bce=1.):
    # start data;pader
    if batch_size is None:
        batch_size = dataset.X.shape[0]
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in tqdm(range(n_epochs)):
        running_loss, running_mse_loss, running_bce_loss = _train_iteration(data_loader, model, 
                                                                            optimizer, presence_absence, lambda_bce)
        logger.debug('[%d] loss: %.6f, mse loss: %.6f, bce loss: %.6f' % (epoch + 1, running_loss,
                                                                          running_mse_loss, running_bce_loss))
    return running_loss, running_mse_loss, running_bce_loss


def _train_iteration(data_loader, model, optimizer, presence_absence, lambda_bce):
    running_loss = 0.0
    running_mse_loss = 0.0
    running_bce_loss = 0.0
    
    n_batches = 0
    for batch_idx, data in enumerate(data_loader):
        x, mask, cov, prot_means = data
        if presence_absence:
            presence = (~mask).double()
            x_in = torch.hstack([x, presence])
        else: 
            x_in = x

        # restore grads and compute model out
        optimizer.zero_grad()
        x_hat = model(x_in, cond=cov)
        
        loss, mse_loss, bce_loss = mse_bce_loss(x_hat, x, mask, lambda_bce, presence_absence)
        
        # Adjust learning weights
        loss.backward()
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        running_mse_loss += mse_loss.item()
        running_bce_loss += bce_loss.item() if bce_loss is not None else 0
        n_batches += 1

    return running_loss / n_batches, running_mse_loss / n_batches, running_bce_loss / n_batches
