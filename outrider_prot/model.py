import torch
import torch.nn as nn
from tqdm import tqdm
import math

class ConditionalEnDecoder(nn.Module):

    def __init__(self, in_dim, out_dim, h_dim=None, n_layers=1):
        super().__init__()
        if n_layers==1:
            self.model = nn.Linear(in_dim, out_dim, bias=True)
            
        elif n_layers>1:
            modules = []
            modules.append( nn.Linear(in_dim, h_dim, bias=True) )
            for _ in range(1, n_layers-1):
                modules.append(nn.Linear(h_dim, h_dim, bias=True))
                modules.append(nn.ReLU())
            modules.append(nn.Linear(h_dim, out_dim, bias=True))
            self.model = nn.Sequential(*modules)

    def forward(self, x, cond=None):
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
        self.encoder = ConditionalEnDecoder(in_dim=in_dim+n_cov, 
                                            out_dim=latent_dim, 
                                            h_dim=h_dim, n_layers=n_layers)
        
        self.decoder = ConditionalEnDecoder(in_dim=latent_dim+n_cov, 
                                            out_dim=in_dim, 
                                            h_dim=h_dim, n_layers=n_layers)
        
    def forward(self, x, cond=None):
        return self.decoder(self.encoder(x, cond), cond)

    def _initialize_wPCA(self, Vt_q, prot_means, n_cov=0):
        if self.n_layers>1:
            print('[Warning] Initialization only possible for n_layers=1. Going back to random init...')
            return
        stdv = 1. / math.sqrt(n_cov+1)
        
        self.encoder.model.weight.data.copy_(torch.from_numpy(Vt_q))
        b = torch.cat([torch.from_numpy(prot_means), 
                       torch.FloatTensor(1, n_cov).uniform_(-stdv, stdv) # alternatively just set to zero
                      ], axis=1) # prot_means
        self.encoder.model.bias.data.copy_(-(torch.from_numpy(Vt_q) @ b.T).flatten())
    
        ## Init cov with uniform distribution in range stdv
        
        cov_dec_init = self.decoder.model.weight.data.uniform_(-stdv, stdv)[:, 0:n_cov]
        self.decoder.model.weight.data.copy_(torch.cat([torch.from_numpy(Vt_q.T)[:prot_means.shape[1]],
                                                        cov_dec_init], axis=1) )
        self.decoder.model.bias.data.copy_( torch.from_numpy( prot_means).squeeze(0))


def mse_masked(x_hat, x, mask):
    mse_loss = nn.MSELoss(reduction="none")
    loss = mse_loss(x_hat, x)
    masked_loss = torch.where(mask, torch.nan, loss)
    mse_loss_val = masked_loss.nanmean()
    return(mse_loss_val)


def train(dataset, model, 
          n_epochs = 100, learning_rate=1e-3, 
          batch_size=None):
    # start data;pader
    if batch_size is None:
        batch_size = dataset.X.shape[0]
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size, 
                                              shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
    for epoch in tqdm(range(n_epochs)):
        running_loss = _train_iteration(data_loader, model, optimizer)
        #print('[%d] loss: %.6f' %
        #      (epoch + 1, running_loss))
    return running_loss

def _train_iteration(data_loader, model, optimizer):
    running_loss = 0.0
    n_batches = 0
    for batch_idx, data in enumerate(data_loader):
        x, mask, cov = data

        # restore grads and compute model out
        optimizer.zero_grad()
        x_hat = model(x, cov)

        # Compute the loss and its gradients
        loss = mse_masked(x_hat, x, mask)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        n_batches += 1

    return running_loss / n_batches



