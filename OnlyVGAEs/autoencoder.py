import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINConv, GCNConv, GraphConv, PNAConv
from torch_geometric.nn import global_add_pool


# Decoder
class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        hidden_dim,
        n_layers,
        n_nodes,
        *,
        node_hidden_dim: int = None,
        dropout: float = 0.0,
        use_gumbel: bool = False,
        gumbel_tau: float = 1.0,
    ):
        super(Decoder, self).__init__()
        self.n_layers = max(n_layers, 1)
        self.n_nodes = n_nodes
        self.use_gumbel = use_gumbel
        self.gumbel_tau = gumbel_tau
        self.node_hidden_dim = node_hidden_dim or hidden_dim

        mlp_layers = []
        in_dim = latent_dim
        for _ in range(self.n_layers - 1):
            linear = nn.Linear(in_dim, hidden_dim)
            nn.init.xavier_uniform_(linear.weight)
            mlp_layers.append(linear)
            in_dim = hidden_dim
        self.hidden_layers = nn.ModuleList(mlp_layers)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.node_projection = nn.Linear(in_dim, self.n_nodes * self.node_hidden_dim)
        nn.init.xavier_uniform_(self.node_projection.weight)
        self.node_norm = nn.LayerNorm(self.node_hidden_dim)
        self.edge_bias = nn.Parameter(torch.zeros(self.n_nodes, self.n_nodes))

        mask = 1.0 - torch.eye(self.n_nodes)
        self.register_buffer("off_diag_mask", mask)

    def forward(self, latent):
        h = latent
        for linear in self.hidden_layers:
            h = self.activation(linear(h))
            h = self.dropout(h)

        node_repr = self.node_projection(h)
        node_repr = node_repr.view(latent.size(0), self.n_nodes, self.node_hidden_dim)
        node_repr = self.node_norm(node_repr)
        node_repr = self.dropout(node_repr)

        scores = torch.matmul(node_repr, node_repr.transpose(1, 2))
        scores = scores / math.sqrt(self.node_hidden_dim)
        bias = 0.5 * (self.edge_bias + self.edge_bias.t())
        scores = scores + bias.unsqueeze(0)

        if self.use_gumbel:
            stacked = torch.stack([scores, torch.zeros_like(scores)], dim=-1)
            sampled = F.gumbel_softmax(stacked, tau=self.gumbel_tau, hard=True)[..., 0]
            adj = sampled
        else:
            adj = torch.sigmoid(scores)

        adj = adj * self.off_diag_mask
        adj = 0.5 * (adj + adj.transpose(1, 2))
        return adj


class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim),  
                            nn.LeakyReLU(0.2),
                            nn.BatchNorm1d(hidden_dim),
                            nn.Linear(hidden_dim, hidden_dim), 
                            nn.LeakyReLU(0.2))
                            ))                        
        for layer in range(n_layers-1):
            self.convs.append(GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim),  
                            nn.LeakyReLU(0.2),
                            nn.BatchNorm1d(hidden_dim),
                            nn.Linear(hidden_dim, hidden_dim), 
                            nn.LeakyReLU(0.2))
                            )) 

        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, latent_dim)
        

    def forward(self, data):
        edge_index = data.edge_index
        x = data.x

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.dropout(x, self.dropout, training=self.training)

        out = global_add_pool(x, data.batch)
        out = self.bn(out)
        out = self.fc(out)
        return out


class PNA(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.convs.append(PNAConv(input_dim, hidden_dim))                        
        for layer in range(n_layers-1):
            self.convs.append(PNAConv(hidden_dim, hidden_dim))

        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, latent_dim)
        self.relu = nn.ReLU()
        

    def forward(self, data):
        edge_index = data.edge_index
        x = data.x

        for conv in self.convs:
            x = self.relu(conv(x, edge_index))
            x = F.dropout(x, self.dropout, training=self.training)

        out = global_add_pool(x, data.batch)
        out = self.bn(out)
        out = self.fc(out)
        return out


# Autoencoder
class AutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim_enc,
        hidden_dim_dec,
        latent_dim,
        n_layers_enc,
        n_layers_dec,
        n_max_nodes,
        *,
        decoder_node_hidden_dim: int = None,
        decoder_dropout: float = 0.0,
        decoder_use_gumbel: bool = False,
        decoder_gumbel_tau: float = 1.0,
    ):
        super(AutoEncoder, self).__init__()
        self.n_max_nodes = n_max_nodes
        self.input_dim = input_dim
        self.encoder = GIN(input_dim, hidden_dim_enc, latent_dim, n_layers_enc)
        self.decoder = Decoder(
            latent_dim,
            hidden_dim_dec,
            n_layers_dec,
            n_max_nodes,
            node_hidden_dim=decoder_node_hidden_dim,
            dropout=decoder_dropout,
            use_gumbel=decoder_use_gumbel,
            gumbel_tau=decoder_gumbel_tau,
        )

    def forward(self, data):
        x_g = self.encoder(data)
        adj = self.decoder(x_g)
        return adj

    def encode(self, data):
        x_g = self.encoder(data)
        return x_g

    def decode(self, x_g):
        adj = self.decoder(x_g)
        return adj

    def loss_function(self, data):
        x_g  = self.encoder(data)
        adj = self.decoder(x_g)
        A = data.A[:,:,:,0]
        return F.l1_loss(adj, data.A)


# Variational Autoencoder
class VariationalAutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim_enc,
        hidden_dim_dec,
        latent_dim,
        n_layers_enc,
        n_layers_dec,
        n_max_nodes,
        *,
        decoder_node_hidden_dim: int = None,
        decoder_dropout: float = 0.0,
        decoder_use_gumbel: bool = False,
        decoder_gumbel_tau: float = 1.0,
    ):
        super(VariationalAutoEncoder, self).__init__()
        self.n_max_nodes = n_max_nodes
        self.input_dim = input_dim
        #self.encoder = GPS(input_dim, hidden_dim_enc, hidden_dim_enc, n_layers_enc)
        self.encoder = GIN(input_dim, hidden_dim_enc, hidden_dim_enc, n_layers_enc)
        #self.encoder = Powerful(input_dim=input_dim+1, num_layers=n_layers_enc, hidden=hidden_dim_enc, hidden_final=hidden_dim_enc, dropout_prob=0.0, simplified=False)
        self.fc_mu = nn.Linear(hidden_dim_enc, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim_enc, latent_dim)
        self.decoder = Decoder(
            latent_dim,
            hidden_dim_dec,
            n_layers_dec,
            n_max_nodes,
            node_hidden_dim=decoder_node_hidden_dim,
            dropout=decoder_dropout,
            use_gumbel=decoder_use_gumbel,
            gumbel_tau=decoder_gumbel_tau,
        )

    def forward(self, data):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        adj = self.decoder(x_g)
        return adj

    def encode(self, data):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        return x_g

    def reparameterize(self, mu, logvar, eps_scale=1.):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, mu, logvar):
       x_g = self.reparameterize(mu, logvar)
       adj = self.decoder(x_g)
       return adj

    def decode_mu(self, mu):
       adj = self.decoder(mu)
       return adj

    def loss_function(self, data, beta=0.05):
        x_g  = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar) # concat or sum fully connected layer apo ta feats tou graph
        adj = self.decoder(x_g)
        
        #A = data.A[:,:,:,0]
        recon = F.l1_loss(adj, data.A, reduction='sum')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon + beta*kld

        return loss, recon, kld
