import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

from torch_geometric.nn import GINConv, GCNConv, GraphConv, PNAConv
from torch_geometric.nn import global_add_pool


# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes

        mlp_layers = [nn.Linear(latent_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for i in range(n_layers-2)]
        mlp_layers.append(nn.Linear(hidden_dim, 2*n_nodes*(n_nodes-1)//2))

        self.mlp = nn.ModuleList(mlp_layers)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for i in range(self.n_layers-1):
            x = self.relu(self.mlp[i](x))
        
        x = self.mlp[self.n_layers-1](x)
        x = torch.reshape(x, (x.size(0), -1, 2))
        x = F.gumbel_softmax(x, tau=1, hard=True)[:,:,0]

        adj = torch.zeros(x.size(0), self.n_nodes, self.n_nodes, device=x.device)
        idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1)
        adj[:,idx[0],idx[1]] = x
        adj = adj + torch.transpose(adj, 1, 2)
        return adj


class MultiHeadDecoder(nn.Module):
    """Decoder that mirrors the spectral generator with learnable parameters."""

    def __init__(
        self,
        latent_dim,
        hidden_dim,
        n_layers,
        n_nodes,
        feature_dim,
        num_classes,
        propagation_steps: int = 3,
        alpha_max: float = 0.8,
        affinity_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if n_layers < 2:
            raise ValueError("MultiHeadDecoder expects at least two layers")

        hidden_layers = [nn.Linear(latent_dim, hidden_dim)]
        for _ in range(n_layers - 2):
            hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.final_hidden = nn.Linear(hidden_dim, hidden_dim)

        self.relu = nn.ReLU()
        self.n_nodes = n_nodes
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.propagation_steps = propagation_steps
        self.alpha_max = alpha_max
        self.affinity_weight = affinity_weight

        # Global parameter heads
        self.class_affinity_head = nn.Linear(hidden_dim, num_classes * num_classes)
        self.alpha_head = nn.Linear(hidden_dim, 1)
        self.class_feature_mean_head = nn.Linear(hidden_dim, num_classes * feature_dim)
        self.class_feature_logvar_head = nn.Linear(hidden_dim, num_classes * feature_dim)

        # Node-level context
        self.node_embeddings = nn.Parameter(torch.empty(n_nodes, hidden_dim))
        nn.init.xavier_uniform_(self.node_embeddings)
        self.node_context_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.label_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

        # Pairwise edge scorer
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        base_logit = math.log(0.1 / 0.9)
        self.base_edge_logit = nn.Parameter(torch.tensor(base_logit, dtype=torch.float32))
        self.edge_scale_raw = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def _build_adjacency(
        self,
        node_context: torch.Tensor,
        label_probs: torch.Tensor,
        class_affinity: torch.Tensor,
    ) -> torch.Tensor:
        """Construct symmetric adjacency probabilities using neural and affinity cues."""

        batch_size, n_nodes, hidden_dim = node_context.shape

        ctx_i = node_context.unsqueeze(2).expand(batch_size, n_nodes, n_nodes, hidden_dim)
        ctx_j = node_context.unsqueeze(1).expand(batch_size, n_nodes, n_nodes, hidden_dim)
        edge_input = torch.cat([ctx_i, ctx_j], dim=-1)
        edge_scores = self.edge_mlp(edge_input.view(batch_size * n_nodes * n_nodes, -1))
        edge_scores = edge_scores.view(batch_size, n_nodes, n_nodes)
        edge_scores = 0.5 * (edge_scores + edge_scores.transpose(1, 2))

        affinity_distribution = torch.matmul(label_probs, class_affinity)
        affinity_scores = torch.matmul(affinity_distribution, label_probs.transpose(1, 2))

        eps = 1e-6
        affinity_scores = torch.clamp(affinity_scores, min=eps)
        affinity_logits = torch.log(affinity_scores)
        scale = F.softplus(self.edge_scale_raw) + 1e-3
        edge_logits = self.base_edge_logit + scale * edge_scores + self.affinity_weight * affinity_logits

        diag_mask = torch.eye(n_nodes, device=edge_logits.device).unsqueeze(0)
        edge_logits = edge_logits * (1.0 - diag_mask) - 10.0 * diag_mask

        adj_prob = torch.sigmoid(edge_logits)
        adj_prob = 0.5 * (adj_prob + adj_prob.transpose(1, 2))
        adj_prob = adj_prob * (1.0 - diag_mask)

        return adj_prob

    def _generate_features(
        self,
        adj: torch.Tensor,
        label_probs: torch.Tensor,
        class_means: torch.Tensor,
        class_logvars: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """Generate features via truncated spectral propagation."""

        batch_size, n_nodes, _ = adj.shape

        class_vars = torch.exp(class_logvars)
        mean_per_node = torch.matmul(label_probs, class_means)
        var_per_node = torch.matmul(label_probs, class_vars)

        if self.training:
            noise = torch.randn_like(mean_per_node)
            x0 = mean_per_node + noise * torch.sqrt(var_per_node + 1e-6)
        else:
            x0 = mean_per_node

        deg = adj.sum(dim=-1) + 1e-6
        deg_inv_sqrt = deg.pow(-0.5)
        adj_norm = adj * deg_inv_sqrt.unsqueeze(-1) * deg_inv_sqrt.unsqueeze(-2)

        propagated = x0
        current = x0
        alpha = alpha.view(batch_size, 1, 1)

        for step in range(1, self.propagation_steps + 1):
            current = torch.bmm(adj_norm, current)
            propagated = propagated + (alpha ** step) * current

        return propagated

    def forward(self, z):
        batch_size = z.size(0)
        h = z
        for layer in self.hidden_layers:
            h = self.relu(layer(h))
        h = self.relu(self.final_hidden(h))

        class_affinity_logits = self.class_affinity_head(h).view(batch_size, self.num_classes, self.num_classes)
        class_affinity = F.softmax(class_affinity_logits, dim=-1)

        alpha_raw = torch.tanh(self.alpha_head(h))
        alpha = alpha_raw * self.alpha_max

        class_means = self.class_feature_mean_head(h).view(batch_size, self.num_classes, self.feature_dim)
        class_logvars = self.class_feature_logvar_head(h).view(batch_size, self.num_classes, self.feature_dim)

        node_basis = self.node_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        global_context = h.unsqueeze(1).expand(-1, self.n_nodes, -1)
        node_context_input = torch.cat([global_context, node_basis], dim=-1)
        node_context = self.node_context_mlp(node_context_input)
        label_logits = self.label_mlp(node_context)
        label_probs = F.softmax(label_logits, dim=-1)

        adj_prob = self._build_adjacency(node_context, label_probs, class_affinity)
        feature_pred = self._generate_features(adj_prob, label_probs, class_means, class_logvars, alpha)

        return adj_prob, feature_pred, label_logits


class StructuredAdjacencyDecoder(nn.Module):
    """Structure-aware decoder that builds node-specific contexts from the latent."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        n_nodes: int,
        num_classes: int,
        affinity_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.num_classes = num_classes
        self.affinity_weight = affinity_weight

        self.global_mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.node_weight = nn.Parameter(torch.empty(n_nodes, latent_dim, hidden_dim))
        nn.init.normal_(self.node_weight, mean=0.0, std=0.02)
        self.node_bias = nn.Parameter(torch.zeros(n_nodes, hidden_dim))
        self.node_norm = nn.LayerNorm(hidden_dim)

        self.label_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

        self.class_affinity_head = nn.Linear(hidden_dim, num_classes * num_classes)

        self.struct_scale_raw = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.base_edge_logit = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, z: torch.Tensor):
        batch_size = z.size(0)
        global_context = self.global_mlp(z)

        # Generate node-specific representations via per-node weight tensors
        node_context = torch.einsum('bl,nlh->bnh', z, self.node_weight) + self.node_bias.unsqueeze(0)
        node_context = node_context + global_context.unsqueeze(1)
        node_context = self.node_norm(node_context)

        label_logits = self.label_head(node_context)
        label_probs = F.softmax(label_logits, dim=-1)

        class_affinity_logits = self.class_affinity_head(global_context).view(batch_size, self.num_classes, self.num_classes)
        class_affinity = F.softmax(class_affinity_logits, dim=-1)
        affinity_scores = torch.matmul(torch.matmul(label_probs, class_affinity), label_probs.transpose(1, 2))

        struct_scale = F.softplus(self.struct_scale_raw) + 1e-3
        scale = math.sqrt(node_context.size(-1))
        struct_scores = torch.matmul(node_context, node_context.transpose(1, 2)) / max(scale, 1.0)
        struct_scores = 0.5 * (struct_scores + struct_scores.transpose(1, 2))

        eps = 1e-6
        affinity_logits = torch.log(affinity_scores + eps)
        edge_logits = self.base_edge_logit + struct_scale * struct_scores + self.affinity_weight * affinity_logits

        diag_mask = torch.eye(self.n_nodes, device=edge_logits.device).unsqueeze(0)
        edge_logits = edge_logits * (1.0 - diag_mask) - 10.0 * diag_mask

        adj_prob = torch.sigmoid(edge_logits)

        return adj_prob, label_logits, class_affinity


class FeatureLabelDecoder(nn.Module):
    """Decode node features conditioned on predicted labels and adjacency."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        feature_dim: int,
        num_classes: int,
        propagation_steps: int = 3,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.propagation_steps = propagation_steps

        self.global_mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.prototype_mean_head = nn.Linear(hidden_dim, num_classes * feature_dim)
        self.prototype_logvar_head = nn.Linear(hidden_dim, num_classes * feature_dim)

        self.residual_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

        self.smoothing_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.feature_norm = nn.LayerNorm(feature_dim)

    def forward(
        self,
        z_feat: torch.Tensor,
        label_probs: torch.Tensor,
        adj_prob: torch.Tensor,
        training: bool = True,
    ):
        batch_size, n_nodes, _ = label_probs.shape
        device = z_feat.device

        global_context = self.global_mlp(z_feat)
        proto_means = self.prototype_mean_head(global_context).view(batch_size, self.num_classes, self.feature_dim)
        proto_logvars = self.prototype_logvar_head(global_context).view(batch_size, self.num_classes, self.feature_dim)

        class_vars = torch.exp(proto_logvars)
        mean_per_node = torch.matmul(label_probs, proto_means)
        var_per_node = torch.matmul(label_probs, class_vars)

        if training:
            noise = torch.randn_like(mean_per_node)
            base_features = mean_per_node + noise * torch.sqrt(var_per_node + 1e-6)
        else:
            base_features = mean_per_node

        residual = torch.tanh(self.residual_gate(global_context)).unsqueeze(1)
        base_features = base_features + residual

        base_features = self.feature_norm(base_features)

        deg = adj_prob.sum(dim=-1) + 1e-6
        deg_inv_sqrt = deg.pow(-0.5)
        adj_norm = adj_prob * deg_inv_sqrt.unsqueeze(-1) * deg_inv_sqrt.unsqueeze(-2)

        smoothing_weight = torch.sigmoid(self.smoothing_gate(global_context)).view(batch_size, 1, 1)
        smoothed = base_features
        current = base_features
        for _ in range(self.propagation_steps):
            current = torch.bmm(adj_norm, current)
            smoothed = smoothed + current
        smoothed = smoothed / float(self.propagation_steps + 1)

        features = (1.0 - smoothing_weight) * base_features + smoothing_weight * smoothed
        return features, {
            'proto_means': proto_means,
            'proto_logvars': proto_logvars,
            'smoothing_weight': smoothing_weight,
        }


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
    def __init__(self, input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim, n_layers_enc, n_layers_dec, n_max_nodes):
        super(AutoEncoder, self).__init__()
        self.n_max_nodes = n_max_nodes
        self.input_dim = input_dim
        self.encoder = GIN(input_dim, hidden_dim_enc, latent_dim, n_layers_enc)
        self.decoder = Decoder(latent_dim, hidden_dim_dec, n_layers_dec, n_max_nodes)

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
    def __init__(self, input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim, n_layers_enc, n_layers_dec, n_max_nodes):
        super(VariationalAutoEncoder, self).__init__()
        self.n_max_nodes = n_max_nodes
        self.input_dim = input_dim
        #self.encoder = GPS(input_dim, hidden_dim_enc, hidden_dim_enc, n_layers_enc)
        self.encoder = GIN(input_dim, hidden_dim_enc, hidden_dim_enc, n_layers_enc)
        #self.encoder = Powerful(input_dim=input_dim+1, num_layers=n_layers_enc, hidden=hidden_dim_enc, hidden_final=hidden_dim_enc, dropout_prob=0.0, simplified=False)
        self.fc_mu = nn.Linear(hidden_dim_enc, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim_enc, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim_dec, n_layers_dec, n_max_nodes)

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


# Variational Autoencoder with Features
class VariationalAutoEncoderWithFeatures(nn.Module):
    """VAE that reconstructs adjacency matrices, node features, and node labels jointly."""

    def __init__(
        self,
        feature_dim,
        hidden_dim_enc,
        hidden_dim_dec,
        latent_dim,
        n_layers_enc,
        n_layers_dec,
        n_max_nodes,
        num_classes,
        feature_loss_weight: float = 1.0,
        label_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_max_nodes = n_max_nodes
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.feature_loss_weight = feature_loss_weight
        self.label_loss_weight = label_loss_weight

        self.struct_latent_dim = latent_dim // 2
        self.feature_latent_dim = latent_dim - self.struct_latent_dim

        encoder_input_dim = feature_dim + num_classes
        self.encoder = GIN(encoder_input_dim, hidden_dim_enc, hidden_dim_enc, n_layers_enc)
        self.fc_mu_struct = nn.Linear(hidden_dim_enc, self.struct_latent_dim)
        self.fc_logvar_struct = nn.Linear(hidden_dim_enc, self.struct_latent_dim)
        self.fc_mu_feat = nn.Linear(hidden_dim_enc, self.feature_latent_dim)
        self.fc_logvar_feat = nn.Linear(hidden_dim_enc, self.feature_latent_dim)

        self.struct_decoder = StructuredAdjacencyDecoder(
            latent_dim=self.struct_latent_dim,
            hidden_dim=hidden_dim_dec,
            n_nodes=n_max_nodes,
            num_classes=num_classes,
        )
        self.feature_decoder = FeatureLabelDecoder(
            latent_dim=self.feature_latent_dim,
            hidden_dim=hidden_dim_dec,
            feature_dim=feature_dim,
            num_classes=num_classes,
        )

    def _prepare_encoder_batch(self, data):
        y = data.y.long()
        y_one_hot = F.one_hot(y, num_classes=self.num_classes).float()
        encoder_input = torch.cat([data.x, y_one_hot], dim=-1)
        return SimpleNamespace(x=encoder_input, edge_index=data.edge_index, batch=data.batch)

    def _encode_hidden(self, data):
        encoder_batch = self._prepare_encoder_batch(data)
        hidden = self.encoder(encoder_batch)
        mu_struct = self.fc_mu_struct(hidden)
        logvar_struct = self.fc_logvar_struct(hidden)
        mu_feat = self.fc_mu_feat(hidden)
        logvar_feat = self.fc_logvar_feat(hidden)
        return hidden, mu_struct, logvar_struct, mu_feat, logvar_feat

    def encode(self, data, eps_scale: float = 1.0):
        _, mu_struct, logvar_struct, mu_feat, logvar_feat = self._encode_hidden(data)
        if not self.training:
            eps_scale = 0.0
        z_struct = self.reparameterize(mu_struct, logvar_struct, eps_scale=eps_scale)
        z_feat = self.reparameterize(mu_feat, logvar_feat, eps_scale=eps_scale)
        return torch.cat([z_struct, z_feat], dim=-1)

    def encode_mu_logvar(self, data):
        _, mu_struct, logvar_struct, mu_feat, logvar_feat = self._encode_hidden(data)
        mu = torch.cat([mu_struct, mu_feat], dim=-1)
        logvar = torch.cat([logvar_struct, logvar_feat], dim=-1)
        return mu, logvar

    def reparameterize(self, mu, logvar, eps_scale: float = 1.0):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(std) * eps_scale
        return eps.mul(std).add_(mu)

    def _split_latent(self, z: torch.Tensor):
        z_struct = z[:, : self.struct_latent_dim]
        z_feat = z[:, self.struct_latent_dim :]
        return z_struct, z_feat

    def decode(self, z):
        z_struct, z_feat = self._split_latent(z)
        adj_prob, label_logits, class_affinity = self.struct_decoder(z_struct)
        label_probs = F.softmax(label_logits, dim=-1)
        feat_pred, feature_aux = self.feature_decoder(z_feat, label_probs, adj_prob, training=self.training)
        aux = {
            'label_probs': label_probs,
            'expected_degrees': adj_prob.sum(dim=-1),
            'class_affinity': class_affinity,
            'feature_aux': feature_aux,
        }
        return adj_prob, feat_pred, label_logits, aux

    def forward(self, data):
        z = self.encode(data)
        return self.decode(z)

    def decode_mu(self, mu):
        adj_prob, feat_pred, label_logits, aux = self.decode(mu)
        return adj_prob, feat_pred, label_logits, aux

    def compute_feature_homophily_loss(self, features, label_logits, adj_matrix, target_homophily=None):
        """
        Compute actual feature homophily metric and minimize deviation from target.
        
        Feature homophily ~ fraction of same-label edges with positively correlated features.
        
        Args:
            features: (batch, n_nodes, feature_dim) predicted features
            label_logits: (batch, n_nodes, num_classes) predicted label logits
            adj_matrix: (batch, n_nodes, n_nodes) adjacency matrix
            target_homophily: optional target value to match (default: maximize)
            
        Returns:
            loss: scalar measuring deviation from target or negative homophily
        """
        batch_size = features.shape[0]
        label_probs = F.softmax(label_logits, dim=-1)
        
        total_homophily = 0.0
        for b in range(batch_size):
            # Normalize features for cosine similarity
            feat_norm = F.normalize(features[b], p=2, dim=-1)
            similarity_matrix = torch.mm(feat_norm, feat_norm.t())
            
            # Hard label predictions for homophily
            hard_labels = torch.argmax(label_probs[b], dim=-1)
            same_label_matrix = (hard_labels.unsqueeze(0) == hard_labels.unsqueeze(1)).float()
            
            # Upper triangle edges only
            edge_mask = torch.triu((adj_matrix[b] > 0.5).float(), diagonal=1)
            
            # Count same-label edges
            same_label_edges = edge_mask * same_label_matrix
            num_same_label_edges = same_label_edges.sum()
            
            if num_same_label_edges > 0:
                # Homophily = fraction with positive feature correlation
                positive_corr = (similarity_matrix > 0).float()
                homophilic_edges = same_label_edges * positive_corr
                batch_homophily = homophilic_edges.sum() / num_same_label_edges
            else:
                batch_homophily = torch.tensor(0.0, device=features.device)
            
            total_homophily += batch_homophily
        
        avg_homophily = total_homophily / batch_size
        
        # Minimize squared error from target, or maximize if no target
        if target_homophily is not None:
            loss = (avg_homophily - target_homophily) ** 2
        else:
            loss = -avg_homophily
        
        return loss

    def _feature_homophily(self, features, label_probs, adjacency):
        feat_norm = F.normalize(features, p=2, dim=-1)
        similarity = torch.matmul(feat_norm, feat_norm.transpose(1, 2))
        similarity = torch.sigmoid(5.0 * similarity)

        same_label_prob = torch.matmul(label_probs, label_probs.transpose(1, 2))

        adjacency = adjacency.clamp(min=0.0)
        numerator = torch.sum(adjacency * same_label_prob * similarity, dim=(1, 2))
        denominator = torch.sum(adjacency * same_label_prob, dim=(1, 2)) + 1e-6
        return numerator / denominator

    def loss_function(
        self,
        data,
        beta: float = 0.05,
        homophily_weight: float = 0.1,
        target_homophily: float = None,
        mi_weight: float = 0.05,
        degree_weight: float = 0.05,
        homophily_teacher_weight: float = 0.05,
        entropy_weight: float = 0.01,
    ):
        batch_size = int(torch.max(data.batch).item()) + 1 if hasattr(data, "batch") else 1

        _, mu_struct, logvar_struct, mu_feat, logvar_feat = self._encode_hidden(data)
        z_struct = self.reparameterize(mu_struct, logvar_struct)
        z_feat = self.reparameterize(mu_feat, logvar_feat)
        z = torch.cat([z_struct, z_feat], dim=-1)
        adj_pred, feat_pred, label_logits, aux = self.decode(z)
        label_probs = aux['label_probs']

        adj_target = data.A
        feat_target = data.x.view(batch_size, self.n_max_nodes, self.feature_dim)
        label_target = data.y.long().view(batch_size, self.n_max_nodes)

        # Reweight adjacency reconstruction to counter heavy class imbalance
        eps = 1e-6
        total_elements = adj_target.numel()
        positive_edges = adj_target.sum()
        negative_edges = total_elements - positive_edges
        # Fall back to neutral weighting if a batch has no positive edges
        pos_weight_value = (negative_edges / (positive_edges + eps)).clamp(min=1.0)
        pos_weight = pos_weight_value.detach().to(adj_target.dtype)

        adj_logits = torch.logit(adj_pred.clamp(min=eps, max=1.0 - eps))
        recon_adj = F.binary_cross_entropy_with_logits(
            adj_logits,
            adj_target,
            pos_weight=pos_weight,
            reduction="sum",
        )
        recon_feat = F.mse_loss(feat_pred, feat_target, reduction="sum")
        label_loss = F.cross_entropy(
            label_logits.view(-1, self.num_classes),
            label_target.view(-1),
            reduction="sum",
        )

        # Add feature homophily loss targeting specific value
        homophily_observed = self._feature_homophily(feat_pred, label_probs, adj_pred)
        if target_homophily is not None:
            target_tensor = feat_pred.new_full(homophily_observed.shape, target_homophily)
            homophily_loss = F.mse_loss(homophily_observed, target_tensor, reduction='mean')
        else:
            homophily_loss = -torch.mean(homophily_observed)

        homophily_teacher = self._feature_homophily(feat_pred, label_probs, adj_target)
        if target_homophily is not None:
            target_tensor = feat_pred.new_full(homophily_teacher.shape, target_homophily)
            homophily_teacher_loss = F.mse_loss(homophily_teacher, target_tensor, reduction='mean')
        else:
            homophily_teacher_loss = -torch.mean(homophily_teacher)

        entropy = -(label_probs * torch.log(label_probs + 1e-6)).sum(dim=-1).mean()

        degree_pred = adj_pred.sum(dim=-1)
        degree_target = adj_target.sum(dim=-1)
        degree_loss = F.smooth_l1_loss(degree_pred, degree_target, reduction='mean')

        z_struct_norm = (z_struct - z_struct.mean(dim=0, keepdim=True))
        z_feat_norm = (z_feat - z_feat.mean(dim=0, keepdim=True))
        struct_std = z_struct_norm.std(dim=0, keepdim=True) + 1e-6
        feat_std = z_feat_norm.std(dim=0, keepdim=True) + 1e-6
        cross_corr = torch.sum((z_struct_norm / struct_std) * (z_feat_norm / feat_std), dim=1)
        mi_loss = -torch.mean(cross_corr)

        recon = recon_adj + self.feature_loss_weight * recon_feat + self.label_loss_weight * label_loss
        recon = recon + homophily_weight * homophily_loss * batch_size
        recon = recon + homophily_teacher_weight * homophily_teacher_loss * batch_size
        recon = recon + degree_weight * degree_loss * batch_size
        recon = recon + mi_weight * mi_loss * batch_size
        recon = recon + entropy_weight * entropy * batch_size

        kld_struct = -0.5 * torch.sum(1 + logvar_struct - mu_struct.pow(2) - logvar_struct.exp())
        kld_feat = -0.5 * torch.sum(1 + logvar_feat - mu_feat.pow(2) - logvar_feat.exp())
        kld = kld_struct + kld_feat
        loss = recon + beta * kld

        metrics = {
            'homophily_observed': homophily_observed.detach().cpu(),
            'homophily_loss': float(homophily_loss.detach().cpu()),
            'homophily_teacher_loss': float(homophily_teacher_loss.detach().cpu()),
            'degree_loss': float(degree_loss.detach().cpu()),
            'mi_loss': float(mi_loss.detach().cpu()),
            'entropy': float(entropy.detach().cpu()),
        }

        return loss, recon, kld, metrics
