#!/usr/bin/env python3
"""
Synthetic Graph Generator

Generates synthetic graphs with controllable homophily levels (label, structural, feature)
and saves them as PyTorch Geometric datasets for GNN training.

Usage:
    python synthgraphgenerator.py --homophily_type label --min_hom 0.1 --max_hom 0.9 --n_graphs 5
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset as PyGDataset
from torch_geometric.utils import to_undirected, remove_self_loops
from tqdm import tqdm
import pickle
import csv
import datetime
from torch.distributions import multivariate_normal
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import scipy as scipy_main
import networkx as nx
import math
import time
import warnings
warnings.filterwarnings('ignore')


# Utility functions (adapted from util_funcs.py to avoid DGL dependency)
def spectral_radius_sp_matrix(edge_index, values, num_nodes):
    """
    Compute spectral radius for a sparse matrix using scipy
    Input
        edge_index: edge set from src to dst (2,num_edges) or (num_edges, 2)
        values: weight for the edge (num_edges)
    Output
        Spectral radius of the matrix
    """
    if edge_index.shape[0] != 2:
        edge_index = edge_index.t()
    
    adj = sp.coo_matrix((values.numpy(), (edge_index[0].numpy(), edge_index[1].numpy())), 
                       shape=(num_nodes, num_nodes))
    try:
        eigenvalues, _ = sla.eigs(adj, k=1, which='LM')
        return float(np.abs(eigenvalues[0]).real)
    except:
        # Fallback to power iteration estimate
        return float(torch.max(torch.bincount(edge_index[0], minlength=num_nodes)))


def sym_matrix(A, device):
    """Make matrix symmetric by taking upper triangular and mirroring"""
    n = A.shape[0]
    indices = torch.triu_indices(n, n).to(device)
    matrix = torch.zeros(n, n).to(device)
    matrix[indices[0], indices[1]] = A[indices[0], indices[1]]
    matrix = matrix.t()
    matrix[indices[0], indices[1]] = A[indices[0], indices[1]]
    return matrix


def handle_nan(x):
    """Handle NaN values by replacing with -100"""
    if math.isnan(x):
        return float(-100)
    return x


def calculate_graph_stats(G):
    """Calculate the 15 standard graph properties used in main.py"""
    stats = []
    
    # Number of nodes
    num_nodes = handle_nan(float(G.number_of_nodes()))
    stats.append(num_nodes)
    
    # Number of edges
    num_edges = handle_nan(float(G.number_of_edges()))
    stats.append(num_edges)
    
    # Density
    density = handle_nan(float(nx.density(G)))
    stats.append(density)
    
    # Degree statistics
    degrees = [deg for node, deg in G.degree()]
    if len(degrees) > 0:
        max_degree = handle_nan(float(max(degrees)))
        min_degree = handle_nan(float(min(degrees)))
        avg_degree = handle_nan(float(sum(degrees) / len(degrees)))
    else:
        max_degree = min_degree = avg_degree = 0.0
    stats.append(max_degree)
    stats.append(min_degree)
    stats.append(avg_degree)
    
    # Assortativity coefficient
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            assortativity = handle_nan(float(nx.degree_assortativity_coefficient(G)))
    except:
        assortativity = 0.0
    stats.append(assortativity)
    
    # Number of triangles
    triangles = nx.triangles(G)
    num_triangles = handle_nan(float(sum(triangles.values()) // 3))
    stats.append(num_triangles)
    
    # Average number of triangles formed by an edge
    if num_edges > 0:
        avg_triangles = handle_nan(float(sum(triangles.values()) / num_edges))
    else:
        avg_triangles = 0.0
    stats.append(avg_triangles)
    
    # Maximum number of triangles formed by an edge
    if len(triangles) > 0:
        max_triangles_per_edge = handle_nan(float(max(triangles.values())))
    else:
        max_triangles_per_edge = 0.0
    stats.append(max_triangles_per_edge)
    
    # Average local clustering coefficient
    try:
        avg_clustering_coefficient = handle_nan(float(nx.average_clustering(G)))
    except:
        avg_clustering_coefficient = 0.0
    stats.append(avg_clustering_coefficient)
    
    # Global clustering coefficient
    try:
        global_clustering_coefficient = handle_nan(float(nx.transitivity(G)))
    except:
        global_clustering_coefficient = 0.0
    stats.append(global_clustering_coefficient)
    
    # Maximum k-core
    try:
        core_numbers = nx.core_number(G)
        if len(core_numbers) > 0:
            max_k_core = handle_nan(float(max(core_numbers.values())))
        else:
            max_k_core = 0.0
    except:
        max_k_core = 0.0
    stats.append(max_k_core)
    
    # Calculate communities (using connected components as approximation)
    try:
        # Try to use NetworkX's built-in community detection if available
        try:
            import networkx.algorithms.community as nxcom
            communities = nxcom.greedy_modularity_communities(G)
            n_communities = handle_nan(float(len(communities)))
        except:
            # Fallback to connected components
            connected_components = list(nx.connected_components(G))
            n_communities = handle_nan(float(len(connected_components)))
    except:
        n_communities = 1.0
    stats.append(n_communities)
    
    # Calculate diameter
    try:
        connected_components = list(nx.connected_components(G))
        diameter = float(0)
        for component in connected_components:
            subgraph = G.subgraph(component)
            if subgraph.number_of_nodes() > 1:
                component_diameter = nx.diameter(subgraph)
                diameter = handle_nan(float(max(diameter, component_diameter)))
    except:
        diameter = 0.0
    stats.append(diameter)
    
    return stats


def construct_nx_from_adj(adj):
    """Convert adjacency matrix to NetworkX graph"""
    G = nx.from_numpy_array(adj, create_using=nx.Graph)
    to_remove = []
    for node in G.nodes():
        if G.degree(node) == 0:
            to_remove.append(node)
    G.remove_nodes_from(to_remove)
    return G


def random_graph_with_feature(num_node, num_class, node_degree, feat_dim,
                             label_homophily, structural_homophily, feature_homophily, 
                             seed, device):
    """
    Generate random graphs with features sampled from labels and neighbors
    Uses iterative search to achieve target feature homophily level.
    
    Parameters:
    num_node -- int. Number of total node number
    num_class -- int. Number of class number
    node_degree -- string of "lowest highest". the node degrees follow a uniform distribution
    feat_dim -- feature dimension
    label_homophily -- float, ranging from [0,1]
    structural_homophily -- float, ranging from [0,1]
    feature_homophily -- float, ranging from (-1,1)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Prepare data format
    num_node = int(num_node)
    num_class = int(num_class)
    feat_dim = int(feat_dim)
    node_degree = [int(d) for d in node_degree.split(' ')]
    
    # Generate graph structure (same for all feature parameter attempts)
    degrees = torch.randint(node_degree[0], node_degree[1] + 1, (num_node,)).to(device)
    D = torch.diag(degrees.float())
    
    # Labels Nx1 & one-hot labels NxC
    Y = torch.randint(0, num_class, (num_node,)).to(device)
    Z = F.one_hot(Y, num_classes=num_class).float()
    
    # Sample matrix CxC for label homophily
    S = torch.ones(num_class, num_class).to(device) * (1 - label_homophily) / (num_class - 1)
    S.fill_diagonal_(label_homophily)
    
    # Neighbor distribution with structural homophily control
    Nei_dist = Z @ S + torch.normal(
        mean=0, 
        std=np.sqrt(np.power(1 - structural_homophily, 2) / (num_class - 1)),
        size=(num_node, num_class)
    ).to(device)
    
    # Ensure non-negative probabilities
    Nei_dist = torch.clamp(Nei_dist, min=0)
    
    # Bernoulli sampling for adjacency matrix
    A_p = (num_class / num_node) * torch.sqrt(D) @ Nei_dist @ Z.t() @ torch.sqrt(D)
    A_p = torch.clamp(A_p, min=0, max=1)
    
    # Sample adjacency matrix
    A = torch.bernoulli(A_p)
    A = sym_matrix(A, device)  # Make symmetric
    
    # Remove self-loops
    A.fill_diagonal_(0)
    
    # Get edge indices
    edge_indices = A.nonzero().t()
    
    if len(edge_indices) > 0 and abs(feature_homophily) > 1e-6:
        # Use iterative search to find the generation parameter that achieves target homophily
        best_X = None
        best_diff = float('inf')
        
        # Search range for generation parameters
        search_range = np.linspace(-1.0, 1.0, 21)  # Search from -1 to 1
        
        for gen_param in search_range:
            # Generate features with this parameter
            X_candidate = generate_features_with_param(
                num_node, num_class, feat_dim, Y, Z, A, edge_indices, 
                gen_param, device, seed
            )
            
            # Create temporary data object to measure homophily
            temp_data = Data(x=X_candidate.cpu(), edge_index=edge_indices.cpu(), y=Y.cpu())
            
            # Measure actual feature homophily
            actual_homophily = measure_feature_homophily_spectral(temp_data, num_class, feat_dim)
            diff = abs(actual_homophily - feature_homophily)
            
            if diff < best_diff:
                best_diff = diff
                best_X = X_candidate
                
            # If we're close enough, stop early
            if diff < 0.05:  # Within 5% of target
                break
        
        X = best_X if best_X is not None else generate_features_with_param(
            num_node, num_class, feat_dim, Y, Z, A, edge_indices, 
            feature_homophily, device, seed
        )
    else:
        # No edges or zero target homophily - just generate class-based features
        X = generate_features_with_param(
            num_node, num_class, feat_dim, Y, Z, A, edge_indices, 
            0.0, device, seed
        )
    
    return edge_indices.cpu(), X.cpu(), Y.cpu()


def generate_features_with_param(num_node, num_class, feat_dim, Y, Z, A, edge_indices, 
                                generation_param, device, seed):
    """
    Generate node features using a specific generation parameter.
    Uses the same spectral method as datasets.py.
    
    Args:
        num_node: Number of nodes
        num_class: Number of classes
        feat_dim: Feature dimension
        Y: Node labels
        Z: One-hot encoded labels
        A: Adjacency matrix
        edge_indices: Edge indices
        generation_param: Feature homophily generation parameter
        device: Device to use
        seed: Random seed
        
    Returns:
        torch.Tensor: Generated node features
    """
    # Generate initial class-based features (same as datasets.py)
    X0 = torch.zeros(num_node, feat_dim).to(device)
    jitter = 1e-6
    for d in range(feat_dim):
        C_mean = torch.rand(num_class).to(device)
        C_vars = torch.rand(num_class).to(device)
        X_mean = Z @ C_mean
        X_vars = Z @ C_vars
        cov_diag = torch.clamp(X_vars, min=jitter)
        covariance = torch.diag(cov_diag) + torch.eye(num_node, device=device) * jitter
        X0[:, d] = multivariate_normal.MultivariateNormal(
            X_mean, covariance
        ).sample()
    
    # Apply feature homophily using spectral transformation (same as datasets.py)
    if len(edge_indices) > 0 and abs(generation_param) > 1e-6:
        try:
            spectral_radius = spectral_radius_sp_matrix(edge_indices, torch.ones(edge_indices.shape[1]), num_node)
            if spectral_radius > 0:
                alpha = generation_param / spectral_radius
                I = torch.eye(num_node).to(device)
                # Use torch.matrix_power with -1 for matrix inverse (same as datasets.py)
                nei_info = torch.matrix_power(I - alpha * A, -1)
                X = nei_info @ X0
            else:
                X = X0
        except:
            # Fallback if matrix inversion fails
            X = X0
    else:
        X = X0
        
    return X


def measure_feature_homophily_spectral(data, num_classes, feat_dim):
    """
    Measure feature homophily using the same spectral method as datasets.py.
    Simplified version for faster computation during iterative search.
    
    Args:
        data: PyTorch Geometric Data object
        num_classes: Number of classes
        feat_dim: Feature dimension
        
    Returns:
        float: Measured feature homophily value
    """
    edge_index = data.edge_index
    features = data.x
    labels = data.y
    
    if edge_index.shape[1] == 0:
        return 0.0
    
    num_nodes = features.shape[0]
    device = 'cpu'  # Keep computations on CPU for stability
    
    # Convert to required format (same as datasets.py)
    edges = edge_index.long()
    A = torch.sparse_coo_tensor(edges, torch.ones(edges.shape[1]), 
                               (num_nodes, num_nodes)).to(device)
    I = torch.sparse_coo_tensor(torch.arange(num_nodes).repeat(2,1), 
                               torch.ones(num_nodes), 
                               (num_nodes, num_nodes)).to(device)
    X = features.to(device)
    Y = F.one_hot(labels.long(), num_classes=num_classes).float().to(device)
    
    try:
        # Calculate spectral radius
        spectral_radius = spectral_radius_sp_matrix(edges, torch.ones(edges.shape[1]), num_nodes)
        
        if spectral_radius == 0 or not np.isfinite(spectral_radius):
            return 0.0
        
        # Search over h_F values from -1 to 1 with step 0.1 (coarser for speed)
        h_F_lst = 0.1 * torch.arange(-10, 11)  # Range from [-1,1] with step=0.1
        v_lst = []
        
        for h_F in h_F_lst:
            try:
                w = h_F / spectral_radius
                X0 = torch.sparse.mm(I - w * A, X)
                X0_cls = ((X0.t() @ Y) / Y.sum(dim=0).repeat(feat_dim, 1))
                X0_cls = (X0_cls @ Y.t()).t()
                v = torch.abs(X0_cls - X0).sum(dim=0)
                v_lst.append(v)
            except:
                # If computation fails, use a large penalty
                v_lst.append(torch.ones(feat_dim) * 1e6)
        
        v_lst = torch.stack(v_lst)
        
        # Find the h_F that minimizes the total reconstruction error
        h_F_graph = h_F_lst[int(torch.argmin(v_lst.sum(dim=1)))]
        
        return float(h_F_graph)
    except:
        return 0.0


class SyntheticGraphDataset(PyGDataset):
    """PyTorch Geometric Dataset for synthetic graphs"""
    
    def __init__(self, root, data_list):
        self.data_list = data_list
        super().__init__(root)
    
    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        return self.data_list[idx]


class SyntheticGraphGenerator:
    """
    Generator for synthetic graphs with controllable homophily
    
    Note on node features:
    - We generate graphs with semantic node features (class-based, used for homophily)
    - But we convert to spectral features (degree + eigenvectors) for main.py compatibility
    - The semantic features are stored in 'raw_node_features' for analysis
    - The spectral features in 'x' are used by the autoencoder/diffusion model
    """
    
    def __init__(self, 
                 num_nodes=500, 
                 num_classes=3, 
                 node_degree_range="3 10", 
                 feat_dim=32,
                 train_ratio=0.6,
                 val_ratio=0.2,
                 test_ratio=0.2,
                 seed=42,
                 n_max_nodes=None,
                 spectral_emb_dim=10,
                 main_py_compatible=True):
        """
        Initialize the synthetic graph generator
        
        Args:
            num_nodes: Number of nodes in the graph
            num_classes: Number of node classes
            node_degree_range: String "min_degree max_degree" for uniform degree sampling
            feat_dim: Feature dimension
            train_ratio: Training set ratio
            val_ratio: Validation set ratio  
            test_ratio: Test set ratio
            seed: Random seed
        """
        self.num_nodes = num_nodes
        self.num_classes = num_classes
        self.node_degree_range = node_degree_range
        self.feat_dim = feat_dim
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.n_max_nodes = n_max_nodes if n_max_nodes is not None else num_nodes
        self.spectral_emb_dim = spectral_emb_dim
        self.main_py_compatible = main_py_compatible
        
        # Validate ratios
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    def generate_single_graph(self, label_hom=0.5, structural_hom=0.5, feature_hom=0.0, graph_idx=0,
                             n_max_nodes=None, spectral_emb_dim=None):
        """
        Generate a single synthetic graph with specified homophily levels in main.py compatible format
        
        Args:
            label_hom: Label homophily level [0, 1]
            structural_hom: Structural homophily level [0, 1]  
            feature_hom: Feature homophily level [-1, 1]
            graph_idx: Index for this graph (for reproducibility)
            n_max_nodes: Maximum nodes for padding (default: self.num_nodes)
            spectral_emb_dim: Spectral embedding dimension (default: 10)
            
        Returns:
            PyTorch Geometric Data object compatible with main.py
        """
        if n_max_nodes is None:
            n_max_nodes = self.num_nodes
        if spectral_emb_dim is None:
            spectral_emb_dim = min(10, self.num_nodes - 1)
            
        # Set seed for reproducibility
        current_seed = self.seed + graph_idx * 1000
        torch.manual_seed(current_seed)
        np.random.seed(current_seed)
        
        device = torch.device('cpu')  # Generate on CPU for consistency
        
        # Generate graph with semantic features that contain homophily information
        edge_index, node_features, labels = random_graph_with_feature(
            num_node=self.num_nodes,
            num_class=self.num_classes,
            node_degree=self.node_degree_range,
            feat_dim=self.feat_dim,
            label_homophily=label_hom,
            structural_homophily=structural_hom,
            feature_homophily=feature_hom,
            seed=current_seed,
            device=device
        )
        
        # Convert edge_index to proper format (2 x num_edges)
        if edge_index.shape[0] != 2:
            edge_index = edge_index.t()
        
        # Make graph undirected
        edge_index = to_undirected(edge_index)
        
        # Create adjacency matrix for NetworkX and statistics
        adj_matrix = torch.zeros(self.num_nodes, self.num_nodes)
        adj_matrix[edge_index[0], edge_index[1]] = 1.0
        
        # Convert to NetworkX for statistics calculation
        G = construct_nx_from_adj(adj_matrix.numpy())
        
        # Calculate the 15 standard graph properties
        graph_stats = calculate_graph_stats(G)
        
        # Add homophily properties as additional conditioning features
        # Measure actual homophily levels using semantic features
        temp_data = Data(x=node_features.float(), edge_index=edge_index.long(), y=labels.long())
        actual_label_hom = self.measure_label_homophily(temp_data)
        actual_structural_hom = self.measure_structural_homophily(temp_data) 
        actual_feature_hom = self.measure_feature_homophily(temp_data)
        
        # Extend stats with homophily properties (18 total features now)
        extended_stats = graph_stats + [actual_label_hom, actual_structural_hom, actual_feature_hom]
        stats_tensor = torch.FloatTensor(extended_stats).unsqueeze(0)  # Shape: (1, 18)
        
        # Use semantic node features directly (these contain the homophily information)
        x = node_features  # Shape: (num_nodes, feat_dim)
        
        # Pad adjacency matrix to n_max_nodes like in main.py
        size_diff = n_max_nodes - self.num_nodes
        adj_padded = F.pad(adj_matrix, [0, size_diff, 0, size_diff])
        adj_padded = adj_padded.unsqueeze(0)  # Shape: (1, n_max_nodes, n_max_nodes)
        
        # Create train/val/test masks
        num_nodes = len(labels)
        indices = torch.randperm(num_nodes)
        
        train_size = int(self.train_ratio * num_nodes)
        val_size = int(self.val_ratio * num_nodes)
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size + val_size]] = True
        test_mask[indices[train_size + val_size:]] = True
        
        # Create PyTorch Geometric Data object compatible with main.py
        data = Data(
            x=x.float(),                          # Spectral node features
            edge_index=edge_index.long(),         # Graph edges
            A=adj_padded.float(),                 # Padded adjacency matrix
            stats=stats_tensor.float(),           # Extended graph properties (18 features)
            y=labels.long(),                      # Node labels 
            train_mask=train_mask,                # Training mask
            val_mask=val_mask,                    # Validation mask
            test_mask=test_mask,                  # Test mask
            num_classes=self.num_classes,         # Number of classes
            graph_class='synthetic',              # Graph type
            class_label=0,                        # Graph type ID
            # Store homophily parameters as metadata
            label_homophily=torch.tensor(label_hom),
            structural_homophily=torch.tensor(structural_hom),
            feature_homophily=torch.tensor(feature_hom),
            # Store raw semantic node features for analysis
            raw_node_features=node_features.float()
        )
        
        return data
    
    def measure_label_homophily(self, data):
        """Measure actual label homophily of a graph"""
        edge_index = data.edge_index
        labels = data.y
        
        # Remove self loops for measurement
        edge_index_clean = remove_self_loops(edge_index)[0]
        
        src_labels = labels[edge_index_clean[0]]
        tgt_labels = labels[edge_index_clean[1]]
        
        # Calculate fraction of edges connecting same-class nodes
        same_class_edges = (src_labels == tgt_labels).float()
        label_homophily = torch.mean(same_class_edges).item()
        
        return label_homophily
    
    def measure_structural_homophily(self, data):
        """Measure actual structural homophily of a graph"""
        edge_index = data.edge_index
        labels = data.y.long()
        num_nodes = data.x.shape[0]
        
        # Remove self loops
        edge_index_clean = remove_self_loops(edge_index)[0]
        
        try:
            # Create dense adjacency matrix for simplicity
            adj = torch.zeros(num_nodes, num_nodes)
            adj[edge_index_clean[0], edge_index_clean[1]] = 1
            
            # One-hot encode labels
            Y = F.one_hot(labels, num_classes=self.num_classes).float()
            
            # Calculate neighborhood class distributions
            dist = adj @ Y  # Matrix multiplication gives neighbor class counts
            
            # Normalize to get distributions
            row_sums = dist.sum(dim=1, keepdim=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            dist = dist / row_sums
            
            def get_max_std(c):
                return np.sqrt((1-1/c)/c) if c > 1 else 0
            
            h_N_list = []
            for c in range(self.num_classes):
                c_nodes = (labels == c).nonzero().flatten()
                if len(c_nodes) > 1:
                    c_dist = dist[c_nodes]
                    if c_dist.shape[1] > 1:
                        std_list = c_dist.std(dim=0)
                        std_max = get_max_std(c_dist.shape[1])
                        if std_max > 0:
                            h_N_item = (1 - std_list/std_max).mean()
                            h_N_list.append(h_N_item.item())
            
            if h_N_list:
                return np.mean(h_N_list)
            else:
                return 0.5
                
        except Exception:
            # Fallback to simple measurement if complex calculation fails
            return 0.5
    
    def measure_feature_homophily(self, data, max_iterations=50):
        """Measure actual feature homophily of a graph (simplified version)"""
        try:
            edge_index = data.edge_index
            x = data.x
            labels = data.y.long()
            num_nodes = x.shape[0]
            
            # Remove self loops
            edge_index_clean = remove_self_loops(edge_index)[0]
            
            # Create dense adjacency matrix for simplicity
            A = torch.zeros(num_nodes, num_nodes)
            A[edge_index_clean[0], edge_index_clean[1]] = 1.0
            I = torch.eye(num_nodes)
            
            # Calculate spectral radius (approximate)
            try:
                spectral_radius = spectral_radius_sp_matrix(edge_index_clean.cpu(), 
                                                          torch.ones(edge_index_clean.shape[1]), 
                                                          num_nodes)
            except:
                # Fallback: use degree-based approximation
                degrees = torch.bincount(edge_index_clean[0], minlength=num_nodes)
                spectral_radius = float(degrees.max())
            
            Y = F.one_hot(labels, num_classes=self.num_classes).float()
            
            # Test a limited range of feature homophily values
            h_F_range = torch.linspace(-0.9, 0.9, max_iterations)
            best_h_F = 0.0
            min_error = float('inf')
            
            for h_F in h_F_range:
                try:
                    w = h_F / spectral_radius if spectral_radius > 0 else 0
                    if abs(w) < 0.95:  # Ensure convergence (less restrictive)
                        # Use matrix multiplication instead of sparse operations
                        X0 = torch.mm(I - w * A, x)
                        
                        # Calculate class means
                        class_sums = Y.t() @ X0
                        class_counts = Y.sum(dim=0, keepdim=True).t()
                        class_counts[class_counts == 0] = 1
                        X0_cls_mean = class_sums / class_counts
                        
                        # Map back to nodes
                        X0_cls = Y @ X0_cls_mean
                        
                        # Calculate error
                        error = torch.abs(X0_cls - X0).sum()
                        
                        if error < min_error:
                            min_error = error
                            best_h_F = h_F.item()
                except:
                    continue
            
            return best_h_F
            
        except Exception:
            # Return 0 if measurement fails
            return 0.0
    
    def generate_graphs_by_homophily(self, homophily_type, min_hom, max_hom, n_graphs, 
                                   fixed_hom_values=None):
        """
        Generate graphs varying one type of homophily while keeping others at default
        
        Args:
            homophily_type: 'label', 'structural', or 'feature'
            min_hom: Minimum homophily value
            max_hom: Maximum homophily value
            n_graphs: Number of graphs per homophily level
            fixed_hom_values: Dict with fixed values for other homophily types
            
        Returns:
            List of PyTorch Geometric Data objects
        """
        if fixed_hom_values is None:
            fixed_hom_values = {
                'label': 0.5,
                'structural': 0.5, 
                'feature': 0.0
            }
        
        # Generate homophily levels with 0.1 increments
        if min_hom == max_hom:
            hom_levels = [min_hom]
        else:
            hom_levels = np.arange(min_hom, max_hom + 0.05, 0.1)  # +0.05 to include max_hom
            hom_levels = np.round(hom_levels, 1)  # Round to avoid floating point issues
        
        print(f"Generating graphs for {homophily_type} homophily levels: {hom_levels}")
        print(f"Fixed homophily values: {fixed_hom_values}")
        
        all_graphs = []
        graph_metadata = []
        graph_logs = []
        
        for hom_level in tqdm(hom_levels, desc=f"Homophily levels"):
            for graph_idx in range(n_graphs):
                # Set homophily parameters
                label_hom = fixed_hom_values['label']
                structural_hom = fixed_hom_values['structural']
                feature_hom = fixed_hom_values['feature']
                
                # Override the specified homophily type
                if homophily_type == 'label':
                    label_hom = hom_level
                elif homophily_type == 'structural':
                    structural_hom = hom_level
                elif homophily_type == 'feature':
                    feature_hom = hom_level
                else:
                    raise ValueError(f"Unknown homophily type: {homophily_type}")
                
                # Generate graph
                data = self.generate_single_graph(
                    label_hom=label_hom,
                    structural_hom=structural_hom,
                    feature_hom=feature_hom,
                    graph_idx=len(all_graphs),
                    n_max_nodes=self.n_max_nodes,
                    spectral_emb_dim=self.spectral_emb_dim
                )
                
                # Measure actual homophily levels
                actual_label_hom = self.measure_label_homophily(data)
                actual_structural_hom = self.measure_structural_homophily(data)
                actual_feature_hom = self.measure_feature_homophily(data)
                
                # Count actual nodes and edges
                actual_nodes = data.x.shape[0]
                actual_edges = data.edge_index.shape[1]
                
                all_graphs.append(data)
                
                # Store metadata
                metadata = {
                    'graph_idx': len(all_graphs) - 1,
                    'homophily_type': homophily_type,
                    'homophily_level': float(hom_level),
                    'label_homophily': float(label_hom),
                    'structural_homophily': float(structural_hom),
                    'feature_homophily': float(feature_hom),
                    'num_nodes': self.num_nodes,
                    'num_classes': self.num_classes,
                    'feat_dim': self.feat_dim
                }
                graph_metadata.append(metadata)
                
                # Store log entry with actual measurements
                log_entry = {
                    'graph_idx': len(all_graphs) - 1,
                    'timestamp': datetime.datetime.now().isoformat(),
                    'homophily_type': homophily_type,
                    'target_homophily_level': float(hom_level),
                    'target_label_hom': float(label_hom),
                    'target_structural_hom': float(structural_hom),
                    'target_feature_hom': float(feature_hom),
                    'actual_nodes': actual_nodes,
                    'actual_edges': actual_edges,
                    'actual_label_hom': actual_label_hom,
                    'actual_structural_hom': actual_structural_hom,
                    'actual_feature_hom': actual_feature_hom,
                    'avg_degree': actual_edges / actual_nodes if actual_nodes > 0 else 0,
                    'edge_density': actual_edges / (actual_nodes * (actual_nodes - 1)) if actual_nodes > 1 else 0,
                    'stats_dimensions': data.stats.shape[1],  # Number of conditioning features
                    'main_py_compatible': self.main_py_compatible
                }
                graph_logs.append(log_entry)
        
        print(f"Generated {len(all_graphs)} graphs total")
        return all_graphs, graph_metadata, graph_logs
    
    def save_dataset(self, graphs, metadata, output_dir, dataset_name, logs=None):
        """
        Save graphs as PyTorch Geometric dataset
        
        Args:
            graphs: List of PyTorch Geometric Data objects
            metadata: List of metadata dictionaries
            output_dir: Output directory path
            dataset_name: Name for the dataset
            logs: List of log dictionaries with actual measurements
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save graphs as pickle files (PyG format)
        dataset_path = os.path.join(output_dir, f"{dataset_name}_graphs.pkl")
        with open(dataset_path, 'wb') as f:
            pickle.dump(graphs, f)
        
        # Save metadata
        metadata_path = os.path.join(output_dir, f"{dataset_name}_metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        # Save log file with actual measurements
        if logs:
            log_path = os.path.join(output_dir, f"{dataset_name}_log.csv")
            with open(log_path, 'w', newline='') as f:
                if logs:
                    writer = csv.DictWriter(f, fieldnames=logs[0].keys())
                    writer.writeheader()
                    writer.writerows(logs)
        
        # Save a summary file
        summary_path = os.path.join(output_dir, f"{dataset_name}_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Synthetic Graph Dataset: {dataset_name}\n")
            f.write(f"Total graphs: {len(graphs)}\n")
            f.write(f"Nodes per graph: {self.num_nodes}\n")
            f.write(f"Max nodes (padding): {self.n_max_nodes}\n")
            f.write(f"Classes: {self.num_classes}\n")
            f.write(f"Feature dimension: {self.feat_dim}\n")
            f.write(f"Spectral embedding dim: {self.spectral_emb_dim}\n")
            f.write(f"Node degree range: {self.node_degree_range}\n")
            f.write(f"Train/Val/Test ratios: {self.train_ratio}/{self.val_ratio}/{self.test_ratio}\n")
            f.write(f"Main.py compatible: {self.main_py_compatible}\n")
            
            if metadata:
                homophily_type = metadata[0]['homophily_type']
                levels = sorted(list(set([m['homophily_level'] for m in metadata])))
                n_per_level = len([m for m in metadata if m['homophily_level'] == levels[0]])
                
                f.write(f"\nHomophily type varied: {homophily_type}\n")
                f.write(f"Homophily levels: {levels}\n")
                f.write(f"Graphs per level: {n_per_level}\n")
                
                if logs:
                    f.write(f"\nLog file contains actual measurements:\n")
                    f.write(f"- Actual node/edge counts\n")
                    f.write(f"- Measured homophily levels (all types)\n")
                    f.write(f"- Graph density and average degree\n")
                    f.write(f"- Generation timestamps\n")
                    f.write(f"\nGraph properties for conditioning (18 features):\n")
                    f.write(f"Standard properties (15): nodes, edges, density, degree stats,\n")
                    f.write(f"  assortativity, triangles, clustering, k-core, communities, diameter\n")
                    f.write(f"Homophily properties (3): label_hom, structural_hom, feature_hom\n")
        
        print(f"Dataset saved to: {output_dir}")
        print(f"- Graphs: {dataset_path}")
        print(f"- Metadata: {metadata_path}")
        if logs:
            log_path = os.path.join(output_dir, f"{dataset_name}_log.csv")
            print(f"- Log: {log_path}")
        print(f"- Summary: {summary_path}")


def load_synthetic_dataset(dataset_path):
    """
    Load a previously saved synthetic dataset
    
    Args:
        dataset_path: Path to the dataset pickle file
        
    Returns:
        List of PyTorch Geometric Data objects
    """
    with open(dataset_path, 'rb') as f:
        graphs = pickle.load(f)
    return graphs


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic graphs with controllable homophily')
    
    # Homophily parameters
    parser.add_argument('--homophily_type', type=str, required=True, 
                       choices=['label', 'structural', 'feature'],
                       help='Type of homophily to vary')
    parser.add_argument('--min_hom', type=float, required=True,
                       help='Minimum homophily value')
    parser.add_argument('--max_hom', type=float, required=True,  
                       help='Maximum homophily value')
    parser.add_argument('--n_graphs', type=int, default=5,
                       help='Number of graphs per homophily level')
    
    # Graph parameters
    parser.add_argument('--num_nodes', type=int, default=100,
                       help='Number of nodes per graph')
    parser.add_argument('--num_classes', type=int, default=3,
                       help='Number of node classes')
    parser.add_argument('--node_degree_range', type=str, default="3 10",
                       help='Node degree range as "min max"')
    parser.add_argument('--feat_dim', type=int, default=32,
                       help='Feature dimension')
    
    # Fixed homophily values for other types
    parser.add_argument('--fixed_label_hom', type=float, default=0.7,
                       help='Fixed label homophily when varying other types')
    parser.add_argument('--fixed_structural_hom', type=float, default=0.7,
                       help='Fixed structural homophily when varying other types')
    parser.add_argument('--fixed_feature_hom', type=float, default=0.5,
                       help='Fixed feature homophily when varying other types')
    
    # Data splits
    parser.add_argument('--train_ratio', type=float, default=0.6,
                       help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                       help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                       help='Test set ratio')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./data',
                       help='Output directory for generated graphs')
    parser.add_argument('--dataset_name', type=str, default=None,
                       help='Name for the dataset (auto-generated if not provided)')
    
    # Main.py compatibility parameters
    parser.add_argument('--n_max_nodes', type=int, default=100,
                       help='Maximum nodes for padding (default: same as num_nodes)')
    parser.add_argument('--spectral_emb_dim', type=int, default=10,
                       help='Spectral embedding dimension for node features')
    parser.add_argument('--main_py_compatible', action='store_true', default=True,
                       help='Generate data compatible with main.py format')
    
    # Other parameters  
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Validate homophily ranges
    if args.homophily_type in ['label', 'structural']:
        if not (0 <= args.min_hom <= 1 and 0 <= args.max_hom <= 1):
            raise ValueError(f"For {args.homophily_type} homophily, values must be in [0, 1]")
    elif args.homophily_type == 'feature':
        if not (-1 <= args.min_hom <= 1 and -1 <= args.max_hom <= 1):
            raise ValueError("For feature homophily, values must be in [-1, 1]")
    
    if args.min_hom > args.max_hom:
        raise ValueError("min_hom must be <= max_hom")
    
    # Generate dataset name if not provided
    if args.dataset_name is None:
        args.dataset_name = f"synth_{args.homophily_type}_hom_{args.min_hom}_{args.max_hom}_n{args.n_graphs}"
    
    print(f"Generating synthetic graphs...")
    print(f"Homophily type: {args.homophily_type}")
    print(f"Homophily range: [{args.min_hom}, {args.max_hom}]")
    print(f"Graphs per level: {args.n_graphs}")
    print(f"Graph size: {args.num_nodes} nodes, {args.num_classes} classes")
    
    # Initialize generator
    generator = SyntheticGraphGenerator(
        num_nodes=args.num_nodes,
        num_classes=args.num_classes,
        node_degree_range=args.node_degree_range,
        feat_dim=args.feat_dim,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        n_max_nodes=args.n_max_nodes,
        spectral_emb_dim=args.spectral_emb_dim,
        main_py_compatible=args.main_py_compatible
    )
    
    # Set fixed homophily values
    fixed_hom_values = {
        'label': args.fixed_label_hom,
        'structural': args.fixed_structural_hom,
        'feature': args.fixed_feature_hom
    }
    
    # Generate graphs
    graphs, metadata, logs = generator.generate_graphs_by_homophily(
        homophily_type=args.homophily_type,
        min_hom=args.min_hom,
        max_hom=args.max_hom,
        n_graphs=args.n_graphs,
        fixed_hom_values=fixed_hom_values
    )
    
    # Save dataset
    generator.save_dataset(graphs, metadata, args.output_dir, args.dataset_name, logs)
    
    print(f"Successfully generated {len(graphs)} synthetic graphs!")


if __name__ == "__main__":
    main()
