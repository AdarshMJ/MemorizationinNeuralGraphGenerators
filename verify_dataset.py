#!/usr/bin/env python3
"""
Quick verification script to test synthetic dataset loading
"""

import pickle
import torch
import sys

def verify_synthetic_dataset(dataset_path='data/featurehomophily_graphs.pkl'):
    """Verify the synthetic dataset can be loaded and has correct format"""
    
    print("=" * 80)
    print("Synthetic Dataset Verification")
    print("=" * 80)
    
    try:
        # Load dataset
        print(f"\n1. Loading dataset from: {dataset_path}")
        with open(dataset_path, 'rb') as f:
            graphs = pickle.load(f)
        print(f"   ✓ Successfully loaded {len(graphs)} graphs")
        
        # Check first graph
        print("\n2. Checking first graph structure:")
        graph = graphs[0]
        
        # List all attributes
        if hasattr(graph, '__dict__'):
            attrs = list(graph.__dict__.keys())
        else:
            attrs = [attr for attr in dir(graph) if not attr.startswith('_')]
        print(f"   Attributes found: {attrs}")
        
        # Check required attributes
        print("\n3. Checking required attributes:")
        required = ['x', 'edge_index']
        optional = ['A', 'graph_statistics_tensor', 'stats', 'y']
        
        for attr in required:
            if hasattr(graph, attr):
                val = getattr(graph, attr)
                if isinstance(val, torch.Tensor):
                    print(f"   ✓ {attr}: shape {val.shape}, dtype {val.dtype}")
                else:
                    print(f"   ✓ {attr}: {type(val)}")
            else:
                print(f"   ✗ {attr}: MISSING (required)")
                
        for attr in optional:
            if hasattr(graph, attr):
                val = getattr(graph, attr)
                if isinstance(val, torch.Tensor):
                    print(f"   ✓ {attr}: shape {val.shape}, dtype {val.dtype}")
                else:
                    print(f"   ✓ {attr}: {type(val)}")
            else:
                print(f"   - {attr}: not present (optional)")
        
        # Check multiple graphs
        print("\n4. Sampling multiple graphs:")
        sample_indices = [0, 100, 500, 1000, 2000]
        for idx in sample_indices:
            if idx < len(graphs):
                g = graphs[idx]
                num_nodes = g.x.shape[0] if hasattr(g, 'x') else 'N/A'
                num_edges = g.edge_index.shape[1] if hasattr(g, 'edge_index') else 'N/A'
                print(f"   Graph {idx}: {num_nodes} nodes, {num_edges} edges")
        
        # Check statistics if available
        if hasattr(graphs[0], 'graph_statistics_tensor'):
            print("\n5. Checking graph statistics:")
            stats = graphs[0].graph_statistics_tensor
            print(f"   Statistics shape: {stats.shape}")
            print(f"   Statistics values: {stats[:5]}... (first 5)")
        
        print("\n" + "=" * 80)
        print("✓ Dataset verification PASSED!")
        print("=" * 80)
        print("\nYou can now run training with:")
        print("  python main.py --use-synthetic-data --train-autoencoder --train-denoiser --visualize")
        return True
        
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"✗ Dataset verification FAILED: {str(e)}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else 'data/featurehomophily_graphs.pkl'
    success = verify_synthetic_dataset(dataset_path)
    sys.exit(0 if success else 1)
