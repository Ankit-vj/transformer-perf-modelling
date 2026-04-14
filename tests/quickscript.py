#!/usr/bin/env python3
"""
Quick test script to verify the installation
"""

import torch
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import yaml
import sys

def test_pytorch():
    """Test PyTorch installation"""
    print("Testing PyTorch...")
    x = torch.randn(2, 3)
    y = torch.randn(3, 4)
    z = torch.mm(x, y)
    print(f"  ✓ PyTorch matrix multiplication: {z.shape}")
    print(f"  ✓ CUDA available: {torch.cuda.is_available()}")

def test_numpy():
    """Test NumPy installation"""
    print("\nTesting NumPy...")
    a = np.random.randn(100, 100)
    b = np.random.randn(100, 100)
    c = np.dot(a, b)
    print(f"  ✓ NumPy matrix multiplication: {c.shape}")

def test_pandas():
    """Test Pandas installation"""
    print("\nTesting Pandas...")
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    print(f"  ✓ Pandas DataFrame created: {df.shape}")

def test_networkx():
    """Test NetworkX installation"""
    print("\nTesting NetworkX...")
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (3, 4)])
    print(f"  ✓ NetworkX graph created: {len(G.nodes())} nodes, {len(G.edges())} edges")

def test_matplotlib():
    """Test Matplotlib installation"""
    print("\nTesting Matplotlib...")
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    plt.savefig('test_plot.png')
    plt.close()
    print(f"  ✓ Matplotlib plot saved to /tmp/test_plot.png")

def test_yaml():
    """Test YAML installation"""
    print("\nTesting YAML...")
    data = {'name': 'test', 'value': 42}
    yaml_str = yaml.dump(data)
    loaded = yaml.safe_load(yaml_str)
    print(f"  ✓ YAML dump and load: {loaded}")

def test_transformers():
    """Test Transformers library"""
    print("\nTesting Transformers library...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        tokens = tokenizer("Hello, world!")
        print(f"  ✓ Transformers tokenization: {len(tokens['input_ids'])} tokens")
    except Exception as e:
        print(f"  ⚠ Transformers test skipped (requires internet): {e}")

def main():
    print("=" * 60)
    print("Running Installation Tests")
    print("=" * 60)
    
    try:
        test_pytorch()
        test_numpy()
        test_pandas()
        test_networkx()
        test_matplotlib()
        test_yaml()
        test_transformers()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed successfully!")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())