#!/bin/bash

# Training Monitor Script
# This script helps you restart training with improved hyperparameters

echo "======================================================================"
echo "Neural Graph Generator - Training Monitor"
echo "======================================================================"
echo ""

# Check current checkpoints
echo "Current Model Checkpoints:"
echo "-------------------------"
if [ -f "autoencoder.pth.tar" ]; then
    echo "✓ Autoencoder checkpoint exists ($(ls -lh autoencoder.pth.tar | awk '{print $5}'))"
else
    echo "✗ No autoencoder checkpoint found"
fi

if [ -f "denoise_model.pth.tar" ]; then
    echo "✓ Denoiser checkpoint exists ($(ls -lh denoise_model.pth.tar | awk '{print $5}'))"
else
    echo "✗ No denoiser checkpoint found"
fi

echo ""
echo "======================================================================"
echo "Recommended Training Commands"
echo "======================================================================"
echo ""

echo "1. STOP CURRENT TRAINING (Ctrl+C) and restart with early stopping:"
echo ""
echo "   # Train Autoencoder (with early stopping)"
echo "   python main.py --use-synthetic-data \\"
echo "       --train-autoencoder \\"
echo "       --epochs-autoencoder 200 \\"
echo "       --early-stopping-patience 20 \\"
echo "       --grad-clip 1.0 \\"
echo "       --lr 1e-3"
echo ""

echo "2. After autoencoder converges, train denoiser:"
echo ""
echo "   # Train Denoiser (with early stopping)"
echo "   python main.py --use-synthetic-data \\"
echo "       --train-denoiser \\"
echo "       --epochs-denoise 100 \\"
echo "       --early-stopping-patience 15 \\"
echo "       --grad-clip 1.0 \\"
echo "       --lr 1e-3"
echo ""

echo "3. Test generation quality:"
echo ""
echo "   # Generate and visualize"
echo "   python main.py --use-synthetic-data \\"
echo "       --visualize \\"
echo "       --n-vis-samples 10"
echo ""

echo "======================================================================"
echo "Key Improvements:"
echo "======================================================================"
echo "✓ Early stopping with patience=20 (prevents overfitting)"
echo "✓ Gradient clipping (stabilizes training)"
echo "✓ Best model checkpointing (saves only when val loss improves)"
echo "✓ Informative logging (shows patience counter)"
echo ""

echo "======================================================================"
echo "What to Watch For:"
echo "======================================================================"
echo "• Val Loss / Train Loss ratio should be < 1.5"
echo "• Early stopping should trigger before max epochs"
echo "• Best model should be saved multiple times"
echo "• Generation should produce graphs with ~700 edges (not ~2)"
echo ""
