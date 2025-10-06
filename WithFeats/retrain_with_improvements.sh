#!/bin/bash

# Complete Retraining Script with All Improvements
# Use this after the current training finishes

echo "======================================================================"
echo "Neural Graph Generator - Complete Retraining Pipeline"
echo "======================================================================"
echo ""
echo "This script will retrain both models with:"
echo "  ✓ Stratified data splitting (balanced homophily distribution)"
echo "  ✓ Early stopping (patience=20 for autoencoder, 15 for denoiser)"
echo "  ✓ Gradient clipping (prevents exploding gradients)"
echo "  ✓ Best model checkpointing (saves only improvements)"
echo ""
echo "======================================================================"
echo ""

# Check if old checkpoints exist
if [ -f "autoencoder.pth.tar" ] || [ -f "denoise_model.pth.tar" ]; then
    echo "⚠️  WARNING: Old checkpoints detected!"
    echo ""
    echo "Found:"
    [ -f "autoencoder.pth.tar" ] && echo "  - autoencoder.pth.tar"
    [ -f "denoise_model.pth.tar" ] && echo "  - denoise_model.pth.tar"
    echo ""
    read -p "Delete old checkpoints and retrain from scratch? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Backing up old checkpoints..."
        [ -f "autoencoder.pth.tar" ] && mv autoencoder.pth.tar autoencoder.pth.tar.backup
        [ -f "denoise_model.pth.tar" ] && mv denoise_model.pth.tar denoise_model.pth.tar.backup
        echo "✓ Old checkpoints backed up"
        echo ""
    else
        echo "Keeping old checkpoints. New training will overwrite them."
        echo ""
    fi
fi

echo "======================================================================"
echo "STEP 1: Training Autoencoder"
echo "======================================================================"
echo ""
echo "Expected improvements:"
echo "  • Stratified splits ensure balanced homophily across train/val/test"
echo "  • Early stopping prevents overfitting (stops when val loss plateaus)"
echo "  • Gradient clipping stabilizes training"
echo ""
echo "Starting in 3 seconds... (Press Ctrl+C to cancel)"
sleep 3
echo ""

# Train autoencoder
python main.py --use-synthetic-data \
    --train-autoencoder \
    --epochs-autoencoder 200 \
    --early-stopping-patience 50 \
    --grad-clip 1.0 \
    --lr 0.0001

echo ""
echo "======================================================================"
echo "Autoencoder Training Complete!"
echo "======================================================================"
echo ""

# Check if autoencoder checkpoint was created
if [ ! -f "autoencoder.pth.tar" ]; then
    echo "❌ ERROR: Autoencoder checkpoint not found!"
    echo "Training may have failed. Please check the output above."
    exit 1
fi

echo "✓ Autoencoder checkpoint created"
echo ""
echo "======================================================================"
echo "STEP 2: Training Denoiser"
echo "======================================================================"
echo ""
echo "The denoiser will use the trained autoencoder to learn the"
echo "latent space distribution conditioned on graph statistics."
echo ""
echo "Starting in 3 seconds... (Press Ctrl+C to cancel)"
sleep 3
echo ""

# Train denoiser
python main.py --use-synthetic-data \
    --train-denoiser \
    --epochs-denoise 100 \
    --early-stopping-patience 50 \
    --grad-clip 1.0 \
    --lr 0.00001

echo ""
echo "======================================================================"
echo "Denoiser Training Complete!"
echo "======================================================================"
echo ""

# Check if denoiser checkpoint was created
if [ ! -f "denoise_model.pth.tar" ]; then
    echo "❌ ERROR: Denoiser checkpoint not found!"
    echo "Training may have failed. Please check the output above."
    exit 1
fi

echo "✓ Denoiser checkpoint created"
echo ""
echo "======================================================================"
echo "STEP 3: Testing Generation Quality"
echo "======================================================================"
echo ""
echo "Generating 10 sample graphs and comparing with ground truth..."
echo ""

# Test generation
python main.py --use-synthetic-data \
    --visualize \
    --n-vis-samples 10

echo ""
echo "======================================================================"
echo "Training Pipeline Complete!"
echo "======================================================================"
echo ""
echo "Results:"
echo "  ✓ Autoencoder trained with stratified data"
echo "  ✓ Denoiser trained with stratified data"
echo "  ✓ Visualization generated: figures/graph_comparison.png"
echo ""
echo "Next steps:"
echo "  1. Check figures/graph_comparison.png for visual quality"
echo "  2. Review y_stats.txt vs y_pred_stats.txt for statistics"
echo "  3. Compare MSE/MAE/SMAPE metrics with previous training"
echo ""
echo "Expected improvements:"
echo "  • Val/Train loss ratio closer to 1.0-1.3 (was 2.2)"
echo "  • Generated graphs with ~700 edges (was ~2-400)"
echo "  • Lower MSE/MAE on all graph statistics"
echo ""
echo "Open visualization: open figures/graph_comparison.png"
echo ""
