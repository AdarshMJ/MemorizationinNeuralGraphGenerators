#!/bin/bash

# Script to compare multiple training runs

echo "======================================================================"
echo "Neural Graph Generator - Run Comparison Tool"
echo "======================================================================"
echo ""

OUTPUT_DIR="${1:-outputs}"

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "❌ Error: Directory '$OUTPUT_DIR' not found"
    echo ""
    echo "Usage: $0 [output_directory]"
    echo "Example: $0 outputs"
    exit 1
fi

# Count total runs
TOTAL_RUNS=$(find "$OUTPUT_DIR" -maxdepth 1 -type d | tail -n +2 | wc -l)
echo "Found $TOTAL_RUNS training runs in $OUTPUT_DIR/"
echo ""

if [ $TOTAL_RUNS -eq 0 ]; then
    echo "No runs found. Train a model first!"
    echo ""
    echo "Example:"
    echo "  python main.py --use-synthetic-data --train-autoencoder --run-name my_experiment"
    exit 0
fi

echo "======================================================================"
echo "Run Summaries"
echo "======================================================================"
echo ""

# Display summaries
for run_dir in "$OUTPUT_DIR"/*/; do
    if [ -d "$run_dir" ]; then
        run_name=$(basename "$run_dir")
        summary_file="$run_dir/run_summary.txt"
        
        echo "────────────────────────────────────────────────────────────────────────────────"
        echo "📁 Run: $run_name"
        echo "────────────────────────────────────────────────────────────────────────────────"
        
        if [ -f "$summary_file" ]; then
            # Extract key metrics
            echo "⏰ Timestamp: $(grep "Timestamp:" "$summary_file" | cut -d: -f2- | xargs)"
            echo "📊 Dataset: $(grep "Dataset:" "$summary_file" | cut -d: -f2 | xargs)"
            mode=$(grep "Training Mode:" "$summary_file" | cut -d: -f2 | xargs)
            echo "🎯 Mode: $mode"
            
            if grep -q "MSE" "$summary_file"; then
                echo ""
                echo "Results:"
                grep "MSE (all features):" "$summary_file" | sed 's/^/  /'
                grep "MAE (all features):" "$summary_file" | sed 's/^/  /'
                grep "SMAPE (all features):" "$summary_file" | sed 's/^/  /'
            fi
            
            # Check for visualization
            if [ -d "$run_dir/figures" ] && [ "$(ls -A "$run_dir/figures")" ]; then
                echo ""
                echo "📊 Visualizations: $(ls "$run_dir/figures" | wc -l) file(s)"
            fi
            
            # Check for checkpoints
            if [ -d "$run_dir/checkpoints" ]; then
                checkpoints=$(ls "$run_dir/checkpoints"/*.tar 2>/dev/null | wc -l)
                if [ $checkpoints -gt 0 ]; then
                    echo "💾 Checkpoints: $checkpoints model(s)"
                fi
            fi
            
            # Check log file size
            if [ -f "$run_dir/logs/training.log" ]; then
                log_size=$(du -h "$run_dir/logs/training.log" | cut -f1)
                echo "📝 Log size: $log_size"
            fi
        else
            echo "⚠️  No summary file found (run may be incomplete)"
        fi
        
        echo ""
    fi
done

echo "======================================================================"
echo "Storage Usage"
echo "======================================================================"
du -sh "$OUTPUT_DIR"
echo ""

echo "======================================================================"
echo "Recent Runs (Last 5)"
echo "======================================================================"
ls -lt "$OUTPUT_DIR" | head -6 | tail -5
echo ""

echo "======================================================================"
echo "Quick Actions"
echo "======================================================================"
echo ""
echo "View specific run details:"
echo "  cat $OUTPUT_DIR/[RUN_NAME]/run_summary.txt"
echo ""
echo "View training log:"
echo "  cat $OUTPUT_DIR/[RUN_NAME]/logs/training.log"
echo ""
echo "Open visualization:"
echo "  open $OUTPUT_DIR/[RUN_NAME]/figures/*.png"
echo ""
echo "Compare two runs:"
echo "  diff $OUTPUT_DIR/[RUN1]/stats/y_pred_stats.txt $OUTPUT_DIR/[RUN2]/stats/y_pred_stats.txt"
echo ""
echo "Archive old runs:"
echo "  tar -czf archive.tar.gz $OUTPUT_DIR/[RUN_NAME]"
echo ""
echo "Delete old runs:"
echo "  rm -rf $OUTPUT_DIR/[RUN_NAME]"
echo ""
