"""
===============================================================================
Evaluation Script for Mazzuchelli's LSTM Model
===============================================================================
This script calls the existing paaws_activity_detection/scripts/train_loso.py
script for LOSO cross-validation.

The existing scripts in paaws_activity_detection handle:
- Data loading and preprocessing
- Sequence generation
- LOSO cross-validation
- Results aggregation

You need to provide a preprocessed dataset CSV file with columns:
- x, y, z (accelerometer values)
- label (activity label)
- subject_id (participant identifier)

Author: Benchmarking Script
Date: 2025-11
===============================================================================
"""

import os
import sys
import subprocess
import pandas as pd
from pathlib import Path

# Add paaws_activity_detection to path
LSTM_DIR = Path(__file__).parent.parent / "paaws_activity_detection"
sys.path.insert(0, str(LSTM_DIR))


def run_lstm_evaluation(dataset_path, model_type="lstm"):
    """
    Run LSTM evaluation by calling the existing train_loso.py script.

    The existing script handles all preprocessing, training, and evaluation.
    We just need to call it with the right parameters.

    Parameters
    ----------
    dataset_path : str
        Path to the preprocessed dataset CSV file
    model_type : str
        Type of model to use ("lstm", "cnn_lstm", or "attentive_lstm")

    Returns
    -------
    bool
        True if successful, False otherwise
    """

    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"\n✗ ERROR: Dataset not found at: {dataset_path}")
        print("\nThe CSV file should have columns: x, y, z, label, subject_id")
        print("See BENCHMARKING_INSTRUCTIONS.md for data preparation guidance.")
        return False

    print(f"\n{'='*80}")
    print(f"Running LSTM Evaluation (using existing train_loso.py)")
    print(f"{'='*80}")
    print(f"Dataset: {dataset_path}")
    print(f"Model Type: {model_type}")
    print(f"{'='*80}\n")

    # Build command to call existing train_loso.py script
    train_script = LSTM_DIR / "scripts" / "train_loso.py"
    cmd = [
        "python",
        str(train_script),
        f"--dataset={dataset_path}",
        f"--model_type={model_type}",
        "--exclude_unknown",  # Exclude unknown activity samples
    ]

    print(f"Command: {' '.join(cmd)}\n")
    print("This will run the existing LOSO evaluation from paaws_activity_detection.")
    print("Results will be saved to: paaws_activity_detection/loso_results/\n")

    # Run the existing evaluation script
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd=str(LSTM_DIR)
        )

        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        print("\n✓ LSTM Evaluation completed successfully!")

        # Display results
        loso_results_dir = LSTM_DIR / "loso_results"
        if loso_results_dir.exists():
            print(f"\nResults saved to: {loso_results_dir}")

            summary_file = loso_results_dir / "summary.csv"
            if summary_file.exists():
                print(f"Summary file: {summary_file}")

                # Read and display summary
                summary_df = pd.read_csv(summary_file)
                print("\n" + "="*80)
                print("LSTM Model Results Summary")
                print("="*80)
                print(summary_df.to_string(index=False))
                print("="*80)

        return True

    except subprocess.CalledProcessError as e:
        print(f"\n✗ ERROR: LSTM evaluation failed")
        print(f"Return code: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False
    except Exception as e:
        print(f"\n✗ ERROR: Unexpected error: {e}")
        return False


def main():
    """
    Main evaluation function.

    This orchestrates the evaluation by calling the existing train_loso.py script.
    """

    print("""
    ===============================================================================
    LSTM Model Evaluation Script
    ===============================================================================
    This script calls paaws_activity_detection/scripts/train_loso.py which
    handles all data preprocessing, training, and LOSO cross-validation.

    REQUIREMENT: Preprocessed dataset CSV file with columns:
    - x, y, z: Accelerometer values
    - label: Activity label
    - subject_id: Participant identifier (e.g., "DS_10", "DS_36")

    The existing paaws_activity_detection scripts handle:
    - Sequence generation from raw accelerometer data
    - LOSO cross-validation
    - Model training and evaluation
    - Results aggregation and visualization
    ===============================================================================
    """)

    # Prompt user for dataset path
    print("\n" + "="*80)
    print("Dataset Input Required")
    print("="*80)
    print("\nProvide the path to your preprocessed dataset CSV file.")
    print("Example: C:\\path\\to\\preprocessed_paaws_data.csv")
    print("\nColumns required: x, y, z, label, subject_id")
    print("See BENCHMARKING_INSTRUCTIONS.md for data preparation guidance.")
    print("\nPress Enter to use the default path, or provide your own path:")

    dataset_path = Path(__file__).parent / "data" / "preprocessed_paaws_subset.csv"

    # Prompt for model type
    print("\n" + "="*80)
    print("Model Type Selection")
    print("="*80)
    print("Available model types:")
    print("1. lstm (default) - Multi-Head LSTM with SE blocks")
    print("2. cnn_lstm - CNN-LSTM model")
    print("3. attentive_lstm - Multi-Head LSTM with Attention")

    model_type = "lstm" # "lstm", "cnn_lstm", "attentive_lstm"

    # Run evaluation using existing script
    success = run_lstm_evaluation(dataset_path, model_type)

    if success:
        print("\n✓ LSTM Model Evaluation Complete!")
        print("\nResults location: paaws_activity_detection/loso_results/")
        print("Key files:")
        print("  - summary.csv: Per-subject and aggregate metrics")
        print("  - per_class_metrics.csv: Per-class performance")
        print("  - aggregated_confusion_matrix.png: Confusion matrix visualization")
    else:
        print("\n✗ LSTM Model Evaluation Failed!")
        print("\nCheck:")
        print("1. Dataset format: CSV with x, y, z, label, subject_id columns")
        print("2. subject_id format: DS_XX (e.g., DS_10, DS_36)")
        print("3. Required packages: torch, pandas, numpy, pyyaml, etc.")


if __name__ == "__main__":
    main()
