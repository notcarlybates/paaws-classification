"""
===============================================================================
LSTM Model Evaluation Script
===============================================================================
This script trains and evaluates a simple LSTM model using the same data
preprocessing as Potter's RFT model (paaws-benchmarking).

It uses:
- Same data loading (get_and_clean_data.py from paaws-benchmarking)
- Same windowing approach (10-second windows at 80 Hz)
- Same LOSO cross-validation
- Same metrics (accuracy, F1-macro, F1-weighted)

The only difference: Instead of extracting handcrafted features, we feed
raw accelerometer sequences into an LSTM.

Author: Benchmarking Script
Date: 2025-11
===============================================================================
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from pathlib import Path
from tqdm import tqdm

# Add paaws-benchmarking to path
BENCHMARK_DIR = Path(__file__).parent.parent / "paaws-benchmarking"
sys.path.insert(0, str(BENCHMARK_DIR))

from utils import DATASET_LISTS, MAPPING_SCHEMES
from lstm_model import SimpleLSTM, train_model, evaluate_model


def create_sequences_from_windows(windowed_accel, window_size=800):
    """
    Create sequences from windowed accelerometer data.

    Potter's code creates 10-second windows at 80 Hz = 800 samples per window.
    We reshape this into sequences for LSTM.

    Args:
        windowed_accel: DataFrame with windowed accelerometer data
        window_size: Number of samples per window (default: 800 for 10s at 80Hz)

    Returns:
        numpy array of shape (num_windows, window_size, 3) for x, y, z
    """
    sequences = []

    for _, row in windowed_accel.iterrows():
        # Extract the accelerometer values (Axis1, Axis2, Axis3)
        # Potter's windowing creates columns like Axis1_0, Axis1_1, ..., Axis1_799
        x_vals = [row[f'Axis1_{i}'] for i in range(window_size)]
        y_vals = [row[f'Axis2_{i}'] for i in range(window_size)]
        z_vals = [row[f'Axis3_{i}'] for i in range(window_size)]

        # Stack into (window_size, 3) array
        sequence = np.column_stack([x_vals, y_vals, z_vals])
        sequences.append(sequence)

    return np.array(sequences)


def prepare_data_for_lstm(acc_features_dict, windowed_labels, ds_lo, activity_to_idx):
    """
    Prepare data for LSTM training (similar to Potter's make_training_sets_from_np).

    Args:
        acc_features_dict: Dictionary of windowed accelerometer data by participant
        windowed_labels: Dictionary of labels by participant
        ds_lo: Participant ID to leave out
        activity_to_idx: Mapping from activity names to indices

    Returns:
        Tuple of (train_sequences, train_labels, test_sequences, test_labels)
    """
    train_sequences = []
    train_labels = []
    test_sequences = None
    test_labels = None

    for ds in acc_features_dict.keys():
        # Create sequences from windowed data
        sequences = create_sequences_from_windows(acc_features_dict[ds])

        # Get labels
        if isinstance(windowed_labels[ds], pd.DataFrame):
            labels = windowed_labels[ds]["MAPPED_LABEL"].values
        else:
            labels = windowed_labels[ds][:, 1]

        # Convert activity names to indices
        label_indices = np.array([activity_to_idx[label] for label in labels])

        # Split into train/test based on LOSO
        if int(ds) == int(ds_lo):
            test_sequences = sequences
            test_labels = label_indices
        else:
            train_sequences.append(sequences)
            train_labels.append(label_indices)

    # Concatenate training data
    if len(train_sequences) > 0:
        train_sequences = np.concatenate(train_sequences, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)
    else:
        raise ValueError("No training data available!")

    return train_sequences, train_labels, test_sequences, test_labels


def run_lstm_lopo(dataset="2", sensor="LeftWrist", mapping="lab_fl_5", lab=True,
                  hidden_size=64, num_layers=2, batch_size=64, num_epochs=50,
                  learning_rate=0.001, patience=10):
    """
    Run LSTM evaluation with LOSO cross-validation.

    This mirrors the RFT evaluation but uses LSTM instead of Random Forest.

    Args:
        dataset: Dataset ID from DATASET_LISTS
        sensor: Sensor location
        mapping: Activity mapping scheme
        lab: Use SimFL+Lab data
        hidden_size: LSTM hidden size
        num_layers: Number of LSTM layers
        batch_size: Batch size for training
        num_epochs: Maximum number of epochs
        learning_rate: Learning rate
        patience: Early stopping patience

    Returns:
        Dictionary with results for each participant
    """
    # Setup configuration (similar to Potter's run_experiment.py)
    participants = DATASET_LISTS[dataset]
    activity_mapping = MAPPING_SCHEMES[mapping]
    activities = sorted(set(activity_mapping.values()))
    activity_to_idx = {act: idx for idx, act in enumerate(activities)}
    idx_to_activity = {idx: act for act, idx in activity_to_idx.items()}
    num_classes = len(activities)

    # Configuration dict (mimics Potter's config)
    # IMPORTANT: This must be created BEFORE importing get_and_clean_data
    config = {
        "DATASET": dataset,
        "DATASETS": participants,
        "FREQ": 80,
        "T": 10,
        "WINDOW_SIZE": 800,
        "SENSOR": sensor,
        "LAB": lab,
        "FL": not lab,
        "ACT_MAPPING": activity_mapping,
        "ACT_LIST": set(activities),
        "NUM_ACTS": num_classes,
    }
    sys.modules["config"] = type('Config', (), config)()

    # Import Potter's data loading functions (after config is set up)
    from get_and_clean_data import (
        get_dataset_accel,
        get_dataset_labels,
        window_dataset_labels,
        window_dataset_accel,
    )

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    results_dir = Path(__file__).parent / "benchmark_results" / "lstm_lopo"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print("LSTM Model Evaluation - LOSO Cross-Validation")
    print(f"{'='*80}")
    print(f"Dataset: {dataset} (Participants: {participants})")
    print(f"Sensor: {sensor}")
    print(f"Mapping: {mapping} ({num_classes} activities: {activities})")
    print(f"Protocol: {'SimFL+Lab' if lab else 'FL'}")
    print(f"Window size: {config['WINDOW_SIZE']} samples (10s at 80 Hz)")
    print(f"{'='*80}\n")

    # Load and process all participants
    print("Loading and processing data...")
    windowed_accel = {}
    windowed_labels_dict = {}

    for ds in tqdm(participants, desc="Loading participants"):
        try:
            # Load data using Potter's functions
            accel, accel_start = get_dataset_accel(ds)
            labels = get_dataset_labels(ds)

            # Window data
            windowed_label = window_dataset_labels(labels)
            windowed_accel_single = window_dataset_accel(accel, accel_start, windowed_label)

            windowed_accel[ds] = windowed_accel_single
            windowed_labels_dict[ds] = windowed_label

        except Exception as e:
            print(f"\nError loading DS_{ds}: {e}")
            continue

    # Run LOSO cross-validation
    all_results = []

    for ds_lo in participants:
        print(f"\n{'='*80}")
        print(f"LOSO Iteration: Leaving out DS_{ds_lo}")
        print(f"{'='*80}")

        try:
            # Prepare data
            train_X, train_y, test_X, test_y = prepare_data_for_lstm(
                windowed_accel, windowed_labels_dict, ds_lo, activity_to_idx
            )

            print(f"Training samples: {len(train_X):,}")
            print(f"Test samples: {len(test_X):,}")
            print(f"Sequence shape: {train_X.shape}")

            # Convert to PyTorch tensors
            train_X_tensor = torch.FloatTensor(train_X)
            train_y_tensor = torch.LongTensor(train_y)
            test_X_tensor = torch.FloatTensor(test_X)
            test_y_tensor = torch.LongTensor(test_y)

            # Create DataLoaders
            train_dataset = TensorDataset(train_X_tensor, train_y_tensor)
            test_dataset = TensorDataset(test_X_tensor, test_y_tensor)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            # Initialize model
            model = SimpleLSTM(
                input_size=3,  # x, y, z
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_classes=num_classes,
                dropout=0.3
            ).to(device)

            # Loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            # Training loop with early stopping
            best_loss = float('inf')
            patience_counter = 0

            print(f"\nTraining LSTM...")
            for epoch in range(num_epochs):
                train_loss = train_model(model, train_loader, criterion, optimizer, device)

                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}")

                # Early stopping
                if train_loss < best_loss:
                    best_loss = train_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break

            # Evaluate
            print(f"Evaluating on DS_{ds_lo}...")
            predictions, true_labels = evaluate_model(model, test_loader, device)

            # Compute metrics (same as RFT)
            accuracy = accuracy_score(true_labels, predictions)
            f1_macro = f1_score(true_labels, predictions, average='macro', zero_division=0)
            f1_weighted = f1_score(true_labels, predictions, average='weighted', zero_division=0)

            print(f"\nResults for DS_{ds_lo}:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1-Macro: {f1_macro:.4f}")
            print(f"  F1-Weighted: {f1_weighted:.4f}")

            # Save results
            result = {
                "Participant": f"DS_{ds_lo}",
                "Accuracy": accuracy,
                "F1_Macro": f1_macro,
                "F1_Weighted": f1_weighted,
                "Num_Samples": len(test_y),
                "predictions": predictions,
                "true_labels": true_labels
            }
            all_results.append(result)

        except Exception as e:
            print(f"\nError processing DS_{ds_lo}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Aggregate results
    print(f"\n{'='*80}")
    print("Aggregating Results")
    print(f"{'='*80}\n")

    # Compute overall metrics
    all_predictions = []
    all_true_labels = []

    for result in all_results:
        all_predictions.extend(result["predictions"])
        all_true_labels.extend(result["true_labels"])

    overall_accuracy = accuracy_score(all_true_labels, all_predictions)
    overall_f1_macro = f1_score(all_true_labels, all_predictions, average='macro', zero_division=0)
    overall_f1_weighted = f1_score(all_true_labels, all_predictions, average='weighted', zero_division=0)

    # Create summary DataFrame
    summary_df = pd.DataFrame([{
        "Participant": r["Participant"],
        "Accuracy": r["Accuracy"],
        "F1_Macro": r["F1_Macro"],
        "F1_Weighted": r["F1_Weighted"],
        "Num_Samples": r["Num_Samples"]
    } for r in all_results])

    # Add average row
    summary_df.loc[len(summary_df)] = [
        "Average",
        overall_accuracy,
        overall_f1_macro,
        overall_f1_weighted,
        sum(r["Num_Samples"] for r in all_results)
    ]

    # Save summary
    summary_df.to_csv(results_dir / "lstm_per_participant_results.csv", index=False)

    # Confusion matrix
    cm = confusion_matrix(all_true_labels, all_predictions)
    cm_df = pd.DataFrame(cm, index=activities, columns=activities)
    cm_df.to_csv(results_dir / "lstm_confusion_matrix.csv")

    # Classification report
    report = classification_report(all_true_labels, all_predictions, target_names=activities, zero_division=0)

    # Save aggregate metrics
    with open(results_dir / "lstm_aggregate_metrics.txt", "w") as f:
        f.write("="*80 + "\n")
        f.write("LSTM Model - Aggregate Metrics (LOSO Cross-Validation)\n")
        f.write("="*80 + "\n\n")
        f.write(f"Dataset: {dataset} (Participants: {participants})\n")
        f.write(f"Sensor: {sensor}\n")
        f.write(f"Mapping: {mapping}\n")
        f.write(f"Protocol: {'SimFL+Lab' if lab else 'FL'}\n\n")
        f.write(f"Overall Accuracy: {overall_accuracy:.4f}\n")
        f.write(f"Overall F1-Macro: {overall_f1_macro:.4f}\n")
        f.write(f"Overall F1-Weighted: {overall_f1_weighted:.4f}\n\n")
        f.write("Per-Participant Results:\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    # Print summary
    print("="*80)
    print("LSTM Model - Aggregate Results")
    print("="*80)
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Overall F1-Macro: {overall_f1_macro:.4f}")
    print(f"Overall F1-Weighted: {overall_f1_weighted:.4f}")
    print("="*80)
    print("\nPer-Participant Results:")
    print(summary_df.to_string(index=False))
    print("\n" + "="*80)

    print(f"\nResults saved to: {results_dir}")
    print(f"  - lstm_aggregate_metrics.txt")
    print(f"  - lstm_per_participant_results.csv")
    print(f"  - lstm_confusion_matrix.csv")

    return all_results


def main():
    """Main evaluation function."""

    print("""
    ===============================================================================
    LSTM Model Evaluation
    ===============================================================================
    This script trains a simple LSTM model using the same data preprocessing
    as Potter's RFT model (paaws-benchmarking).

    Same as RFT:
    - Data loading and windowing (10s windows at 80 Hz)
    - LOSO cross-validation
    - Metrics (accuracy, F1-macro, F1-weighted)

    Difference:
    - Uses raw accelerometer sequences instead of handcrafted features
    - LSTM neural network instead of Random Forest
    ===============================================================================
    """)

    # Configuration (same as evaluate_rft_model.py)
    dataset = "2"  # Participants [10, 36]
    sensor = "LeftWrist"
    mapping = "lab_fl_5"  # 5 activities
    lab = True  # SimFL+Lab data

    # LSTM hyperparameters
    hidden_size = 64
    num_layers = 2
    batch_size = 64
    num_epochs = 50
    learning_rate = 0.001
    patience = 10

    # Run evaluation
    results = run_lstm_lopo(
        dataset=dataset,
        sensor=sensor,
        mapping=mapping,
        lab=lab,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        patience=patience
    )

    print("\n✓ LSTM Model Evaluation Complete!")


if __name__ == "__main__":
    main()
