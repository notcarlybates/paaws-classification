"""
===============================================================================
Evaluation Script for Potter's Random Forest (RFT) Model
===============================================================================
This script runs the existing paaws-benchmarking scripts for LOPO cross-
validation and aggregates the results.

Author: Benchmarking Script
Date: 2025-11
===============================================================================
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from pathlib import Path

# Add paaws-benchmarking to path to access utils
BENCHMARK_DIR = Path(__file__).parent.parent / "paaws-benchmarking"
sys.path.insert(0, str(BENCHMARK_DIR))
from utils import DATASET_LISTS, MAPPING_SCHEMES


def run_single_lopo_experiment(ds_lo, dataset="2", sensor="LeftWrist", mapping="lab_fl_5", lab=True):
    """
    Run a single LOPO experiment using the existing paaws-benchmarking script.

    This function simply calls run_experiment.py from paaws-benchmarking.

    Parameters
    ----------
    ds_lo : int
        The participant ID to leave out
    dataset : str
        The dataset key from utils.DATASET_LISTS (default: "2" for [10, 36])
    sensor : str
        The sensor location (e.g., "LeftWrist", "RightWrist")
    mapping : str
        The activity mapping scheme (default: "lab_fl_5" for 5 activities)
    lab : bool
        Whether to use SimFL+Lab data (True) or FL data (False)

    Returns
    -------
    tuple
        (success, output_file_path) - success is True/False, output_file_path is the CSV path
    """

    # Create results directory
    results_dir = Path(__file__).parent / "benchmark_results" / "rft_lopo"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Build command - this calls the existing paaws-benchmarking script
    protocol = "SimFL_Lab" if lab else "FL"
    out_file = str(results_dir) + "/"

    cmd = [
        "python",
        str(BENCHMARK_DIR / "run_experiment.py"),
        f"--ds_lo={ds_lo}",
        f"--dataset={dataset}",
        f"--sensor={sensor}",
        f"--mapping={mapping}",
        f"--out_file={out_file}",
    ]

    if lab:
        cmd.append("--lab")

    print(f"\n{'='*80}")
    print(f"Running LOPO for DS_{ds_lo} (leaving out participant {ds_lo})")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    # Run the existing experiment script
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=str(BENCHMARK_DIR))
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        # Find the output CSV file created by run_experiment.py
        num_activities = len(set(MAPPING_SCHEMES[mapping].values()))
        expected_csv = f"{out_file}{protocol}_{sensor}_{num_activities}_Acts_{dataset}_Participants_DS_{ds_lo}.csv"

        if os.path.exists(expected_csv):
            return True, expected_csv
        else:
            print(f"WARNING: Expected output file not found: {expected_csv}")
            # Try to find any CSV files that match the pattern
            matching_files = list(results_dir.glob(f"*DS_{ds_lo}.csv"))
            if matching_files:
                print(f"Found alternative file: {matching_files[0]}")
                return True, str(matching_files[0])
            return False, None

    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to run experiment for DS_{ds_lo}")
        print(f"Return code: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False, None


def compute_aggregate_metrics(result_files):
    """
    Compute aggregate metrics from multiple LOPO result files.

    Parameters
    ----------
    result_files : list of str
        List of CSV file paths containing predictions

    Returns
    -------
    dict
        Dictionary containing aggregate metrics
    """

    all_true_labels = []
    all_predictions = []
    per_participant_results = []

    print(f"\n{'='*80}")
    print("Computing Aggregate Metrics")
    print(f"{'='*80}\n")

    for result_file in result_files:
        if not os.path.exists(result_file):
            print(f"WARNING: File not found: {result_file}")
            continue

        # Read the predictions
        df = pd.read_csv(result_file)

        # Extract participant ID from filename
        # Format: *_DS_10.csv
        participant_id = int(result_file.split("_DS_")[-1].replace(".csv", ""))

        # Get true labels and predictions
        true_labels = df["MAPPED_LABEL"].values
        predictions = df["PREDICTION"].values

        # Compute per-participant metrics
        acc = accuracy_score(true_labels, predictions)
        f1_macro = f1_score(true_labels, predictions, average="macro", zero_division=0)
        f1_weighted = f1_score(true_labels, predictions, average="weighted", zero_division=0)

        per_participant_results.append({
            "Participant": f"DS_{participant_id}",
            "Accuracy": acc,
            "F1_Macro": f1_macro,
            "F1_Weighted": f1_weighted,
            "Num_Samples": len(true_labels)
        })

        print(f"DS_{participant_id}: Accuracy={acc:.4f}, F1-Macro={f1_macro:.4f}, F1-Weighted={f1_weighted:.4f} (n={len(true_labels)})")

        # Aggregate for overall metrics
        all_true_labels.extend(true_labels)
        all_predictions.extend(predictions)

    # Compute overall metrics
    overall_accuracy = accuracy_score(all_true_labels, all_predictions)
    overall_f1_macro = f1_score(all_true_labels, all_predictions, average="macro", zero_division=0)
    overall_f1_weighted = f1_score(all_true_labels, all_predictions, average="weighted", zero_division=0)

    # Generate classification report
    unique_labels = sorted(set(all_true_labels))
    class_report = classification_report(
        all_true_labels,
        all_predictions,
        labels=unique_labels,
        output_dict=True,
        zero_division=0
    )

    # Generate confusion matrix
    cm = confusion_matrix(all_true_labels, all_predictions, labels=unique_labels)

    print(f"\n{'='*80}")
    print("AGGREGATE RESULTS (LOPO Cross-Validation)")
    print(f"{'='*80}")
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Overall F1-Macro: {overall_f1_macro:.4f}")
    print(f"Overall F1-Weighted: {overall_f1_weighted:.4f}")
    print(f"Total Samples: {len(all_true_labels)}")
    print(f"{'='*80}\n")

    print("\nClassification Report:")
    print(classification_report(all_true_labels, all_predictions, labels=unique_labels, zero_division=0))

    return {
        "overall_accuracy": overall_accuracy,
        "overall_f1_macro": overall_f1_macro,
        "overall_f1_weighted": overall_f1_weighted,
        "per_participant_results": pd.DataFrame(per_participant_results),
        "confusion_matrix": cm,
        "classification_report": class_report,
        "unique_labels": unique_labels
    }


def main():
    """
    Main evaluation function.

    This orchestrates the evaluation by calling the existing paaws-benchmarking
    scripts and aggregating results.
    """

    print("""
    ===============================================================================
    RFT Model Evaluation Script
    ===============================================================================
    This script calls the existing paaws-benchmarking/run_experiment.py script
    for LOPO cross-validation and aggregates the results.

    Configuration:
    - Dataset: "2" (participants: 10, 36)
    - Sensor: LeftWrist
    - Mapping: lab_fl_5 (5 activities: Sitting, Standing, Walking, Biking, Lying_Down)
    - Protocol: SimFL+Lab

    This uses Potter's existing code - we just orchestrate and aggregate results.
    ===============================================================================
    """)

    # Configuration
    dataset = "2"  # Small test dataset with participants [10, 36]
    sensor = "LeftWrist"
    mapping = "lab_fl_5"  # 5 activity classes
    lab = True  # Use SimFL+Lab data

    participants = DATASET_LISTS[dataset]
    print(f"Participants to evaluate: {participants}")

    # Run LOPO for each participant
    result_files = []
    for ds_lo in participants:
        success, output_file = run_single_lopo_experiment(
            ds_lo=ds_lo,
            dataset=dataset,
            sensor=sensor,
            mapping=mapping,
            lab=lab
        )

        if success and output_file:
            result_files.append(output_file)
            print(f"✓ Successfully completed DS_{ds_lo}")
        else:
            print(f"✗ Failed to complete DS_{ds_lo}")

    # Compute aggregate metrics
    if len(result_files) > 0:
        metrics = compute_aggregate_metrics(result_files)

        # Save results
        results_dir = Path(__file__).parent / "benchmark_results" / "rft_lopo"

        # Save per-participant results
        per_participant_df = metrics["per_participant_results"]
        per_participant_csv = results_dir / "rft_per_participant_results.csv"
        per_participant_df.to_csv(per_participant_csv, index=False)
        print(f"\nPer-participant results saved to: {per_participant_csv}")

        # Save aggregate metrics
        aggregate_metrics_file = results_dir / "rft_aggregate_metrics.txt"
        with open(aggregate_metrics_file, "w") as f:
            f.write("="*80 + "\n")
            f.write("RFT Model - Aggregate Metrics (LOPO Cross-Validation)\n")
            f.write("="*80 + "\n\n")
            f.write(f"Dataset: {dataset} (Participants: {participants})\n")
            f.write(f"Sensor: {sensor}\n")
            f.write(f"Mapping: {mapping}\n")
            f.write(f"Protocol: {'SimFL+Lab' if lab else 'FL'}\n\n")
            f.write(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}\n")
            f.write(f"Overall F1-Macro: {metrics['overall_f1_macro']:.4f}\n")
            f.write(f"Overall F1-Weighted: {metrics['overall_f1_weighted']:.4f}\n\n")
            f.write("Per-Participant Results:\n")
            f.write(per_participant_df.to_string(index=False))
            f.write("\n\n")
            f.write("Classification Report:\n")
            from sklearn.metrics import classification_report
            # Get labels as strings from confusion matrix
            f.write(classification_report(
                [str(x) for x in metrics["unique_labels"]] * len(result_files),
                [str(x) for x in metrics["unique_labels"]] * len(result_files),
                zero_division=0
            ))

        print(f"Aggregate metrics saved to: {aggregate_metrics_file}")

        # Save confusion matrix
        cm_file = results_dir / "rft_confusion_matrix.csv"
        cm_df = pd.DataFrame(
            metrics["confusion_matrix"],
            index=metrics["unique_labels"],
            columns=metrics["unique_labels"]
        )
        cm_df.to_csv(cm_file)
        print(f"Confusion matrix saved to: {cm_file}")

        print("\n✓ RFT Model Evaluation Complete!")
        print(f"\nSummary:")
        print(f"  Accuracy: {metrics['overall_accuracy']:.4f}")
        print(f"  F1-Macro: {metrics['overall_f1_macro']:.4f}")
        print(f"  F1-Weighted: {metrics['overall_f1_weighted']:.4f}")

    else:
        print("\n✗ No successful experiments completed. Cannot compute aggregate metrics.")
        sys.exit(1)


if __name__ == "__main__":
    main()
