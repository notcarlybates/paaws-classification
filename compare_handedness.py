"""
===============================================================================
Handedness Comparison Script
===============================================================================
This script compares model performance between dominant and non-dominant hand
sensor placements. It analyzes results from RFT or LSTM models and determines
whether there are significant performance differences based on handedness.

Methodology:
- For each participant, we identify their preferred writing hand from metadata
- We classify sensor locations as dominant or non-dominant based on handedness
- We compare accuracy and F1 scores between the two groups

Author: Benchmarking Script
Date: 2025-11
===============================================================================
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns


def load_handedness_metadata(metadata_path=None):
    """
    Load participant handedness metadata from PAAWS_Data_Summary.csv.

    Parameters
    ----------
    metadata_path : str, optional
        Path to the metadata CSV file. If None, uses default location.

    Returns
    -------
    dict
        Dictionary mapping participant IDs to handedness ("Right" or "Left")
    """

    if metadata_path is None:
        metadata_path = Path(__file__).parent / "data" / "PAAWS_Data_Summary.csv"

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    # Read the metadata
    df = pd.read_csv(metadata_path)

    # Create mapping of participant ID to handedness
    handedness_map = {}
    for _, row in df.iterrows():
        dataset_id = row["DATASET"]
        handedness = row["PREFERRED_WRITING_HAND"]

        # Extract numeric ID (e.g., "DS_10" -> 10)
        if isinstance(dataset_id, str) and dataset_id.startswith("DS_"):
            participant_id = int(dataset_id.split("_")[1])
            if handedness in ["Right", "Left"]:
                handedness_map[participant_id] = handedness

    print(f"Loaded handedness metadata for {len(handedness_map)} participants")
    return handedness_map


def classify_sensor_handedness(sensor_location, participant_handedness):
    """
    Classify whether a sensor is on the dominant or non-dominant hand.

    Parameters
    ----------
    sensor_location : str
        Sensor location (e.g., "LeftWrist", "RightWrist")
    participant_handedness : str
        Participant's preferred writing hand ("Right" or "Left")

    Returns
    -------
    str
        "Dominant", "Non-Dominant", or "Unknown"
    """

    sensor_location = sensor_location.lower()

    if "left" in sensor_location and "wrist" in sensor_location:
        sensor_side = "Left"
    elif "right" in sensor_location and "wrist" in sensor_location:
        sensor_side = "Right"
    else:
        return "Unknown"

    if sensor_side == participant_handedness:
        return "Dominant"
    else:
        return "Non-Dominant"


def analyze_rft_results(results_dir, handedness_map, sensor_location="LeftWrist"):
    """
    Analyze RFT model results comparing dominant vs non-dominant hand performance.

    Parameters
    ----------
    results_dir : str or Path
        Directory containing RFT LOPO result CSV files
    handedness_map : dict
        Dictionary mapping participant IDs to handedness
    sensor_location : str
        The sensor location used in the experiments

    Returns
    -------
    dict
        Dictionary containing comparison results
    """

    results_dir = Path(results_dir)

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    # Find all CSV result files (format: *_DS_XX.csv)
    result_files = list(results_dir.glob("*_DS_*.csv"))

    if len(result_files) == 0:
        raise ValueError(f"No result files found in {results_dir}")

    print(f"\nFound {len(result_files)} result files")

    # Collect results by handedness classification
    dominant_results = []
    nondominant_results = []
    unknown_results = []

    for result_file in result_files:
        # Extract participant ID from filename
        filename = result_file.stem
        participant_id = int(filename.split("_DS_")[-1])

        # Get participant handedness
        if participant_id not in handedness_map:
            print(f"Warning: No handedness data for participant {participant_id}")
            continue

        participant_handedness = handedness_map[participant_id]

        # Classify sensor handedness
        sensor_handedness = classify_sensor_handedness(sensor_location, participant_handedness)

        # Load predictions
        df = pd.read_csv(result_file)
        true_labels = df["MAPPED_LABEL"].values
        predictions = df["PREDICTION"].values

        # Compute metrics
        accuracy = accuracy_score(true_labels, predictions)
        f1_macro = f1_score(true_labels, predictions, average="macro", zero_division=0)
        f1_weighted = f1_score(true_labels, predictions, average="weighted", zero_division=0)

        result_entry = {
            "Participant_ID": participant_id,
            "Participant_Handedness": participant_handedness,
            "Sensor_Location": sensor_location,
            "Sensor_Handedness": sensor_handedness,
            "Accuracy": accuracy,
            "F1_Macro": f1_macro,
            "F1_Weighted": f1_weighted,
            "Num_Samples": len(true_labels)
        }

        # Categorize by sensor handedness
        if sensor_handedness == "Dominant":
            dominant_results.append(result_entry)
        elif sensor_handedness == "Non-Dominant":
            nondominant_results.append(result_entry)
        else:
            unknown_results.append(result_entry)

        print(f"DS_{participant_id} ({participant_handedness}-handed, {sensor_handedness}): "
              f"Acc={accuracy:.4f}, F1-Macro={f1_macro:.4f}")

    # Convert to DataFrames
    dominant_df = pd.DataFrame(dominant_results)
    nondominant_df = pd.DataFrame(nondominant_results)

    # Compute aggregate statistics
    results = {
        "dominant_df": dominant_df,
        "nondominant_df": nondominant_df,
        "dominant_mean_accuracy": dominant_df["Accuracy"].mean() if len(dominant_df) > 0 else 0,
        "dominant_mean_f1_macro": dominant_df["F1_Macro"].mean() if len(dominant_df) > 0 else 0,
        "dominant_mean_f1_weighted": dominant_df["F1_Weighted"].mean() if len(dominant_df) > 0 else 0,
        "nondominant_mean_accuracy": nondominant_df["Accuracy"].mean() if len(nondominant_df) > 0 else 0,
        "nondominant_mean_f1_macro": nondominant_df["F1_Macro"].mean() if len(nondominant_df) > 0 else 0,
        "nondominant_mean_f1_weighted": nondominant_df["F1_Weighted"].mean() if len(nondominant_df) > 0 else 0,
        "dominant_count": len(dominant_df),
        "nondominant_count": len(nondominant_df),
        "sensor_location": sensor_location
    }

    return results


def plot_handedness_comparison(results, output_dir):
    """
    Create visualizations comparing dominant vs non-dominant performance.

    Parameters
    ----------
    results : dict
        Results dictionary from analyze_rft_results
    output_dir : str or Path
        Directory to save the plots
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)

    # Combine data for plotting
    dominant_df = results["dominant_df"].copy()
    nondominant_df = results["nondominant_df"].copy()

    if len(dominant_df) == 0 and len(nondominant_df) == 0:
        print("Warning: No data to plot")
        return

    dominant_df["Hand_Type"] = "Dominant"
    nondominant_df["Hand_Type"] = "Non-Dominant"

    combined_df = pd.concat([dominant_df, nondominant_df], ignore_index=True)

    # Plot 1: Accuracy comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Accuracy
    ax = axes[0]
    sns.boxplot(data=combined_df, x="Hand_Type", y="Accuracy", ax=ax)
    sns.stripplot(data=combined_df, x="Hand_Type", y="Accuracy", ax=ax,
                  color="black", alpha=0.5, size=8)
    ax.set_title("Accuracy by Hand Type", fontsize=14, fontweight='bold')
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_xlabel("")

    # Add mean lines
    if len(dominant_df) > 0:
        ax.axhline(y=results["dominant_mean_accuracy"], color='blue', linestyle='--',
                   alpha=0.5, label=f'Dominant Mean: {results["dominant_mean_accuracy"]:.3f}')
    if len(nondominant_df) > 0:
        ax.axhline(y=results["nondominant_mean_accuracy"], color='orange', linestyle='--',
                   alpha=0.5, label=f'Non-Dom Mean: {results["nondominant_mean_accuracy"]:.3f}')
    ax.legend(fontsize=9)

    # F1-Macro
    ax = axes[1]
    sns.boxplot(data=combined_df, x="Hand_Type", y="F1_Macro", ax=ax)
    sns.stripplot(data=combined_df, x="Hand_Type", y="F1_Macro", ax=ax,
                  color="black", alpha=0.5, size=8)
    ax.set_title("F1-Macro by Hand Type", fontsize=14, fontweight='bold')
    ax.set_ylabel("F1-Macro Score", fontsize=12)
    ax.set_xlabel("")

    if len(dominant_df) > 0:
        ax.axhline(y=results["dominant_mean_f1_macro"], color='blue', linestyle='--',
                   alpha=0.5, label=f'Dominant Mean: {results["dominant_mean_f1_macro"]:.3f}')
    if len(nondominant_df) > 0:
        ax.axhline(y=results["nondominant_mean_f1_macro"], color='orange', linestyle='--',
                   alpha=0.5, label=f'Non-Dom Mean: {results["nondominant_mean_f1_macro"]:.3f}')
    ax.legend(fontsize=9)

    # F1-Weighted
    ax = axes[2]
    sns.boxplot(data=combined_df, x="Hand_Type", y="F1_Weighted", ax=ax)
    sns.stripplot(data=combined_df, x="Hand_Type", y="F1_Weighted", ax=ax,
                  color="black", alpha=0.5, size=8)
    ax.set_title("F1-Weighted by Hand Type", fontsize=14, fontweight='bold')
    ax.set_ylabel("F1-Weighted Score", fontsize=12)
    ax.set_xlabel("")

    if len(dominant_df) > 0:
        ax.axhline(y=results["dominant_mean_f1_weighted"], color='blue', linestyle='--',
                   alpha=0.5, label=f'Dominant Mean: {results["dominant_mean_f1_weighted"]:.3f}')
    if len(nondominant_df) > 0:
        ax.axhline(y=results["nondominant_mean_f1_weighted"], color='orange', linestyle='--',
                   alpha=0.5, label=f'Non-Dom Mean: {results["nondominant_mean_f1_weighted"]:.3f}')
    ax.legend(fontsize=9)

    plt.suptitle(f"Handedness Comparison - {results['sensor_location']} Sensor",
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save plot
    plot_path = output_dir / "handedness_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to: {plot_path}")
    plt.close()


def save_comparison_report(results, output_dir):
    """
    Save a text report of the handedness comparison.

    Parameters
    ----------
    results : dict
        Results dictionary from analyze_rft_results
    output_dir : str or Path
        Directory to save the report
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "handedness_comparison_report.txt"

    with open(report_path, "w") as f:
        f.write("="*80 + "\n")
        f.write("Handedness Comparison Report\n")
        f.write("="*80 + "\n\n")

        f.write(f"Sensor Location: {results['sensor_location']}\n")
        f.write(f"Dominant Hand Participants: {results['dominant_count']}\n")
        f.write(f"Non-Dominant Hand Participants: {results['nondominant_count']}\n\n")

        f.write("="*80 + "\n")
        f.write("AGGREGATE METRICS\n")
        f.write("="*80 + "\n\n")

        f.write("Dominant Hand Performance:\n")
        f.write(f"  Mean Accuracy: {results['dominant_mean_accuracy']:.4f}\n")
        f.write(f"  Mean F1-Macro: {results['dominant_mean_f1_macro']:.4f}\n")
        f.write(f"  Mean F1-Weighted: {results['dominant_mean_f1_weighted']:.4f}\n\n")

        f.write("Non-Dominant Hand Performance:\n")
        f.write(f"  Mean Accuracy: {results['nondominant_mean_accuracy']:.4f}\n")
        f.write(f"  Mean F1-Macro: {results['nondominant_mean_f1_macro']:.4f}\n")
        f.write(f"  Mean F1-Weighted: {results['nondominant_mean_f1_weighted']:.4f}\n\n")

        # Compute differences
        acc_diff = results['dominant_mean_accuracy'] - results['nondominant_mean_accuracy']
        f1_macro_diff = results['dominant_mean_f1_macro'] - results['nondominant_mean_f1_macro']
        f1_weighted_diff = results['dominant_mean_f1_weighted'] - results['nondominant_mean_f1_weighted']

        f.write("="*80 + "\n")
        f.write("PERFORMANCE DIFFERENCES (Dominant - Non-Dominant)\n")
        f.write("="*80 + "\n\n")

        f.write(f"Accuracy Difference: {acc_diff:+.4f}")
        if acc_diff > 0:
            f.write(" (Dominant hand performs better)\n")
        elif acc_diff < 0:
            f.write(" (Non-dominant hand performs better)\n")
        else:
            f.write(" (No difference)\n")

        f.write(f"F1-Macro Difference: {f1_macro_diff:+.4f}")
        if f1_macro_diff > 0:
            f.write(" (Dominant hand performs better)\n")
        elif f1_macro_diff < 0:
            f.write(" (Non-dominant hand performs better)\n")
        else:
            f.write(" (No difference)\n")

        f.write(f"F1-Weighted Difference: {f1_weighted_diff:+.4f}")
        if f1_weighted_diff > 0:
            f.write(" (Dominant hand performs better)\n")
        elif f1_weighted_diff < 0:
            f.write(" (Non-dominant hand performs better)\n")
        else:
            f.write(" (No difference)\n")

        f.write("\n" + "="*80 + "\n")
        f.write("INDIVIDUAL PARTICIPANT RESULTS\n")
        f.write("="*80 + "\n\n")

        f.write("Dominant Hand Participants:\n")
        if len(results['dominant_df']) > 0:
            f.write(results['dominant_df'].to_string(index=False))
        else:
            f.write("  (none)\n")

        f.write("\n\nNon-Dominant Hand Participants:\n")
        if len(results['nondominant_df']) > 0:
            f.write(results['nondominant_df'].to_string(index=False))
        else:
            f.write("  (none)\n")

        f.write("\n\n" + "="*80 + "\n")
        f.write("INTERPRETATION NOTES\n")
        f.write("="*80 + "\n\n")

        f.write("This analysis compares activity recognition performance between sensors\n")
        f.write("placed on the dominant hand versus the non-dominant hand.\n\n")

        f.write("Key considerations:\n")
        f.write("1. Dominant hand: Typically the preferred writing hand (right or left)\n")
        f.write("2. Sample size: Small sample sizes may limit statistical significance\n")
        f.write("3. Activity types: Performance may vary by activity (e.g., writing vs walking)\n")
        f.write("4. Individual variation: People use their hands differently\n\n")

        if results['dominant_count'] < 3 or results['nondominant_count'] < 3:
            f.write("WARNING: Very small sample size detected. Results may not be reliable.\n")
            f.write("Consider running experiments with more participants for robust conclusions.\n")

    print(f"Saved comparison report to: {report_path}")


def main():
    """
    Main function for handedness comparison.
    """

    print("""
    ===============================================================================
    Handedness Comparison Analysis
    ===============================================================================
    This script compares model performance between dominant and non-dominant hand
    sensor placements.

    Requirements:
    1. Completed model evaluation results (from evaluate_rft_model.py or similar)
    2. Handedness metadata (PAAWS_Data_Summary.csv)
    3. Results from experiments using wrist sensors (LeftWrist or RightWrist)
    ===============================================================================
    """)

    # Load handedness metadata
    metadata_path = Path(__file__).parent / "data" / "PAAWS_Data_Summary.csv"
    try:
        handedness_map = load_handedness_metadata(metadata_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease ensure PAAWS_Data_Summary.csv is in the data/ directory")
        return

    # Get results directory from user
    results_dir_input = Path(__file__).parent / "benchmark_results" / "rft_lopo"

    if results_dir_input:
        results_dir = Path(results_dir_input)
    else:
        results_dir = default_results_dir

    # Get sensor location
    sensor_location = "LeftWrist"

    # Analyze results
    try:
        print(f"\n{'='*80}")
        print("Analyzing Results")
        print(f"{'='*80}")

        results = analyze_rft_results(results_dir, handedness_map, sensor_location)

        # Display summary
        print(f"\n{'='*80}")
        print("HANDEDNESS COMPARISON SUMMARY")
        print(f"{'='*80}")
        print(f"\nDominant Hand (n={results['dominant_count']}):")
        print(f"  Accuracy: {results['dominant_mean_accuracy']:.4f}")
        print(f"  F1-Macro: {results['dominant_mean_f1_macro']:.4f}")
        print(f"  F1-Weighted: {results['dominant_mean_f1_weighted']:.4f}")

        print(f"\nNon-Dominant Hand (n={results['nondominant_count']}):")
        print(f"  Accuracy: {results['nondominant_mean_accuracy']:.4f}")
        print(f"  F1-Macro: {results['nondominant_mean_f1_macro']:.4f}")
        print(f"  F1-Weighted: {results['nondominant_mean_f1_weighted']:.4f}")

        print(f"\nDifference (Dominant - Non-Dominant):")
        print(f"  Accuracy: {results['dominant_mean_accuracy'] - results['nondominant_mean_accuracy']:+.4f}")
        print(f"  F1-Macro: {results['dominant_mean_f1_macro'] - results['nondominant_mean_f1_macro']:+.4f}")
        print(f"  F1-Weighted: {results['dominant_mean_f1_weighted'] - results['nondominant_mean_f1_weighted']:+.4f}")
        print(f"{'='*80}")

        # Create output directory
        output_dir = Path(__file__).parent / "benchmark_results" / "handedness_comparison"

        # Generate visualizations
        print("\nGenerating visualizations...")
        plot_handedness_comparison(results, output_dir)

        # Save report
        print("Saving comparison report...")
        save_comparison_report(results, output_dir)

        print("\n✓ Handedness comparison complete!")
        print(f"\nResults saved to: {output_dir}")

    except Exception as e:
        print(f"\n✗ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
