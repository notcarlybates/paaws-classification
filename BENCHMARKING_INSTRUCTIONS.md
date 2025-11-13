# PAAWS Model Benchmarking Instructions

## Overview

This document provides comprehensive instructions for benchmarking activity recognition models on the PAAWS dataset. The benchmarking process includes:

1. **Task 1**: Evaluating Potter's Random Forest (RFT) model and Mazzuchelli's LSTM model on a small data subset
2. **Task 2**: Comparing performance between dominant vs. non-dominant hand sensor placements

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Methodology Overview](#methodology-overview)
3. [Task 1: Model Evaluation](#task-1-model-evaluation)
   - [1A: RFT Model Evaluation](#1a-rft-model-evaluation)
   - [1B: LSTM Model Evaluation](#1b-lstm-model-evaluation)
4. [Task 2: Handedness Comparison](#task-2-handedness-comparison)
5. [Interpreting Results](#interpreting-results)
6. [Troubleshooting](#troubleshooting)
7. [Methodological Notes](#methodological-notes)

---

## Prerequisites

### Required Repositories

Ensure you have all three repositories in your workspace:

```
C:\Users\Carly\Desktop\github\
├── paaws-benchmarking/          # Potter's RFT model
├── paaws_activity_detection/    # Mazzuchelli's LSTM model
└── paaws-classification/         # Benchmarking scripts + metadata
```

### Required Data

1. **Raw PAAWS Data**: The PAAWS SimFL+Lab accelerometer dataset
   - Potter's RFT model will access this directly via `paaws-benchmarking`
   - Ensure data paths in `paaws-benchmarking` are configured correctly

2. **Handedness Metadata**: Already available at:
   ```
   paaws-classification/data/PAAWS_Data_Summary.csv
   ```

3. **Preprocessed Data for LSTM** (if evaluating LSTM model):
   - Format required: CSV with columns `x, y, z, label, subject_id`
   - See [LSTM Data Preparation](#lstm-data-preparation) section

### Python Environment

Required packages:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn torch pyyaml tqdm
```

---

## Methodology Overview

### Evaluation Protocol: Leave-One-Participant-Out (LOPO) Cross-Validation

Both models use **LOPO cross-validation**, which:
- Trains on N-1 participants
- Tests on the remaining 1 participant
- Repeats for all participants
- Computes aggregate metrics across all test folds

**Why LOPO?** This protocol evaluates generalization to unseen individuals, which is critical for real-world deployment.

### Test Subset: Dataset "2"

For initial testing, we use a small subset:
- **Participants**: DS_10, DS_36
- **Handedness**: DS_10 (Right-handed), DS_36 (Left-handed)
- **Purpose**: Quick validation before running full experiments

### Metrics Computed

1. **Accuracy**: Overall classification accuracy
2. **F1-Macro**: Unweighted average F1 score across classes (treats all classes equally)
3. **F1-Weighted**: Weighted average F1 score (accounts for class imbalance)

**Why these metrics?**
- **Accuracy**: Simple, interpretable overall performance
- **F1-Macro**: Important when classes have different sample sizes
- **F1-Weighted**: Better represents performance on imbalanced datasets

### Activity Mapping

Both models will use the **"lab_fl_5"** mapping scheme:
- **5 Activities**: Sitting, Standing, Walking, Biking, Lying_Down
- **Purpose**: Ensures fair comparison between models

### Methodological Improvements

**Changes Made from Standard Protocols:**

1. **Consistent Evaluation**: Both models use same participants, same activity mapping
2. **Handedness Awareness**: Explicitly account for dominant/non-dominant hand in analysis
3. **Standardized Metrics**: Same metric computation for both models
4. **Clear Reproducibility**: All scripts document exact parameters used

---

## Task 1: Model Evaluation

### 1A: RFT Model Evaluation

The RFT (Random Forest) model by Potter et al. uses handcrafted features from accelerometer data.

#### Step 1: Verify Data Access

The RFT scripts need access to raw PAAWS data. Check that data paths are configured:

```bash
cd paaws-benchmarking
```

Look for data paths in:
- `get_and_clean_data.py`
- Your local environment setup

#### Step 2: Run RFT Evaluation

Navigate to the classification repository:

```bash
cd C:\Users\Carly\Desktop\github\paaws-classification
```

Run the evaluation script:

```bash
python evaluate_rft_model.py
```

**What This Script Does:**

1. Runs LOPO cross-validation on participants DS_10 and DS_36
2. For each participant:
   - Trains Random Forest on the other participant's data
   - Tests on the held-out participant
   - Saves predictions to CSV
3. Computes aggregate metrics:
   - Per-participant accuracy, F1-macro, F1-weighted
   - Overall aggregate metrics
   - Confusion matrix

**Expected Runtime**: 5-15 minutes per participant (depends on CPU)

#### Step 3: Locate RFT Results

Results are saved to:
```
paaws-classification/benchmark_results/rft_lopo/
├── rft_per_participant_results.csv       # Per-participant metrics
├── rft_aggregate_metrics.txt             # Summary statistics
├── rft_confusion_matrix.csv              # Confusion matrix
└── temp_ds10_*/                          # Individual LOPO results
```

#### Step 4: Review RFT Results

Check the aggregate metrics file:

```bash
cat benchmark_results/rft_lopo/rft_aggregate_metrics.txt
```

Expected format:
```
================================================================================
RFT Model - Aggregate Metrics (LOPO Cross-Validation)
================================================================================

Overall Accuracy: 0.8542
Overall F1-Macro: 0.8234
Overall F1-Weighted: 0.8456
...
```

---

### 1B: LSTM Model Evaluation

The LSTM model by Mazzuchelli et al. uses deep learning on raw accelerometer sequences.

#### Important Note: Data Format Challenge

⚠️ **The LSTM model requires preprocessed data in a specific format that differs from the RFT model's data structure.**

The LSTM model expects a CSV file with:
- `x, y, z`: Raw accelerometer values
- `label`: Activity label (string)
- `subject_id`: Participant identifier (e.g., "DS_10", "DS_36")
- `timestamp`: (optional) Timestamp of measurement

#### LSTM Data Preparation

If you have access to the raw PAAWS accelerometer files, you'll need to:

1. **Extract accelerometer data** for DS_10 and DS_36
2. **Match with activity labels** from the PAAWS label files
3. **Format into CSV** with required columns
4. **Save** as `preprocessed_paaws_subset.csv`

**Example data format:**
```csv
subject_id,timestamp,x,y,z,label
DS_10,1234567890,-0.123,0.456,9.781,Walking
DS_10,1234567891,-0.125,0.458,9.783,Walking
...
DS_36,1234567890,0.234,-0.123,9.802,Sitting
```

**Data Preparation Script (Pseudocode):**

```python
# This is a template - adapt to your specific data structure
import pandas as pd

def prepare_lstm_data(participant_ids, output_path):
    """
    Convert PAAWS data to LSTM format.
    """
    all_data = []

    for pid in participant_ids:
        # Load accelerometer data for this participant
        accel_data = load_actigraph_file(f"PAAWS_Data/DS_{pid}/accel.csv")

        # Load corresponding labels
        labels = load_label_file(f"PAAWS_Data/DS_{pid}/labels.csv")

        # Merge accelerometer data with labels by timestamp
        merged = merge_by_timestamp(accel_data, labels)

        # Add subject_id column
        merged['subject_id'] = f"DS_{pid}"

        # Keep only required columns
        merged = merged[['subject_id', 'timestamp', 'x', 'y', 'z', 'label']]

        all_data.append(merged)

    # Concatenate all participants
    final_df = pd.concat(all_data, ignore_index=True)

    # Save
    final_df.to_csv(output_path, index=False)
    print(f"Saved preprocessed data to: {output_path}")

# Run for test subset
prepare_lstm_data([10, 36], "preprocessed_paaws_subset.csv")
```

#### Step 1: Prepare LSTM Data

**Option A**: If you have raw PAAWS data:
1. Adapt the data preparation script above
2. Run it to generate `preprocessed_paaws_subset.csv`
3. Place it in `paaws-classification/data/`

**Option B**: If data is already preprocessed:
1. Locate your preprocessed CSV file
2. Verify it has the required columns
3. Note its path for the next step

#### Step 2: Run LSTM Evaluation

```bash
cd C:\Users\Carly\Desktop\github\paaws-classification
python evaluate_lstm_model.py
```

The script will:
1. Prompt you for the dataset path
2. Ask which LSTM variant to use (`lstm`, `cnn_lstm`, or `attentive_lstm`)
3. Run LOSO cross-validation
4. Save results to `paaws_activity_detection/loso_results/`

**Expected Runtime**: 10-30 minutes per participant (depends on GPU/CPU, epochs)

#### Step 3: Locate LSTM Results

Results are saved to:
```
paaws_activity_detection/loso_results/
├── summary.csv                           # Per-subject and aggregate metrics
├── per_class_metrics.csv                 # Per-class performance
├── aggregated_confusion_matrix.png       # Confusion matrix
└── all_results.pkl                       # Raw results (for advanced analysis)
```

#### Step 4: Review LSTM Results

Check the summary:

```bash
cat ../paaws_activity_detection/loso_results/summary.csv
```

Expected format:
```csv
Subject,Accuracy,F1_Macro,F1_Weighted
DS_10,0.8234,0.7956,0.8145
DS_36,0.8567,0.8301,0.8489
Average,0.8401,0.8129,0.8317
```

---

### 1C: Comparing RFT vs LSTM Results

After completing both evaluations, create a comparison table:

| Metric | RFT Model | LSTM Model | Difference |
|--------|-----------|------------|------------|
| Accuracy | [Fill from rft_aggregate_metrics.txt] | [Fill from LSTM summary.csv] | Δ = |
| F1-Macro | [Fill from rft_aggregate_metrics.txt] | [Fill from LSTM summary.csv] | Δ = |
| F1-Weighted | [Fill from rft_aggregate_metrics.txt] | [Fill from LSTM summary.csv] | Δ = |

**Interpretation Guidelines:**

- **Difference < 0.02**: Models perform similarly
- **Difference 0.02-0.05**: Moderate performance difference
- **Difference > 0.05**: Substantial performance difference

**Questions to Consider:**

1. Which model achieves higher overall accuracy?
2. Does one model perform better on specific activity classes?
3. Are there trade-offs between macro and weighted F1 scores?
4. How do the confusion matrices differ?

---

## Task 2: Handedness Comparison

**Goal**: Determine if there are performance differences between dominant and non-dominant hand sensor placements.

### Prerequisite: Run Sensor-Specific Evaluations

To compare handedness, you need results from **both wrist locations**:

#### Step 1: Run Left Wrist Evaluation

Already completed in Task 1A (default sensor is LeftWrist).

#### Step 2: Run Right Wrist Evaluation

Modify the RFT evaluation to use RightWrist sensor:

**Manual Approach:**

Edit `evaluate_rft_model.py` line ~249:

```python
# Change from:
sensor = "LeftWrist"

# To:
sensor = "RightWrist"
```

Then run:
```bash
python evaluate_rft_model.py
```

**OR use Potter's script directly:**

```bash
cd ../paaws-benchmarking

# For DS_10 with RightWrist
python run_experiment.py --ds_lo=10 --dataset="2" --sensor="RightWrist" \
       --mapping="lab_fl_5" --lab --out_file="results/right_wrist/"

# For DS_36 with RightWrist
python run_experiment.py --ds_lo=36 --dataset="2" --sensor="RightWrist" \
       --mapping="lab_fl_5" --lab --out_file="results/right_wrist/"
```

### Step 3: Run Handedness Comparison

Navigate back to classification directory:

```bash
cd C:\Users\Carly\Desktop\github\paaws-classification
```

For **LeftWrist** analysis:
```bash
python compare_handedness.py
```

When prompted:
- Results directory: `benchmark_results/rft_lopo` (or press Enter for default)
- Sensor location: `LeftWrist` (or press Enter for default)

For **RightWrist** analysis, repeat with:
- Sensor location: `RightWrist`
- Results directory: `path/to/right_wrist_results/`

### Step 4: Interpret Handedness Results

The script generates:

1. **Text Report**: `benchmark_results/handedness_comparison/handedness_comparison_report.txt`
   - Aggregate statistics by hand type (dominant vs. non-dominant)
   - Performance differences
   - Individual participant breakdowns

2. **Visualization**: `benchmark_results/handedness_comparison/handedness_comparison.png`
   - Boxplots comparing metrics by hand type
   - Individual data points overlaid
   - Mean lines for reference

**Key Metrics to Examine:**

```
Dominant Hand Performance:
  Mean Accuracy: 0.8542
  Mean F1-Macro: 0.8234
  Mean F1-Weighted: 0.8456

Non-Dominant Hand Performance:
  Mean Accuracy: 0.8301
  Mean F1-Macro: 0.7956
  Mean F1-Weighted: 0.8187

Difference (Dominant - Non-Dominant):
  Accuracy: +0.0241 (Dominant hand performs better)
  F1-Macro: +0.0278 (Dominant hand performs better)
  F1-Weighted: +0.0269 (Dominant hand performs better)
```

**Interpretation:**

- **Positive difference**: Dominant hand has better recognition accuracy
- **Negative difference**: Non-dominant hand performs better
- **Near-zero difference** (|Δ| < 0.01): No meaningful difference

**Important Caveats:**

1. **Small Sample Size**: With only 2 participants, statistical significance is limited
2. **Activity-Specific Effects**: Some activities (e.g., writing) may show larger differences than others (e.g., walking)
3. **Individual Variation**: People differ in how they use their dominant vs. non-dominant hand

---

## Interpreting Results

### Understanding the Metrics

#### Accuracy
- **Definition**: Proportion of correct predictions
- **Formula**: (True Positives + True Negatives) / Total Samples
- **Interpretation**: Higher is better; 0.85 = 85% correct
- **Limitation**: Can be misleading with class imbalance

#### F1-Macro Score
- **Definition**: Unweighted average of per-class F1 scores
- **Use Case**: When all activity classes are equally important
- **Interpretation**: Treats rare and common activities equally
- **Best for**: Balanced evaluation across activities

#### F1-Weighted Score
- **Definition**: Weighted average of per-class F1 scores (by support)
- **Use Case**: When some activities are more common
- **Interpretation**: Emphasizes performance on frequent activities
- **Best for**: Real-world performance estimation

#### Confusion Matrix
- **Purpose**: Shows which activities are confused with each other
- **How to Read**: Rows = true label, Columns = predicted label
- **Diagonal**: Correct predictions
- **Off-diagonal**: Misclassifications

**Example Interpretation:**
```
            Sitting  Standing  Walking  Biking  Lying_Down
Sitting         120         5        2       0           8
Standing          3        95        7       0           0
Walking           1        10      145       3           0
Biking            0         0        5      78           0
Lying_Down        6         1        0       0          98
```

Observations:
- Sitting sometimes confused with Lying_Down (8 cases)
- Walking occasionally misclassified as Standing (10 cases)
- Biking well-recognized (78/83 correct)

### Statistical Significance

**Important**: With only 2 participants, you **cannot** make strong statistical claims.

To increase confidence:
- Expand to more participants (e.g., full "SimFL_20" dataset: 20 participants)
- Use statistical tests (e.g., paired t-test) with larger sample sizes
- Report confidence intervals

### Reporting Results

When presenting results, include:

1. **Configuration**:
   - Model(s) evaluated
   - Dataset size and composition
   - Evaluation protocol (LOPO)
   - Activity classes included

2. **Aggregate Metrics**:
   - Overall accuracy, F1-macro, F1-weighted
   - Per-class performance (especially for problematic classes)
   - Confusion matrix

3. **Handedness Analysis** (if performed):
   - Number of participants in each group
   - Mean metrics for dominant vs. non-dominant
   - Magnitude and direction of differences

4. **Caveats**:
   - Sample size limitations
   - Any data quality issues
   - Generalization limitations

---

## Troubleshooting

### Common Issues

#### Issue 1: "Data directory not found"

**Symptoms**: Script cannot find PAAWS data files

**Solution**:
1. Check data path configuration in `paaws-benchmarking/get_and_clean_data.py`
2. Ensure raw PAAWS data is downloaded and extracted
3. Update paths to match your local setup

**Example Fix**:
```python
# In get_and_clean_data.py, update:
DATA_ROOT = "C:/Users/Carly/Desktop/PAAWS_Data"  # Update this path
```

#### Issue 2: "Module not found" errors

**Symptoms**: `ImportError: No module named 'torch'` or similar

**Solution**:
```bash
# Activate your Python environment
# Install missing packages
pip install torch pandas numpy scikit-learn matplotlib seaborn pyyaml tqdm
```

#### Issue 3: LSTM data format errors

**Symptoms**: "KeyError: 'subject_id'" or "Column not found: label"

**Solution**:
1. Verify your CSV has all required columns: `x, y, z, label, subject_id`
2. Check column names match exactly (case-sensitive)
3. Ensure no missing values in key columns

**Verification Script**:
```python
import pandas as pd

df = pd.read_csv("your_data.csv")
print("Columns:", df.columns.tolist())
print("Sample:\n", df.head())
print("Subject IDs:", df['subject_id'].unique())
print("Labels:", df['label'].unique())
print("Missing values:", df.isnull().sum())
```

#### Issue 4: Memory errors during training

**Symptoms**: "Out of memory" or "MemoryError"

**Solution**:
1. Reduce batch size in config files
2. Process participants one at a time
3. Use memory-efficient data loading (already implemented in RFT script)

#### Issue 5: Slow execution

**Symptoms**: Scripts take much longer than expected

**For RFT Model**:
- Random Forest uses CPU: ensure `n_jobs=-1` is set (uses all cores)
- Expected: 5-15 min per participant on modern CPU

**For LSTM Model**:
- Check if PyTorch is using GPU: `torch.cuda.is_available()`
- If no GPU: Reduce epochs or use simpler model variant
- Expected: 10-30 min per participant (GPU), 1-2 hours (CPU)

---

## Methodological Notes

### Why This Approach?

This benchmarking design addresses several methodological issues:

#### 1. Fair Model Comparison
- **Issue**: Different models use different data formats and preprocessing
- **Solution**: Use same participants, same activity mapping, same evaluation protocol
- **Benefit**: Differences in metrics reflect model capability, not data differences

#### 2. Handedness Awareness
- **Issue**: Prior work rarely accounts for hand dominance
- **Solution**: Explicit analysis of dominant vs. non-dominant hand performance
- **Benefit**: Reveals potential biases in sensor placement

#### 3. Small Subset Testing
- **Issue**: Full dataset experiments are time-consuming
- **Solution**: Use 2-participant subset for initial validation
- **Benefit**: Quick iteration and debugging before full experiments

#### 4. Reproducibility
- **Issue**: Many papers lack clear implementation details
- **Solution**: Documented scripts with explicit parameters
- **Benefit**: Others can reproduce and build on your work

### Limitations and Future Work

#### Current Limitations

1. **Small Sample Size**: 2 participants insufficient for strong conclusions
   - **Impact**: Cannot assess statistical significance
   - **Mitigation**: Expand to full SimFL_20 (20 participants) or larger

2. **Single Sensor Location Per Experiment**: Comparing left vs. right wrist requires separate runs
   - **Impact**: More time and effort required
   - **Mitigation**: Could batch process all sensor locations

3. **Data Format Incompatibility**: LSTM and RFT models use different data formats
   - **Impact**: Extra preprocessing needed for LSTM
   - **Mitigation**: Could create unified data pipeline

4. **Activity Mapping**: Fixed 5-activity grouping
   - **Impact**: Doesn't test performance on fine-grained activities
   - **Mitigation**: Could expand to 9-activity or 42-activity mappings

#### Recommended Extensions

1. **Expand Participant Set**:
   ```bash
   # In evaluate_rft_model.py, change:
   dataset = "SimFL_20"  # 20 participants instead of 2
   ```

2. **Multi-Sensor Analysis**:
   - Run experiments for all sensor locations: LeftWrist, RightWrist, LeftAnkle, RightAnkle, etc.
   - Compare performance across body locations
   - Investigate sensor fusion (combining multiple sensors)

3. **Fine-Grained Activities**:
   - Use "lab_fl_9" (9 activities) or "lab_42" (42 activities) mappings
   - Analyze which activities are most challenging
   - Identify common confusions

4. **Cross-Dataset Validation**:
   - Test on FreeLiving data (if available)
   - Assess generalization from lab to real-world conditions

5. **Statistical Analysis**:
   - With larger sample: paired t-tests, ANOVA
   - Bootstrap confidence intervals
   - Inter-subject variability analysis

### Best Practices for Reporting

When publishing results from these experiments:

1. **Clearly state limitations**:
   ```
   "Results are based on 2 participants from the PAAWS dataset.
   Larger-scale validation is needed to confirm findings."
   ```

2. **Provide implementation details**:
   - Model hyperparameters
   - Data preprocessing steps
   - Evaluation protocol
   - Random seeds (for reproducibility)

3. **Share confusion matrices and per-class metrics**:
   - Aggregate metrics can hide class-specific issues
   - Confusion matrices reveal systematic errors

4. **Discuss generalization**:
   - LOSO tests person-independent performance
   - But: Limited age range, demographics in test set

---

## Next Steps

### Immediate Actions

- [ ] Run RFT evaluation on DS_10 and DS_36
- [ ] Record RFT results (accuracy, F1 scores)
- [ ] (Optional) Prepare LSTM data and run LSTM evaluation
- [ ] Compare RFT and LSTM results
- [ ] Run handedness comparison with LeftWrist data
- [ ] (Optional) Repeat with RightWrist data
- [ ] Interpret and document findings

### For More Robust Results

- [ ] Expand to full SimFL_20 dataset (20 participants)
- [ ] Run evaluations on multiple sensor locations
- [ ] Test different activity mappings (5, 9, 42 activities)
- [ ] Perform statistical significance testing
- [ ] Validate on FreeLiving data (if available)

### For Publication

- [ ] Create publication-quality figures
- [ ] Write detailed methods section
- [ ] Prepare supplementary materials with:
  - Full confusion matrices
  - Per-participant results
  - Hyperparameter configurations
- [ ] Make code and data publicly available (if permitted)

---

## Summary

This benchmarking framework provides:

✅ **Standardized evaluation** of RFT and LSTM models on PAAWS data
✅ **Handedness analysis** to identify sensor placement effects
✅ **Clear methodology** with documented scripts and instructions
✅ **Reproducible results** with explicit parameters and configurations

The scripts handle:
- LOPO cross-validation for both models
- Aggregate metric computation (accuracy, F1-macro, F1-weighted)
- Confusion matrix generation
- Dominant vs. non-dominant hand comparison
- Result visualization and reporting

**Remember**: Start with the small 2-participant subset for quick validation, then expand to larger participant sets for robust conclusions.

---

## Questions?

If you encounter issues not covered in this document:

1. Check script comments for additional details
2. Review model-specific documentation:
   - `paaws-benchmarking/README.md` (RFT model)
   - `paaws_activity_detection/README.md` (LSTM model)
3. Verify data paths and file formats
4. Ensure all required packages are installed

Good luck with your benchmarking experiments!
