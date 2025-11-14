# PAAWS Benchmarking - Quick Start

## What This Does

Compare Potter's RFT model vs. Mazzuchelli's LSTM model on PAAWS data, then analyze handedness effects.

## Setup

```bash
cd paaws-classification
mkdir -p logs data
```

## Complete Workflow (3 Steps)

### Step 1: Preprocess Data for LSTM

```bash
# This uses the same PAAWS data paths as paaws-benchmarking
python lstm_preprocess.py
```

**What it does:** Reads raw PAAWS data and creates CSV for LSTM model

**Output:** `data/preprocessed_paaws_subset.csv`

---

### Step 2: Run Model Evaluations

#### Option A: Batch Jobs (Recommended)

```bash
cd slurm_scripts

# Run both models
sbatch run_rft_evaluation.sh
sbatch run_lstm_evaluation.sh

# Then run handedness comparison
sbatch run_handedness.sh
```

#### Option B: Interactive

```bash
cd paaws-classification

# RFT evaluation
srun --partition=short --cpus-per-task=16 --mem=64G --time=02:00:00 --pty \
    python evaluate_rft_model.py

# LSTM evaluation
srun --partition=short --cpus-per-task=8 --mem=32G --time=04:00:00 --pty \
    python evaluate_lstm_model.py

# Handedness comparison
srun --partition=short --cpus-per-task=2 --mem=8G --time=00:30:00 --pty \
    python compare_handedness.py
```

---

### Step 3: Compare Results

**RFT results:**
```bash
cat benchmark_results/rft_lopo/rft_aggregate_metrics.txt
```

**LSTM results:**
```bash
cat ../paaws_activity_detection/loso_results/summary.csv
```

**Handedness analysis:**
```bash
cat benchmark_results/handedness_comparison/handedness_comparison_report.txt
```

---

## What You Get

### 1. Preprocessed Data
- **File:** `data/preprocessed_paaws_subset.csv`
- **Format:** `subject_id, timestamp, x, y, z, label`
- **Use:** Input for LSTM model

### 2. RFT Model Results
- **Location:** `benchmark_results/rft_lopo/`
- **Key file:** `rft_aggregate_metrics.txt`
- **Metrics:** Accuracy, F1-Macro, F1-Weighted
- **Example:**
  ```
  Overall Accuracy: 0.8542
  Overall F1-Macro: 0.8234
  Overall F1-Weighted: 0.8456
  ```

### 3. LSTM Model Results
- **Location:** `../paaws_activity_detection/loso_results/`
- **Key file:** `summary.csv`
- **Metrics:** Same as RFT
- **Example:**
  ```
  Subject,Accuracy,F1_Macro,F1_Weighted
  DS_10,0.8234,0.7956,0.8145
  DS_36,0.8567,0.8301,0.8489
  Average,0.8401,0.8129,0.8317
  ```

### 4. Model Comparison
Compare the "Average" row from both models:

| Metric | RFT | LSTM | Better |
|--------|-----|------|--------|
| Accuracy | 0.8542 | 0.8401 | RFT +0.0141 |
| F1-Macro | 0.8234 | 0.8129 | RFT +0.0105 |
| F1-Weighted | 0.8456 | 0.8317 | RFT +0.0139 |

### 5. Handedness Analysis
- **Location:** `benchmark_results/handedness_comparison/`
- **Files:** Report (txt) + Visualization (png)
- **Shows:** Dominant vs. non-dominant hand performance

---

## Configuration

Current settings (same for both models):
- **Participants:** DS_10, DS_36
- **Sensor:** LeftWrist
- **Activities:** lab_fl_5 (Sitting, Standing, Walking, Biking, Lying_Down)
- **Protocol:** SimFL+Lab

**To change:**
- Edit `lstm_preprocess.py` lines 294-296
- Edit `evaluate_rft_model.py` lines 233-236

---

## Direct Potter/Mazzuchelli Scripts

### Run Potter's RFT script directly:

```bash
cd paaws-benchmarking

# DS_10
srun --partition=short --cpus-per-task=16 --mem=32G --time=01:00:00 --pty \
    python run_experiment.py --ds_lo=10 --dataset="2" --sensor="LeftWrist" \
    --mapping="lab_fl_5" --out_file="../paaws-classification/benchmark_results/rft_lopo/" --lab

# DS_36
srun --partition=short --cpus-per-task=16 --mem=32G --time=01:00:00 --pty \
    python run_experiment.py --ds_lo=36 --dataset="2" --sensor="LeftWrist" \
    --mapping="lab_fl_5" --out_file="../paaws-classification/benchmark_results/rft_lopo/" --lab
```

### Run Mazzuchelli's LSTM script directly:

```bash
cd paaws_activity_detection

srun --partition=short --cpus-per-task=8 --mem=32G --time=04:00:00 --pty \
    python scripts/train_loso.py \
    --dataset=../paaws-classification/data/preprocessed_paaws_subset.csv \
    --model_type=lstm \
    --exclude_unknown
```

---

## Troubleshooting

### "Data not found" (lstm_preprocess.py)

**Issue:** Can't find PAAWS data files

**Fix:** Edit `lstm_preprocess.py` line 292:
```python
data_base_path = "/absolute/path/to/PAAWS/data/"
```

### "Module not found"

**Fix:**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn torch pyyaml tqdm
```

---

## Need Help?

- **Quick reference:** This file
- **Methodology:** `BENCHMARKING_INSTRUCTIONS.md`
- **General info:** `README_BENCHMARKING.md`
- **SLURM details:** `slurm_scripts/README.md`
