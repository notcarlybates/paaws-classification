# SLURM Scripts for PAAWS Benchmarking

## Setup

```bash
cd paaws-classification
mkdir -p logs
```

Edit each script to uncomment your Python environment activation method.

## Submit Jobs

```bash
cd slurm_scripts

# Run RFT evaluation (DS_10 and DS_36)
sbatch run_rft_evaluation.sh

# Run handedness comparison
sbatch run_handedness.sh

# Run LSTM evaluation
sbatch run_lstm_evaluation.sh

# Run single participant (for parallel jobs)
sbatch --export=DS_LO=10 run_single_participant.sh
sbatch --export=DS_LO=36 run_single_participant.sh
```

## Interactive Commands

```bash
# RFT evaluation
srun --partition=short --nodes=1 --cpus-per-task=16 --mem=64G --time=02:00:00 --pty \
    python evaluate_rft_model.py

# Single participant (from paaws-benchmarking)
cd ../paaws-benchmarking
srun --partition=short --nodes=1 --cpus-per-task=16 --mem=32G --time=01:00:00 --pty \
    python run_experiment.py --ds_lo=10 --dataset="2" --sensor="LeftWrist" \
    --mapping="lab_fl_5" --out_file="../paaws-classification/benchmark_results/rft_lopo/" --lab

# Handedness comparison
srun --partition=short --nodes=1 --cpus-per-task=2 --mem=8G --time=00:30:00 --pty \
    python compare_handedness.py

# LSTM evaluation
srun --partition=gpu --nodes=1 --cpus-per-task=4 --gres=gpu:1 --mem=32G --time=04:00:00 --pty \
    python evaluate_lstm_model.py
```

## Monitor Jobs

```bash
squeue -u $USER
tail -f logs/rft_*.out
```
