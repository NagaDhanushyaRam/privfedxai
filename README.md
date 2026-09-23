# PrivFedXAI

A privacy-preserving explainable federated learning framework for clinical decision support.
COSC 880 Capstone, Towson University, Fall 2026. Advisor: Dr. Wassila Lalouani.

## Status
Sprint 1: Environment setup and exploratory data analysis (in progress)

## Setup
1. Install Anaconda or Miniconda.
2. Create and activate a conda environment:
```bash
   conda create -n privfedxai python=3.11
   conda activate privfedxai
```
   Install all packages below with pip, not conda.
3. Install PyTorch for your hardware (pip option) from https://pytorch.org/get-started/locally/
4. Install remaining dependencies:
```bash
   python -m pip install -r requirements.txt
```
5. Verify:
```bash
   python check_env.py
```

## Dataset
PTB-XL (PhysioNet). Download instructions will be added in Sprint 1.
Raw data is not committed to this repository.

## Repository structure
```
src/data/        data pipeline and site-based client partitioning
src/models/      1D-CNN + attention model
src/federated/   Flower client/server (FedAvg)
src/privacy/     Opacus DP-SGD wrapper
src/explain/     attention heatmaps, SHAP, IoU alignment
src/eval/        metrics (F1, AUC, MCC) and per-class fairness
notebooks/       EDA
results/         figures and tables
```