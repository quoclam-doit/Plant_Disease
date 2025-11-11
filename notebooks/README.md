# 📓 Jupyter Notebooks

## 📋 Available Notebooks (4 files)

### 1. Plant_Disease_EDA.ipynb

**Exploratory Data Analysis**

- Dataset overview and statistics
- Class distribution analysis
- Image visualization
- Data quality checks

### 2. Plant_Disease_MambaTSR.ipynb

**MambaTSR Training (Local)**

- Complete training pipeline
- Model architecture exploration
- Training visualization
- Results analysis

### 3. Plant_Disease_MambaTSR_Colab.ipynb

**MambaTSR Training (Google Colab)**

- Colab-optimized version
- Cloud GPU training
- Simplified setup

### 4. Plant_Disease_YOLOv4_Ensemble.ipynb

**YOLOv4 + Ensemble Methods**

- Object detection approach
- Ensemble model experiments
- Comparison with MambaTSR

## 🚀 Quick Start

### Open in Jupyter

```bash
# Activate environment
source .venv/bin/activate  # Linux/WSL
# or
.venv\Scripts\Activate.ps1  # Windows

# Start Jupyter
jupyter notebook
```

### Open in VS Code

- Install Jupyter extension
- Open `.ipynb` file
- Select kernel: `.venv`

## 🎯 Main Results (from MambaTSR notebook)

**Best Model Performance:**

- Validation Accuracy: **98.96%**
- Training Time: 3 hours
- Model Size: 843.72 MB (77M params)

## 📌 Related Files

Main Python script: `../train_mambatsr_plantvillage.py`  
Documentation: `../docs/`  
Trained models: `../models/MambaTSR/`
