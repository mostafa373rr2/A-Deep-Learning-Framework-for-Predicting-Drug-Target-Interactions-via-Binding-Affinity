# A Deep Learning Framework for Predicting Drug-Target Interactions via Binding Affinity

A comprehensive PyTorch deep learning framework designed to predict **Drug-Target Interactions (DTI)** and binding affinity ($pK_d$) from molecular SMILES sequences and target protein amino acid sequences.

---

## 🌟 Key Highlights & Architecture

The framework leverages a multi-modal deep learning architecture composed of:
1. **Drug Feature Representation**: Transformer Encoder architecture processing SMILES representations with positional encoding and multi-head self-attention.
2. **Protein Sequence Representation**: Multi-scale  Convolutional Neural Network (CNN) with adaptive pooling over target protein amino acid sequences.
3. **Binding Affinity Decoder**: Multi-Layer Perceptron (MLP) decoder fusing latent drug-target feature representations to predict continuous binding affinity ($pK_d$).
4. **Data Normalization & Stability**: Automated conversion of raw binding affinity $Y$ ($\text{nM}$) into $-\log_{10}(Y \times 10^{-9})$ ($pK_d$) to resolve non-linear distributions and optimize $R^2$ regression performance.
5. **Hyperparameter Tuning**: Automated Bayesian optimization via **Optuna** for optimal architectural dimensions, learning rates, and regularization.

```
   Drug (SMILES)               Target (Amino Acid Sequence)
        │                                    │
        ▼                                    ▼
┌──────────────────┐               ┌──────────────────┐
│   Transformer    │               │      CNN       │
│     Encoder      │               │     Encoder      │
└────────┬─────────┘               └────────┬─────────┘
         │                                  │
         └─────────────► ⨁ ◄────────────────┘
                   (Concatenation)
                          │
                          ▼
               ┌───────────────────────┐
               │      MLP Decoder      │
               └──────────┬────────────┘
                          │
                          ▼
            Predicted Binding Affinity (pKd)
```

---

## 📁 Repository Structure

```
.
├── 01_data_preprocessor.py                     # Cleans raw BindingDB, converts nM to pKd, and outputs clean CSVs
├── 02_drug_target_models.py                    # Transformer, CNN, and MLP model architectures
├── 03_smiles_to_fingerprint.py                 # Molecular fingerprinting & SMILES sequence processing
├── 04_training_pipeline_MODIFIED.py            # Baseline PyTorch training and evaluation loop
├── 05_training_pipeline_enhanced_MODIFIED.py   # Enhanced training pipeline with schedulers, metrics, & early stopping
├── 06_hyperparameter_optimization_MODIFIED.py  # Optuna hyperparameter optimization engine
├── TRAIN_WORKFLOW_2DAY_OPTIMIZED.py            # Automated multi-stage training routine
├── baseline_reproduction_WORKING.py            # Script for reproducing baseline performance
├── model_comparison_tool.py                    # Multi-model benchmarking and radar chart generator
├── quick_comparison.py                         # Fast evaluation and metrics aggregator
├── run_comparison_now.py                       # Automated benchmark execution script
├── requirements.txt                            # Project dependencies
├── data/                                       # Raw dataset storage (e.g. BindingDB.csv)
├── data_processed/                             # Preprocessed datasets and distribution plots
├── models_saved/                               # Serialized model checkpoints (.pt)
├── plots/                                      # Loss curves, regression plots, and parameter importance
├── result/                                     # Intermediate loss curves and validation tables
└── results/                                    # Optimization metrics, best_config.json, and evaluation sheets
```

---

## 🚀 Getting Started

### 1. Prerequisites & Environment Setup

Clone this repository and create a virtual environment:

```bash
git clone https://github.com/mostafa373rr2/A-Deep-Learning-Framework-for-Predicting-Drug-Target-Interactions-via-Binding-Affinity.git
cd A-Deep-Learning-Framework-for-Predicting-Drug-Target-Interactions-via-Binding-Affinity

# Create and activate virtual environment
python -m venv venv

# Windows (Command Prompt / PowerShell)
.\venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔬 Pipeline Workflow

### Step 1: Preprocess Data
Converts raw binding values (nM) into normalized $pK_d$ and cleans missing/invalid entries:
```bash
python 01_data_preprocessor.py
```

### Step 2: Run Baseline Training
Trains the standard Transformer-CNN-MLP model on the preprocessed dataset:
```bash
python 04_training_pipeline_MODIFIED.py
```

### Step 3: Run Enhanced Training Pipeline
Includes learning rate scheduling, advanced metrics ($R^2$, RMSE, Pearson correlation), and automated checkpointing:
```bash
python 05_training_pipeline_enhanced_MODIFIED.py
```

### Step 4: Hyperparameter Optimization
Executes Bayesian optimization using Optuna to find the best architectural parameters:
```bash
python 06_hyperparameter_optimization_MODIFIED.py
```

### Step 5: Model Evaluation & Benchmarking
Generate comparative metrics, radar charts, and Excel summaries across model variants:
```bash
python run_comparison_now.py
```

---

## 📊 Evaluation Metrics

The framework evaluates model prediction accuracy across standard regression metrics:
* **$R^2$ Score** (Coefficient of Determination)
* **RMSE** (Root Mean Squared Error)
* **MAE** (Mean Absolute Error)
* **Pearson Correlation ($r$)**
* **Spearman Correlation ($\rho$)**

---

## 📄 License
This project is licensed under the MIT License.
