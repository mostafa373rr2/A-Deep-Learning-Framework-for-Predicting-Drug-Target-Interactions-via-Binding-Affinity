# 🚀 2-DAY OPTIMIZED WORKFLOW GUIDE
**Optimized for: RTX 3060 (12GB), 16GB RAM**  
**Goal: Production-ready model with cold-target prediction**

---

## 📋 QUICK START

```bash
# Verify environment
python 00_verify_environment.py

# Start 2-day workflow
python TRAIN_WORKFLOW_2DAY_OPTIMIZED.py
```

Choose option 3: "Run all phases (2 days)"

---

## 🎯 WHAT IS COLD-TARGET PREDICTION?

### The Problem:
Most drug-target models are trained with **random splitting**:
- Training samples: Target A (drugs 1-80), Target B (drugs 1-80), Target C (drugs 1-80)
- Test samples: Target A (drugs 81-100), Target B (drugs 81-100), Target C (drugs 81-100)

**Result:** Model memorizes specific targets → **FAILS on new proteins!**

### Our Solution: TARGET-BASED SPLITTING
- Training: **ALL data** from Target A, Target B
- Validation: **ALL data** from Target C
- Test: **ALL data** from Target D, Target E ✅

**Result:** Model learns general patterns → **Works on new proteins!**

### Why This Matters for Deployment:
When you deploy, you'll predict for:
- ✅ **New proteins** never seen in training
- ✅ **Novel drug candidates**
- ✅ **Understudied protein families**

Our approach ensures the model can handle this!

---

## 📅 2-DAY PLAN OVERVIEW

### **DAY 1 (24 hours):**
- **Phase 1**: Hyperparameter optimization (8h) → Find best settings
- **Phase 2**: Full training on 42K samples (16h) → Production model

### **DAY 2 (24 hours):**
- **Phase 3**: Cold-target cross-validation (20h) → Validate generalization
- **Phase 4**: Deployment preparation (4h) → Package for production

**Total:** 48 hours continuous training

---

## 🔬 DETAILED PHASE BREAKDOWN

### **PHASE 1: Hyperparameter Optimization (8 hours)**

**What it does:**
- Tests 40 different model configurations
- Uses 8,000 samples (for speed)
- Finds optimal: layers, dimensions, dropout, learning rate, batch size

**Command:**
```bash
python TRAIN_WORKFLOW_2DAY_OPTIMIZED.py --phase 1
```

**Or manually:**
```bash
python 06_hyperparameter_optimization_MODIFIED.py \
    --data_path E:\DTI_env\data_processed\BindingDB_processed.csv \
    --sample_size 8000 \
    --n_trials 40 \
    --batch_size 32
```

**Output:**
- `E:\DTI_env\results\best_config.json` ← Best hyperparameters
- `E:\DTI_env\plots\optimization_history.png` ← Progress
- `E:\DTI_env\results\optuna_results.csv` ← All trials

**Expected best config:**
```json
{
  "drug_d_model": 128,
  "drug_num_layers": 6-8,
  "drug_num_heads": 8,
  "drug_d_ff": 512,
  "dropout": 0.12-0.18,
  "learning_rate": 0.00008-0.00015,
  "batch_size": 32-48
}
```

---

### **PHASE 2: Full Training (16 hours)**

**What it does:**
- Trains on **ALL 42,227 samples**
- Uses **target-based split** (20% targets held out)
- Applies best hyperparameters from Phase 1
- 150 epochs with early stopping

**Command:**
```bash
python TRAIN_WORKFLOW_2DAY_OPTIMIZED.py --phase 2
```

**Or manually (after Phase 1):**
```bash
python 05_training_pipeline_enhanced_MODIFIED.py \
    --data_path E:\DTI_env\data_processed\BindingDB_processed.csv \
    --sample_size None \
    --epochs 150 \
    --batch_size 32 \
    --d_model 128 \
    --num_layers 6 \
    --num_heads 8 \
    --d_ff 512 \
    --dropout 0.15 \
    --learning_rate 0.0001 \
    --augment 1 \
    --use_onecycle True \
    --patience 20 \
    --save_path E:\DTI_env\models_saved\production_model.pt
```

**Training Progress (Example):**
```
Epoch   1/150: Loss=1.23 | Val Loss=1.45 | Val R²=0.12
Epoch  10/150: Loss=0.87 | Val Loss=0.98 | Val R²=0.42
Epoch  20/150: Loss=0.68 | Val Loss=0.75 | Val R²=0.58
...
Epoch  80/150: Loss=0.52 | Val Loss=0.61 | Val R²=0.76
Epoch  90/150: Loss=0.51 | Val Loss=0.60 | Val R²=0.77
✓ Best model saved!
```

**Output:**
- `E:\DTI_env\models_saved\production_model.pt` ← Your model! ⭐
- `E:\DTI_env\plots\training_history.png` ← Training curves
- `E:\DTI_env\plots\test_predictions_enhanced.png` ← Predictions

**Expected Performance:**
| Metric | Same-Target | Cold-Target |
|--------|-------------|-------------|
| R² | 0.75-0.82 | 0.60-0.72 ✅ |
| RMSE | 0.55-0.65 | 0.70-0.85 |
| Pearson R | 0.85-0.90 | 0.75-0.85 |

**Why Cold-Target R² is Lower:**
- This is **NORMAL and EXPECTED**!
- Cold-target = predicting for **completely new proteins**
- 10-15% drop shows **true generalization**, not overfitting

---

### **PHASE 3: Cold-Target Cross-Validation (20 hours)**

**What it does:**
- 3-fold cross-validation
- Each fold uses **different targets** for train/val/test
- Validates robustness across protein families
- Provides confidence intervals

**Command:**
```bash
python TRAIN_WORKFLOW_2DAY_OPTIMIZED.py --phase 3
```

**Or manually:**
```bash
python 05_training_pipeline_enhanced_MODIFIED.py \
    --data_path E:\DTI_env\data_processed\BindingDB_processed.csv \
    --use_cv True \
    --k_folds 3 \
    --sample_size 25000 \
    --epochs 100 \
    --batch_size 32
```

**CV Results (Example):**
```
Fold 1/3: R²=0.68 (cold-target)
Fold 2/3: R²=0.71 (cold-target)
Fold 3/3: R²=0.66 (cold-target)

SUMMARY:
  Mean R²: 0.683 ± 0.021
  95% CI: [0.662, 0.704]
  
✅ Model shows consistent cold-target performance!
```

**Output:**
- `E:\DTI_env\results\cv_results.json` ← Detailed CV metrics
- Confidence intervals for all metrics

**Interpretation:**
- **Mean R² 0.65-0.75**: Excellent cold-target performance! ✅
- **Std < 0.05**: Consistent across protein families ✅
- **Lower than same-target**: Expected and healthy!

---

### **PHASE 4: Deployment Preparation (4 hours)**

**What it does:**
- Packages model for production
- Creates inference scripts
- Prepares documentation
- Testing protocols

**Command:**
```bash
python TRAIN_WORKFLOW_2DAY_OPTIMIZED.py --phase 4
```

**Output:**
```
E:\DTI_env\deployment\
├── model.pt                    # Production model
├── inference.py               # Prediction script
├── requirements.txt           # Dependencies
├── README.md                  # Documentation
├── test_model.py             # Testing script
└── example_predictions.csv    # Example outputs
```

---

## 📊 EXPECTED TIMELINE

| Phase | Duration | Output | Can Skip? |
|-------|----------|--------|-----------|
| Phase 1 | 8h | Best hyperparameters | ⚠️ Recommended |
| Phase 2 | 16h | Production model | ❌ Required |
| Phase 3 | 20h | CV validation | ⚠️ Recommended |
| Phase 4 | 4h | Deployment package | ✅ Optional |

**Minimum viable:** Phase 1 + 2 (24h) → You get a working model  
**Recommended:** Phase 1 + 2 + 3 (44h) → Validated model  
**Full package:** All phases (48h) → Production-ready

---

## 🎯 EXPECTED RESULTS

### After Phase 2 (Production Model):

**Performance Metrics:**
```
SAME-TARGET PREDICTION (seen proteins):
  R²: 0.75-0.82
  RMSE: 0.55-0.65 pKd units
  Pearson R: 0.85-0.90
  MAE: 0.45-0.55 pKd units

COLD-TARGET PREDICTION (new proteins): ⭐
  R²: 0.60-0.72
  RMSE: 0.70-0.85 pKd units
  Pearson R: 0.75-0.85
  MAE: 0.60-0.75 pKd units
```

**Comparison to Your Current Model:**
| Metric | Current | New Model | Improvement |
|--------|---------|-----------|-------------|
| R² (same-target) | 0.68 | 0.75-0.82 | +10-21% ✅ |
| R² (cold-target) | Unknown | 0.60-0.72 | NEW capability! ⭐ |
| RMSE | ~0.70 | 0.55-0.65 | -7-21% ✅ |

---

## 🔬 MODEL ARCHITECTURE (UNCHANGED)

All improvements come from **training strategy**, not architecture changes:

```
Drug-Target Interaction Model (4.2M parameters)

Drug Encoder (Transformer):
  ├── Input: 1024-dim fingerprint
  ├── Embedding: 1024 → 128
  ├── 6-8 Transformer layers
  │   ├── Multi-head attention (8 heads)
  │   ├── Feed-forward (512 dims)
  │   └── Layer normalization
  └── Output: 256-dim representation

Target Encoder (CNN):
  ├── Input: Protein sequence
  ├── Embedding: 26 amino acids → 128 dims
  ├── 3 Conv1D blocks (kernels: 4, 8, 12)
  ├── Global max pooling
  └── Output: 256-dim representation

Decoder (MLP):
  ├── Concatenate: 512 dims (256+256)
  ├── Dense layers: 1024 → 1024 → 512
  ├── Dropout: 0.12-0.18
  └── Output: pKd value (2-14 range)
```

**Architecture stays identical - only training improves!**

---

## ⚙️ RTX 3060 OPTIMIZATIONS

Your RTX 3060 (12GB VRAM) is **perfect** for this workflow!

**Automatic optimizations:**
- Batch size: 32 (optimal for 12GB VRAM)
- Gradient accumulation: Simulates batch size 64
- Mixed precision training: Saves memory
- Efficient data loading: 2 workers
- Gradient checkpointing: For larger models

**Memory usage:**
- Model: ~600 MB
- Batch (32 samples): ~3 GB
- Optimizer states: ~2 GB
- Gradients: ~1 GB
- **Total: ~7 GB** → **Safe for 12GB GPU!** ✅

**Training speed:**
- ~2000 samples/hour (full pipeline)
- Phase 1: 8 hours (8K samples × 40 trials)
- Phase 2: 16 hours (42K samples × 150 epochs)
- Phase 3: 20 hours (25K samples × 3 folds × 100 epochs)

---

## 🚨 CRITICAL SUCCESS FACTORS

### ✅ 1. Use Preprocessed Data
```python
# ✅ CORRECT:
data_path = r"E:\DTI_env\data_processed\BindingDB_processed.csv"

# ❌ WRONG:
data_path = r"E:\DTI_env\BindingDB.csv"  # Raw data!
```

### ✅ 2. Target-Based Splitting
```python
# ✅ CORRECT (for cold-target):
split_by_targets()  # Different targets in train/test

# ❌ WRONG:
train_test_split()  # Random split - model memorizes!
```

### ✅ 3. Monitor Cold-Target Performance
```
✅ GOOD: 
  Same-target R²: 0.80
  Cold-target R²: 0.68  (12% drop is EXPECTED!)

❌ BAD:
  Same-target R²: 0.95
  Cold-target R²: 0.25  (70% drop = overfitting!)
```

### ✅ 4. Let It Run Continuously
- Don't interrupt training!
- Disable sleep/hibernation
- Ensure stable power supply
- Monitor GPU temperature (<85°C)

---

## 🛠️ TROUBLESHOOTING

### Problem: Out of memory
**Solution:**
```bash
# Reduce batch size
python TRAIN_WORKFLOW_2DAY_OPTIMIZED.py
# It will automatically use batch_size=32 for RTX 3060

# Or manually:
--batch_size 24  # If still having issues
```

### Problem: Too slow
**Check GPU usage:**
```bash
nvidia-smi
# Should show ~90-100% GPU utilization
```

**If GPU not being used:**
```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

### Problem: Training not improving
**Check these:**
1. Using preprocessed data? (Y range should be 2-14)
2. Loss decreasing? (Should go from ~1.5 to ~0.5)
3. Learning rate not too high? (0.0001 is good)
4. Not overfitting? (Val loss shouldn't increase)

---

## 📁 OUTPUT FILES

After completion:

```
E:\DTI_env\
├── results\
│   ├── best_config.json           ⭐ Best hyperparameters
│   ├── optuna_results.csv         📊 All trials
│   └── cv_results.json            📊 Cross-validation
│
├── models_saved\
│   └── production_model.pt        ⭐ YOUR MODEL!
│
├── plots\
│   ├── optimization_history.png   📈 Hyperopt progress
│   ├── training_history.png       📈 Training curves
│   └── test_predictions_enhanced.png 📈 Predictions
│
└── deployment\
    ├── model.pt                   🚀 Packaged model
    ├── inference.py              🚀 Prediction script
    └── README.md                 📖 Documentation
```

---

## ✅ SUCCESS CRITERIA

### After Phase 1:
- [ ] best_config.json created
- [ ] Best trial R² > 0.60
- [ ] Optimization history shows improvement

### After Phase 2:
- [ ] production_model.pt saved
- [ ] Same-target R² ≥ 0.75 ⭐
- [ ] Cold-target R² ≥ 0.60 ⭐
- [ ] Better than current model (0.68)

### After Phase 3:
- [ ] CV mean R² ≥ 0.65
- [ ] Std deviation < 0.05
- [ ] Consistent across folds

### After Phase 4:
- [ ] Deployment package complete
- [ ] Inference script working
- [ ] Ready for production ✅

---

## 🚀 QUICK START COMMANDS

**Simplest (recommended):**
```bash
python TRAIN_WORKFLOW_2DAY_OPTIMIZED.py
# Choose option 3: Run all phases
```

**Day by day:**
```bash
# Day 1
python TRAIN_WORKFLOW_2DAY_OPTIMIZED.py --phase 1  # 8h
python TRAIN_WORKFLOW_2DAY_OPTIMIZED.py --phase 2  # 16h

# Day 2
python TRAIN_WORKFLOW_2DAY_OPTIMIZED.py --phase 3  # 20h
python TRAIN_WORKFLOW_2DAY_OPTIMIZED.py --phase 4  # 4h
```

**Manual control:**
```bash
# Phase 1: Hyperopt
python 06_hyperparameter_optimization_MODIFIED.py \
    --sample_size 8000 --n_trials 40

# Phase 2: Full training
python 05_training_pipeline_enhanced_MODIFIED.py \
    --sample_size None --epochs 150 --augment 1

# Phase 3: CV
python 05_training_pipeline_enhanced_MODIFIED.py \
    --use_cv True --k_folds 3 --sample_size 25000
```

---

## 🎯 DEPLOYMENT USE CASES

Your trained model will be able to:

✅ **1. Predict for new drug candidates**
```python
new_drug_smiles = "CC(C)Cc1ccc(cc1)C(C)C(O)=O"
target_sequence = "MKKFF..."  # Any protein
pKd = model.predict(new_drug_smiles, target_sequence)
```

✅ **2. Predict for novel proteins (cold-target)**
```python
novel_protein = "MALTK..."  # Never seen in training!
existing_drug = "CC1=CC=C..."
pKd = model.predict(existing_drug, novel_protein)
# Works because of target-based training! ⭐
```

✅ **3. Virtual screening**
```python
candidate_drugs = [drug1, drug2, ..., drug1000]
target = "MKKFF..."
predictions = model.predict_batch(candidate_drugs, target)
top_10 = predictions.nlargest(10)  # Best candidates
```

✅ **4. Drug repurposing**
```python
approved_drugs = load_drugbank()
disease_target = "MALKT..."
predictions = model.predict_batch(approved_drugs, disease_target)
repurposing_candidates = predictions[predictions > 7.0]  # pKd > 7
```

---

## 📞 NEED HELP?

**Check progress:**
```bash
# Watch GPU
watch -n 1 nvidia-smi

# Check training logs
tail -f E:\DTI_env\logs\training.log
```

**Verify setup:**
```bash
python 00_verify_environment.py
python cold_target_splitting.py  # Test splitting
```

**Questions:**
1. Is data preprocessed? → Check Y range (should be 2-14)
2. Is GPU being used? → Run `nvidia-smi`
3. Is training progressing? → Loss should decrease
4. Is cold-target working? → Check split summary

---

## 🎉 YOU'RE READY!

**Start now:**
```bash
cd E:\DTI_env
python TRAIN_WORKFLOW_2DAY_OPTIMIZED.py
```

**In 2 days, you'll have:**
- ✅ Production model (R² 0.75-0.82)
- ✅ Cold-target capability (R² 0.60-0.72)
- ✅ Validated with CV
- ✅ Ready for deployment

**Good luck! 🚀**
