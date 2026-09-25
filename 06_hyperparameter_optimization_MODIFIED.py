"""
🎯 Hyperparameter Optimization with Optuna
Find the best hyperparameters for your model (SAME ARCHITECTURE!)

This script will:
1. ✅ Search for optimal hyperparameters
2. ✅ Test different configurations automatically
3. ✅ Save the best configuration
4. ✅ Visualize optimization process

NO architecture changes - just finding best parameters!
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tqdm import tqdm
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import matplotlib.pyplot as plt
import argparse
import sys
import warnings
warnings.filterwarnings('ignore')

# Import modules
import importlib.util

def import_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

try:
    models_module = import_module_from_file("drug_target_models", "02_drug_target_models.py")
    DrugTargetInteractionModel = models_module.DrugTargetInteractionModel
    
    smiles_module = import_module_from_file("smiles_converter", "03_smiles_to_fingerprint.py")
    batch_smiles_to_fingerprints = smiles_module.batch_smiles_to_fingerprints
    
    training_module = import_module_from_file("training_enhanced", "05_training_pipeline_enhanced_MODIFIED.py")
    ProteinSequenceEncoder = training_module.ProteinSequenceEncoder
    DTIDataset = training_module.DTIDataset
    batch_enhanced_fingerprints = training_module.batch_enhanced_fingerprints
    clean_data = training_module.clean_data
    
except Exception as e:
    print(f"ERROR: Could not import required modules: {e}")
    sys.exit(1)


# Global data storage for Optuna
GLOBAL_DATA = {}


def prepare_data_once(args):
    """Prepare data once and store globally"""
    print("=" * 70)
    print("PREPARING DATA (ONE TIME)")
    print("=" * 70)
    
    # Load and clean data
    df = pd.read_csv(args.data_path)
    
    # ⚡ CRITICAL: Verify data is in pKd scale
    print("\nVerifying data preprocessing...")
    y_min, y_max = df['Y'].min(), df['Y'].max()
    print(f"Y range: [{y_min:.2f}, {y_max:.2f}]")
    
    if y_max > 100:
        print("\n❌ ERROR: Data NOT preprocessed!")
        print("Run: python 01_data_preprocessor.py --input YOUR_RAW_DATA.csv")
        sys.exit(1)
    
    print("✅ Data in pKd scale - proceeding...\n")

    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    df_clean = clean_data(df)
    
    # Sample
    if args.sample_size:
        df_clean = df_clean.sample(n=min(args.sample_size, len(df_clean)), random_state=42)
    
    print(f"\nUsing {len(df_clean)} samples for optimization")
    
    # Create encoder
    protein_encoder = ProteinSequenceEncoder()
    
    # Prepare features
    print("\nGenerating molecular features...")
    drug_features = batch_enhanced_fingerprints(
        df_clean['Drug'].tolist(),
        radius=2,
        n_bits=1024
    )
    
    print("\nEncoding protein sequences...")
    target_features = []
    for seq in tqdm(df_clean['Target'], desc="Encoding"):
        encoded = protein_encoder.encode(seq, max_len=1000)
        target_features.append(encoded)
    target_features = np.array(target_features)
    
    labels = df_clean['Y'].values
    
    # Split data
    indices = np.arange(len(drug_features))
    train_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.15, random_state=42)
    
    # Store in global dict
    GLOBAL_DATA['drug_features'] = drug_features
    GLOBAL_DATA['target_features'] = target_features
    GLOBAL_DATA['labels'] = labels
    GLOBAL_DATA['train_idx'] = train_idx
    GLOBAL_DATA['val_idx'] = val_idx
    GLOBAL_DATA['test_idx'] = test_idx
    GLOBAL_DATA['feature_dim'] = drug_features.shape[1]
    
    print("\n✅ Data prepared and stored")
    print(f"   Feature dimension: {drug_features.shape[1]}")
    print(f"   Train samples: {len(train_idx)}")
    print(f"   Val samples: {len(val_idx)}")
    print(f"   Test samples: {len(test_idx)}")


def create_dataloaders(batch_size):
    """Create dataloaders from global data"""
    drug_features = GLOBAL_DATA['drug_features']
    target_features = GLOBAL_DATA['target_features']
    labels = GLOBAL_DATA['labels']
    train_idx = GLOBAL_DATA['train_idx']
    val_idx = GLOBAL_DATA['val_idx']
    
    # Create tensors
    train_drug = torch.FloatTensor(drug_features[train_idx])
    train_target = torch.LongTensor(target_features[train_idx])
    train_label = torch.FloatTensor(labels[train_idx])
    
    val_drug = torch.FloatTensor(drug_features[val_idx])
    val_target = torch.LongTensor(target_features[val_idx])
    val_label = torch.FloatTensor(labels[val_idx])
    
    # Create datasets
    train_data = DTIDataset({'drug': train_drug, 'target': train_target, 'label': train_label})
    val_data = DTIDataset({'drug': val_drug, 'target': val_target, 'label': val_label})
    
    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader


def objective(trial):
    """Optuna objective function"""
    
    # Suggest hyperparameters
    config = {
        'drug_input_dim': GLOBAL_DATA['feature_dim'],
        'drug_d_model': trial.suggest_categorical('d_model', [64, 128, 256]),
        'drug_num_layers': trial.suggest_int('num_layers', 4, 10),
        'drug_num_heads': trial.suggest_categorical('num_heads', [4, 8, 16]),
        'drug_d_ff': trial.suggest_categorical('d_ff', [256, 512, 1024]),
        'drug_hidden_dim': 256,
        'target_vocab_size': 26,
        'target_embedding_dim': 128,
        'target_num_filters': [32, 64, 96],
        'target_kernel_sizes': [4, 8, 12],
        'target_hidden_dim': 256,
        'decoder_hidden_dims': [
            trial.suggest_categorical('decoder_dim1', [512, 1024, 2048]),
            trial.suggest_categorical('decoder_dim2', [256, 512, 1024]),
            trial.suggest_categorical('decoder_dim3', [128, 256, 512])
        ],
        'dropout': trial.suggest_uniform('dropout', 0.1, 0.4),
        'binary': False
    }
    
    # Training hyperparameters
    lr = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    
    print(f"\nTrial {trial.number}: Testing configuration...")
    print(f"  d_model={config['drug_d_model']}, layers={config['drug_num_layers']}, "
          f"heads={config['drug_num_heads']}, lr={lr:.6f}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(batch_size)
    
    # Create model
    model = DrugTargetInteractionModel(**config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Train for limited epochs
    max_epochs = 20  # Quick training for optimization
    best_val_r2 = -float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(max_epochs):
        # Training
        model.train()
        train_loss = 0
        for drug, target, label in train_loader:
            drug = drug.to(device)
            target = target.to(device)
            label = label.to(device)
            
            optimizer.zero_grad()
            pred = model(drug, target)
            loss = criterion(pred, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for drug, target, label in val_loader:
                drug = drug.to(device)
                target = target.to(device)
                pred = model(drug, target)
                val_preds.extend(pred.cpu().numpy().flatten())
                val_labels.extend(label.numpy().flatten())
        
        # Calculate R²
        val_r2 = r2_score(val_labels, val_preds)
        
        # Report intermediate value
        trial.report(val_r2, epoch)
        
        # Handle pruning
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        # Track best
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    print(f"  Best R²: {best_val_r2:.4f}")
    return best_val_r2


def run_optimization(args):
    """Run hyperparameter optimization"""
    
    print("\n" + "=" * 70)
    print("HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
    print("=" * 70)
    
    # Prepare data once
    prepare_data_once(args)
    
    # Create Optuna study
    study = optuna.create_study(
        direction='maximize',
        study_name='dti_optimization',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )
    
    print(f"\nStarting optimization with {args.n_trials} trials...")
    print("This will take some time. Progress will be shown below.\n")
    
    # Optimize
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
    
    # Results
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE!")
    print("=" * 70)
    
    print(f"\nBest Trial: {study.best_trial.number}")
    print(f"Best R²: {study.best_value:.4f}")
    
    print("\nBest Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key:20s}: {value}")
    
    # Save results
    results_df = study.trials_dataframe()
    results_df.to_csv(r'E:\DTI_env\results\optuna_results.csv', index=False)
    print("\n✅ Results saved to: optuna_results.csv")
    
    # Visualizations
    print("\nGenerating visualizations...")
    
    # Optimization history
    fig = plot_optimization_history(study)
    fig.write_image(r'E:\DTI_env\plots\optimization_history.png')
    print("  ✅ Saved: optimization_history.png")
    
    # Parameter importances
    try:
        fig = plot_param_importances(study)
        fig.write_image(r'E:\DTI_env\plots\param_importances.png')
        print("  ✅ Saved: param_importances.png")
    except:
        print("  ⚠️ Could not create param_importances plot (need more trials)")
    
    # Create config file for best parameters
    best_config = {
        'drug_input_dim': GLOBAL_DATA['feature_dim'],
        'drug_d_model': study.best_params['d_model'],
        'drug_num_layers': study.best_params['num_layers'],
        'drug_num_heads': study.best_params['num_heads'],
        'drug_d_ff': study.best_params['d_ff'],
        'drug_hidden_dim': 256,
        'target_vocab_size': 26,
        'target_embedding_dim': 128,
        'target_num_filters': [32, 64, 96],
        'target_kernel_sizes': [4, 8, 12],
        'target_hidden_dim': 256,
        'decoder_hidden_dims': [
            study.best_params['decoder_dim1'],
            study.best_params['decoder_dim2'],
            study.best_params['decoder_dim3']
        ],
        'dropout': study.best_params['dropout'],
        'learning_rate': study.best_params['learning_rate'],
        'batch_size': study.best_params['batch_size']
    }
    
    # Save config
    import json
    with open(r'E:\DTI_env\results\best_config.json', 'w') as f:
        json.dump(best_config, f, indent=4)
    print("  ✅ Saved: best_config.json")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("\n1. Use the best hyperparameters from best_config.json")
    print("2. Train full model with these parameters:")
    print(f"\n   python 05_training_pipeline_enhanced.py \\")
    print(f"       --d_model {best_config['drug_d_model']} \\")
    print(f"       --num_layers {best_config['drug_num_layers']} \\")
    print(f"       --num_heads {best_config['drug_num_heads']} \\")
    print(f"       --d_ff {best_config['drug_d_ff']} \\")
    print(f"       --dropout {best_config['dropout']:.4f} \\")
    print(f"       --learning_rate {best_config['learning_rate']:.6f} \\")
    print(f"       --batch_size {best_config['batch_size']}")
    
    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparameter Optimization for DTI Model')
    
    parser.add_argument('--data_path', type=str, 
                       default=r'E:\DTI_env\data_processed\BindingDB_processed.csv',
                       help='Path to BindingDB CSV')
    parser.add_argument('--sample_size', type=int, default=10000,
                       help='Number of samples for optimization (smaller = faster)')
    parser.add_argument('--n_trials', type=int, default=50,
                       help='Number of optimization trials')
    parser.add_argument('--batch_size', type=int, default=None)

    args = parser.parse_args()
    
    study = run_optimization(args)
