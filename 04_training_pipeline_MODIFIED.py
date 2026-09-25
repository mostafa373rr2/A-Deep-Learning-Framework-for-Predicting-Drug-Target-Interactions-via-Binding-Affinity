"""
🚀 Training Drug-Target Interaction Model
Notebook 3: Complete Training Pipeline on BindingDB

تدريب النموذج على بيانات حقيقية

المحتوى:
1. ✅ Data Loading & Preprocessing
2. ✅ Protein Sequence Encoding  
3. ✅ Training Loop with Validation
4. ✅ Model Evaluation & Metrics
5. ✅ Save Best Model

Dataset: BindingDB (38,000+ drug-target pairs)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import sys
import argparse

# Import models and SMILES converter from previous scripts
# Python doesn't allow importing modules that start with numbers,
# so we use importlib to import them dynamically
import importlib.util

def import_module_from_file(module_name, file_path):
    """Import a module from a file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

try:
    # Import models from script 1
    models_module = import_module_from_file("drug_target_models", "02_drug_target_models.py")
    DrugTargetInteractionModel = models_module.DrugTargetInteractionModel
    
    # Import SMILES converter from script 2
    smiles_module = import_module_from_file("smiles_converter", "03_smiles_to_fingerprint.py")
    batch_smiles_to_fingerprints = smiles_module.batch_smiles_to_fingerprints
    
except Exception as e:
    print("ERROR: Could not import required modules!")
    print(f"Error: {e}")
    print("\nMake sure you have these files in the same directory:")
    print("  - 02_drug_target_models.py")
    print("  - 03_smiles_to_fingerprint.py")
    print("\nAlternatively, rename them without leading numbers:")
    print("  - drug_target_models.py")
    print("  - smiles_to_fingerprint.py")
    sys.exit(1)


# ============================================================================
# Protein Sequence Encoder
# ============================================================================

class ProteinSequenceEncoder:
    """Protein Sequence Encoder"""
    AMINO_ACIDS = ['<PAD>', '<UNK>'] + list('ACDEFGHIKLMNPQRSTVWY') + ['X', 'B', 'Z', 'J']
    
    def __init__(self):
        self.aa_to_idx = {aa: idx for idx, aa in enumerate(self.AMINO_ACIDS)}
        self.idx_to_aa = {idx: aa for aa, idx in self.aa_to_idx.items()}
        self.vocab_size = len(self.AMINO_ACIDS)
    
    def encode(self, sequence, max_len=1000):
        encoded = [self.aa_to_idx.get(aa, self.aa_to_idx['<UNK>']) for aa in sequence.upper()]
        
        if len(encoded) > max_len:
            encoded = encoded[:max_len]
        else:
            encoded = encoded + [self.aa_to_idx['<PAD>']] * (max_len - len(encoded))
        
        return np.array(encoded)
    
    def decode(self, indices):
        return ''.join([self.idx_to_aa.get(idx, '<UNK>') for idx in indices])


# ============================================================================
# Data Processing Functions
# ============================================================================

def clean_data(df):
    """Clean and prepare the dataset"""
    print("\nCleaning data...")
    print(f"Initial size: {len(df)}")
    
    # Remove NaN
    df = df.dropna()
    print(f"After removing NaN: {len(df)}")
    
    # Remove empty strings
    df = df[df['Drug'].str.len() > 0]
    df = df[df['Target'].str.len() > 0]
    print(f"After removing empty strings: {len(df)}")
    
    # Convert Y to pKd scale if needed
    if df['Y'].max() > 100:
        df['Y'] = -np.log10(df['Y'] * 1e-9)
        print("✅ Converted Y to pKd scale")
    
    # Remove outliers
    q1 = df['Y'].quantile(0.01)
    q99 = df['Y'].quantile(0.99)
    df = df[(df['Y'] >= q1) & (df['Y'] <= q99)]
    print(f"After removing outliers: {len(df)}")
    
    return df


def prepare_data(df, protein_encoder, sample_size=None, fp_size=1024, 
                 max_protein_len=1000, test_size=0.2, val_size=0.1):
    """Prepare data for training"""
    
    # Sample data if specified (for quick testing)
    if sample_size is not None:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
        print(f"Using {len(df)} samples")
    
    # Extract data
    smiles_list = df['Drug'].tolist()
    sequences = df['Target'].tolist()
    labels = df['Y'].values
    
    print(f"\n📊 Data summary:")
    print(f"Total samples: {len(labels)}")
    print(f"Label range: [{labels.min():.2f}, {labels.max():.2f}]")
    
    # Generate drug fingerprints
    print(f"\nGenerating drug fingerprints...")
    drug_features = batch_smiles_to_fingerprints(
        smiles_list,
        radius=2,
        n_bits=fp_size,
        verbose=True
    )
    print(f"✅ Drug features: {drug_features.shape}")
    print(f"   Average bits set: {np.mean(np.sum(drug_features, axis=1)):.1f}")
    
    # Encode protein sequences
    print(f"\nEncoding protein sequences...")
    target_features = np.array([
        protein_encoder.encode(seq, max_len=max_protein_len) 
        for seq in tqdm(sequences, desc="Encoding proteins")
    ])
    print(f"✅ Target features: {target_features.shape}")
    
    # Split data
    print(f"\nSplitting data...")
    X_drug_temp, X_drug_test, X_target_temp, X_target_test, y_temp, y_test = train_test_split(
        drug_features, target_features, labels,
        test_size=test_size, random_state=42
    )
    
    val_ratio = val_size / (1 - test_size)
    X_drug_train, X_drug_val, X_target_train, X_target_val, y_train, y_val = train_test_split(
        X_drug_temp, X_target_temp, y_temp,
        test_size=val_ratio, random_state=42
    )
    
    print(f"✅ Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")
    
    # Convert to PyTorch tensors
    train_data = {
        'drug': torch.FloatTensor(X_drug_train),
        'target': torch.LongTensor(X_target_train),
        'label': torch.FloatTensor(y_train)
    }
    
    val_data = {
        'drug': torch.FloatTensor(X_drug_val),
        'target': torch.LongTensor(X_target_val),
        'label': torch.FloatTensor(y_val)
    }
    
    test_data = {
        'drug': torch.FloatTensor(X_drug_test),
        'target': torch.LongTensor(X_target_test),
        'label': torch.FloatTensor(y_test)
    }
    
    return train_data, val_data, test_data


# ============================================================================
# PyTorch Dataset
# ============================================================================

class DTIDataset(Dataset):
    """Drug-Target Interaction Dataset"""
    def __init__(self, data_dict):
        self.drug = data_dict['drug']
        self.target = data_dict['target']
        self.label = data_dict['label']
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        return self.drug[idx], self.target[idx], self.label[idx].unsqueeze(0)


# ============================================================================
# Evaluation Metrics
# ============================================================================

def concordance_index(y_true, y_pred):
    """Calculate Concordance Index (C-Index)"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    concordant = 0
    total = 0
    
    for i in range(len(y_true)):
        for j in range(i+1, len(y_true)):
            if y_true[i] != y_true[j]:
                total += 1
                if (y_true[i] < y_true[j] and y_pred[i] < y_pred[j]) or \
                   (y_true[i] > y_true[j] and y_pred[i] > y_pred[j]):
                    concordant += 1
    
    return concordant / total if total > 0 else 0.5


def evaluate_regression(y_true, y_pred):
    """Calculate regression metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    pearson_r, pearson_p = pearsonr(y_true, y_pred)
    ci = concordance_index(y_true, y_pred)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'Pearson_R': pearson_r,
        'Concordance_Index': ci
    }


# ============================================================================
# Training Function
# ============================================================================

def train_model(model, train_loader, val_loader, epochs=50, lr=1e-4, 
                device='cuda', patience=10, save_path='best_model.pt'):
    """Train the model"""
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_metrics': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        
        for drug, target, label in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            drug = drug.to(device)
            target = target.to(device)
            label = label.to(device)
            
            optimizer.zero_grad()
            pred = model(drug, target)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for drug, target, label in val_loader:
                drug = drug.to(device)
                target = target.to(device)
                label = label.to(device)
                
                pred = model(drug, target)
                loss = criterion(pred, label)
                val_loss += loss.item()
                
                all_preds.extend(pred.cpu().numpy().flatten())
                all_labels.extend(label.cpu().numpy().flatten())
        
        val_loss /= len(val_loader)
        metrics = evaluate_regression(all_labels, all_preds)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_metrics'].append(metrics)
        
        # Print progress
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"MSE: {metrics['MSE']:.4f} | RMSE: {metrics['RMSE']:.4f}")
        print(f"R²: {metrics['R2']:.4f} | C-Index: {metrics['Concordance_Index']:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'metrics': metrics
            }, save_path)
            print("✓ Saved best model")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⚠️ Early stopping at epoch {epoch+1}")
            break
    
    return history


# ============================================================================
# Main Training Script
# ============================================================================

def main(args):
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('✅ Libraries imported')
    print(f'Using device: {device}')
    
    # Load dataset
    print(f"\nLoading BindingDB dataset from: {args.data_path}")
    df = pd.read_csv(args.data_path)
    # ⚡ CRITICAL: Verify data is preprocessed
    print("\n" + "="*70)
    print("VERIFYING DATA PREPROCESSING")
    print("="*70)
    y_min, y_max = df['Y'].min(), df['Y'].max()
    print(f"Y range: [{y_min:.2f}, {y_max:.2f}]")
    
    if y_max > 100:
        print("\n❌ ERROR: Using raw data!")
        print("Expected pKd range: [2, 14]")
        print(f"Actual range: [0, {y_max:.0f}]")
        print("\nRun: python 01_data_preprocessor.py --input YOUR_RAW_DATA.csv")
        sys.exit(1)
    
    print("✅ Data in pKd scale!")
    print("="*70 + "\n")
    
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    print(f"✅ Dataset loaded: {len(df)} samples")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Clean data
    df_clean = clean_data(df)
    
    print(f"\n📊 Data Statistics:")
    print(f"Label (Y) range: [{df_clean['Y'].min():.2f}, {df_clean['Y'].max():.2f}]")
    print(f"Label mean: {df_clean['Y'].mean():.2f} ± {df_clean['Y'].std():.2f}")
    
    # Create protein encoder
    protein_encoder = ProteinSequenceEncoder()
    print(f"✅ ProteinSequenceEncoder created")
    print(f"   Vocabulary size: {protein_encoder.vocab_size}")
    
    # Prepare data
    print("\n" + "=" * 70)
    print("PREPARING DATA")
    print("=" * 70)
    
    train_data, val_data, test_data = prepare_data(
        df_clean,
        protein_encoder,
        sample_size=args.sample_size,
        fp_size=1024,
        max_protein_len=1000
    )
    
    print("\n✅ Data preparation complete!")
    
    # Create datasets and dataloaders
    train_dataset = DTIDataset(train_data)
    val_dataset = DTIDataset(val_data)
    test_dataset = DTIDataset(test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"✅ DataLoaders created")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    # Create model
    config = {
        'drug_input_dim': 1024,
        'drug_d_model': 128,
        'drug_num_layers': args.num_layers,
        'drug_num_heads': 8,
        'drug_d_ff': 512,
        'drug_hidden_dim': 256,
        'target_vocab_size': 26,
        'target_embedding_dim': 128,
        'target_num_filters': [32, 64, 96],
        'target_kernel_sizes': [4, 8, 12],
        'target_hidden_dim': 256,
        'decoder_hidden_dims': [1024, 512, 256],
        'dropout': 0.1,
        'binary': False
    }
    
    print("\nCreating model...")
    model = DrugTargetInteractionModel(**config)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Model created!")
    print(f"📊 Total parameters: {n_params:,}")
    
    # Train model
    print("\n🚀 Starting training...")
    
    history = train_model(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.learning_rate,
        device=device,
        patience=args.patience,
        save_path=args.save_path
    )
    
    print("\n✅ Training complete!")
    
    # Plot training history
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('MSE Loss', fontsize=12)
    axes[0].set_title('Training Progress', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    r2_scores = [m['R2'] for m in history['val_metrics']]
    axes[1].plot(r2_scores, color='green', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('R² Score', fontsize=12)
    axes[1].set_title('Validation R²', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    ci_scores = [m['Concordance_Index'] for m in history['val_metrics']]
    axes[2].plot(ci_scores, color='orange', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('C-Index', fontsize=12)
    axes[2].set_title('Concordance Index', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(r'E:\Drug_Protein_Interaction\outputs\plots\training_history.png', dpi=300, bbox_inches='tight')
    print("✅ Training history plot saved!")
    
    # Load best model and evaluate on test set
    print("\nLoading best model...")
    checkpoint = torch.load(args.save_path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print("Evaluating on test set...")
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for drug, target, label in tqdm(test_loader, desc="Testing"):
            drug = drug.to(device)
            target = target.to(device)
            
            pred = model(drug, target)
            test_preds.extend(pred.cpu().numpy().flatten())
            test_labels.extend(label.numpy().flatten())
    
    # Calculate metrics
    test_metrics = evaluate_regression(test_labels, test_preds)
    
    print("\n" + "=" * 70)
    print("TEST SET PERFORMANCE")
    print("=" * 70)
    for metric, value in test_metrics.items():
        print(f"{metric:20s}: {value:.4f}")
    print("=" * 70)
    
    # Scatter plot of predictions
    plt.figure(figsize=(10, 10))
    plt.scatter(test_labels, test_preds, alpha=0.5, s=20)
    plt.plot([min(test_labels), max(test_labels)], 
             [min(test_labels), max(test_labels)], 
             'r--', linewidth=3, label='Perfect Prediction')
    plt.xlabel('True pKd', fontsize=14, fontweight='bold')
    plt.ylabel('Predicted pKd', fontsize=14, fontweight='bold')
    plt.title(f'Test Set Predictions (R²={test_metrics["R2"]:.3f})', 
              fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(r'E:\Drug_Protein_Interaction\outputs\plots\test_predictions.png', dpi=300, bbox_inches='tight')
    print("✅ Prediction plot saved!")
    
    print("\n" + "="*70)
    print("✅ Notebook 3 Complete!")
    print("="*70)
    print("\nNext Step:")
    print("➡️ Run 04_practical_applications.py for real-world applications")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Drug-Target Interaction Model')
    parser.add_argument('--data_path', type=str, default=r'E:\Drug_Protein_Interaction\data_processed\BindingDB_processed.csv',
                        help='Path to BindingDB dataset CSV file')
    parser.add_argument('--sample_size', type=str, default='None',
                        help='Number of samples to use (None or "None" for all data, or specify a number)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--num_layers', type=int, default=6,
                        help='Number of transformer layers')
    parser.add_argument('--save_path', type=str, default=r'E:\Drug_Protein_Interaction\outputs\models\bindingdb_best_model.pt',
                        help='Path to save best model')
    
    args = parser.parse_args()
    
    # Convert sample_size to int or None
    if args.sample_size.lower() == 'none':
        args.sample_size = None
    else:
        args.sample_size = int(args.sample_size)
    
    main(args)
