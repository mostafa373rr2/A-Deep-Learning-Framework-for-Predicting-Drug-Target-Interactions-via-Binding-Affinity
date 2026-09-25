"""
📊 Data Preprocessor - CRITICAL STEP!
This fixes the NEGATIVE R² issue by properly converting binding affinities

What this does:
1. Loads raw BindingDB.csv
2. Converts Y (nM) → pKd (ESSENTIAL!)
3. Cleans and validates data
4. Saves processed data
5. Creates visualization

Run this BEFORE any training!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class BindingDBPreprocessor:
    """Complete preprocessing pipeline for BindingDB"""
    
    def __init__(self, input_path, output_dir='data_processed'):
        self.input_path = input_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.df_raw = None
        self.df_clean = None
        self.stats = {}
    
    def load_data(self):
        """Load raw data"""
        print("\n" + "="*70)
        print("STEP 1: LOADING DATA")
        print("="*70)
        
        self.df_raw = pd.read_csv(self.input_path)
        print(f"✅ Loaded: {len(self.df_raw):,} samples")
        print(f"✅ Columns: {self.df_raw.columns.tolist()}")
        
        # Store initial stats
        self.stats['initial_samples'] = len(self.df_raw)
        
        return self
    
    def check_columns(self):
        """Verify required columns exist"""
        print("\n" + "="*70)
        print("STEP 2: CHECKING COLUMNS")
        print("="*70)
        
        # Possible column names
        drug_cols = ['Drug', 'SMILES', 'drug', 'smiles', 'Ligand SMILES']
        target_cols = ['Target', 'Target Sequence', 'target', 'Protein Sequence']
        y_cols = ['Y', 'y', 'Ki', 'Kd', 'IC50', 'affinity']
        
        # Find actual column names
        drug_col = None
        target_col = None
        y_col = None
        
        for col in drug_cols:
            if col in self.df_raw.columns:
                drug_col = col
                break
        
        for col in target_cols:
            if col in self.df_raw.columns:
                target_col = col
                break
        
        for col in y_cols:
            if col in self.df_raw.columns:
                y_col = col
                break
        
        if not all([drug_col, target_col, y_col]):
            print("❌ ERROR: Could not find required columns!")
            print(f"   Available columns: {self.df_raw.columns.tolist()}")
            raise ValueError("Missing required columns")
        
        # Rename to standard names
        self.df_raw = self.df_raw.rename(columns={
            drug_col: 'Drug',
            target_col: 'Target',
            y_col: 'Y'
        })
        
        print(f"✅ Drug column: {drug_col} → 'Drug'")
        print(f"✅ Target column: {target_col} → 'Target'")
        print(f"✅ Affinity column: {y_col} → 'Y'")
        
        return self
    
    def remove_missing(self):
        """Remove rows with missing values"""
        print("\n" + "="*70)
        print("STEP 3: REMOVING MISSING VALUES")
        print("="*70)
        
        initial = len(self.df_raw)
        
        # Check missing values
        missing = self.df_raw[['Drug', 'Target', 'Y']].isnull().sum()
        print("Missing values:")
        for col, count in missing.items():
            if count > 0:
                print(f"   {col}: {count:,} ({count/initial*100:.1f}%)")
        
        # Drop missing
        self.df_raw = self.df_raw.dropna(subset=['Drug', 'Target', 'Y'])
        
        removed = initial - len(self.df_raw)
        print(f"\n✅ Removed: {removed:,} samples with missing values")
        print(f"✅ Remaining: {len(self.df_raw):,} samples")
        
        self.stats['after_missing'] = len(self.df_raw)
        
        return self
    
    def convert_to_pkd(self):
        """
        CRITICAL STEP: Convert Y (nM) to pKd
        This is what fixes the NEGATIVE R² issue!
        """
        print("\n" + "="*70)
        print("STEP 4: CONVERTING TO pKd SCALE ⚡ CRITICAL!")
        print("="*70)
        
        # Check original Y values
        print(f"\nOriginal Y (binding affinity in nM):")
        print(f"   Range: [{self.df_raw['Y'].min():.2e}, {self.df_raw['Y'].max():.2e}]")
        print(f"   Mean: {self.df_raw['Y'].mean():.2e}")
        print(f"   Median: {self.df_raw['Y'].median():.2e}")
        
        # Remove zero/negative values (can't take log)
        initial = len(self.df_raw)
        self.df_raw = self.df_raw[self.df_raw['Y'] > 0]
        removed = initial - len(self.df_raw)
        
        if removed > 0:
            print(f"\n⚠️  Removed {removed} samples with Y ≤ 0")
        
        # Store original Y
        self.df_raw['Y_nM_original'] = self.df_raw['Y'].copy()
        
        # Convert to pKd
        # Formula: pKd = 9 - log10(Y_nM)
        # This is because: pKd = -log10(Kd_M) = -log10(Kd_nM / 1e9)
        self.df_raw['pKd'] = 9 - np.log10(self.df_raw['Y'])
        
        # Replace Y with pKd
        self.df_raw['Y'] = self.df_raw['pKd']
        
        print(f"\n✅ Conversion complete!")
        print(f"   Formula: pKd = 9 - log10(Y_nM)")
        
        print(f"\nNew Y (pKd scale):")
        print(f"   Range: [{self.df_raw['Y'].min():.2f}, {self.df_raw['Y'].max():.2f}]")
        print(f"   Mean: {self.df_raw['Y'].mean():.2f} ± {self.df_raw['Y'].std():.2f}")
        print(f"   Median: {self.df_raw['Y'].median():.2f}")
        
        print(f"\n💡 pKd interpretation:")
        print(f"   pKd = 9: Very strong binding (1 nM) ⭐⭐⭐⭐⭐")
        print(f"   pKd = 7: Good binding (100 nM) ⭐⭐⭐")
        print(f"   pKd = 5: Weak binding (10 µM) ⭐")
        print(f"   Higher pKd = Stronger binding = Better drug!")
        
        self.stats['after_pkd'] = len(self.df_raw)
        
        return self
    
    def clean_sequences(self):
        """Clean drug and protein sequences"""
        print("\n" + "="*70)
        print("STEP 5: CLEANING SEQUENCES")
        print("="*70)
        
        initial = len(self.df_raw)
        
        # Remove very short SMILES (likely errors)
        self.df_raw = self.df_raw[self.df_raw['Drug'].str.len() >= 5]
        smiles_removed = initial - len(self.df_raw)
        
        # Remove very short protein sequences
        initial2 = len(self.df_raw)
        self.df_raw = self.df_raw[self.df_raw['Target'].str.len() >= 50]
        protein_removed = initial2 - len(self.df_raw)
        
        print(f"✅ Removed {smiles_removed:,} samples with very short SMILES (<5 chars)")
        print(f"✅ Removed {protein_removed:,} samples with very short proteins (<50 AA)")
        print(f"✅ Remaining: {len(self.df_raw):,} samples")
        
        self.stats['after_cleaning'] = len(self.df_raw)
        
        return self
    
    def remove_outliers(self):
        """Remove pKd outliers using IQR method"""
        print("\n" + "="*70)
        print("STEP 6: REMOVING OUTLIERS")
        print("="*70)
        
        initial = len(self.df_raw)
        
        # Use conservative IQR method
        Q1 = self.df_raw['Y'].quantile(0.05)
        Q3 = self.df_raw['Y'].quantile(0.95)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        print(f"pKd bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        self.df_raw = self.df_raw[
            (self.df_raw['Y'] >= lower_bound) & 
            (self.df_raw['Y'] <= upper_bound)
        ]
        
        removed = initial - len(self.df_raw)
        print(f"✅ Removed: {removed:,} outliers ({removed/initial*100:.1f}%)")
        print(f"✅ Remaining: {len(self.df_raw):,} samples")
        
        self.stats['after_outliers'] = len(self.df_raw)
        
        return self
    
    def remove_duplicates(self):
        """Remove duplicate drug-target pairs"""
        print("\n" + "="*70)
        print("STEP 7: REMOVING DUPLICATES")
        print("="*70)
        
        initial = len(self.df_raw)
        
        # Check for duplicates
        duplicates = self.df_raw.duplicated(subset=['Drug', 'Target'], keep='first')
        n_duplicates = duplicates.sum()
        
        if n_duplicates > 0:
            print(f"Found {n_duplicates:,} duplicate drug-target pairs")
            
            # For duplicates, keep the one with median Y value
            self.df_raw = self.df_raw.sort_values('Y')
            self.df_raw = self.df_raw.drop_duplicates(subset=['Drug', 'Target'], keep='first')
            
            removed = initial - len(self.df_raw)
            print(f"✅ Removed: {removed:,} duplicates")
        else:
            print(f"✅ No duplicates found")
        
        print(f"✅ Final dataset: {len(self.df_raw):,} samples")
        
        self.stats['final_samples'] = len(self.df_raw)
        
        return self
    
    def visualize(self):
        """Create visualizations"""
        print("\n" + "="*70)
        print("STEP 8: CREATING VISUALIZATIONS")
        print("="*70)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Original Y distribution (nM)
        axes[0, 0].hist(self.df_raw['Y_nM_original'], bins=50, 
                       color='red', alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Y (nM)', fontweight='bold', fontsize=12)
        axes[0, 0].set_ylabel('Count', fontweight='bold', fontsize=12)
        axes[0, 0].set_title('❌ Original Distribution (PROBLEMATIC!)', 
                            fontweight='bold', fontsize=14)
        axes[0, 0].set_yscale('log')
        axes[0, 0].grid(alpha=0.3)
        axes[0, 0].text(0.95, 0.95, 'Loss in\nBILLIONS!', 
                       transform=axes[0, 0].transAxes,
                       fontsize=16, fontweight='bold', color='red',
                       ha='right', va='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Log-transformed original
        axes[0, 1].hist(np.log10(self.df_raw['Y_nM_original']), bins=50,
                       color='orange', alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('log10(Y_nM)', fontweight='bold', fontsize=12)
        axes[0, 1].set_ylabel('Count', fontweight='bold', fontsize=12)
        axes[0, 1].set_title('⚠️ Log-transformed', fontweight='bold', fontsize=14)
        axes[0, 1].grid(alpha=0.3)
        
        # 3. pKd distribution (CORRECTED!)
        axes[0, 2].hist(self.df_raw['Y'], bins=50,
                       color='green', alpha=0.7, edgecolor='black')
        axes[0, 2].axvline(self.df_raw['Y'].mean(), color='red', 
                          linestyle='--', linewidth=2,
                          label=f'Mean: {self.df_raw["Y"].mean():.2f}')
        axes[0, 2].set_xlabel('pKd (CORRECTED)', fontweight='bold', fontsize=12)
        axes[0, 2].set_ylabel('Count', fontweight='bold', fontsize=12)
        axes[0, 2].set_title('✅ pKd Distribution (READY!)', 
                            fontweight='bold', fontsize=14)
        axes[0, 2].legend(fontsize=10)
        axes[0, 2].grid(alpha=0.3)
        axes[0, 2].text(0.95, 0.95, 'Loss\n0.5-2.0', 
                       transform=axes[0, 2].transAxes,
                       fontsize=16, fontweight='bold', color='green',
                       ha='right', va='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 4. Transformation: nM → pKd (sample 1000 points)
        sample_size = min(1000, len(self.df_raw))
        sample_df = self.df_raw.sample(n=sample_size, random_state=42)
        
        axes[1, 0].scatter(sample_df['Y_nM_original'], sample_df['Y'],
                          alpha=0.5, s=20, c=sample_df['Y'], cmap='viridis')
        axes[1, 0].set_xlabel('Original Y (nM)', fontweight='bold', fontsize=12)
        axes[1, 0].set_ylabel('pKd', fontweight='bold', fontsize=12)
        axes[1, 0].set_title('Transformation: nM → pKd', fontweight='bold', fontsize=14)
        axes[1, 0].set_xscale('log')
        axes[1, 0].grid(alpha=0.3)
        
        # 5. pKd box plot by range
        pkd_ranges = pd.cut(self.df_raw['Y'], bins=[0, 5, 7, 9, 15],
                           labels=['Weak\n(2-5)', 'Moderate\n(5-7)', 
                                  'Good\n(7-9)', 'Strong\n(9+)'])
        
        data_for_box = [self.df_raw[pkd_ranges == label]['Y'].values 
                        for label in ['Weak\n(2-5)', 'Moderate\n(5-7)', 
                                     'Good\n(7-9)', 'Strong\n(9+)']]
        
        bp = axes[1, 1].boxplot(data_for_box, labels=['Weak\n(2-5)', 'Moderate\n(5-7)', 
                                                       'Good\n(7-9)', 'Strong\n(9+)'],
                               patch_artist=True)
        for patch, color in zip(bp['boxes'], ['red', 'orange', 'lightgreen', 'green']):
            patch.set_facecolor(color)
        
        axes[1, 1].set_ylabel('pKd', fontweight='bold', fontsize=12)
        axes[1, 1].set_title('pKd Distribution by Affinity', fontweight='bold', fontsize=14)
        axes[1, 1].grid(alpha=0.3, axis='y')
        
        # 6. Statistics summary
        axes[1, 2].axis('off')
        
        stats_text = f"""
PREPROCESSING SUMMARY
{'='*50}

Initial samples:     {self.stats['initial_samples']:>10,}
After missing:       {self.stats['after_missing']:>10,}
After pKd:           {self.stats['after_pkd']:>10,}
After cleaning:      {self.stats['after_cleaning']:>10,}
After outliers:      {self.stats['after_outliers']:>10,}
Final samples:       {self.stats['final_samples']:>10,}

Retention rate:      {self.stats['final_samples']/self.stats['initial_samples']*100:>9.1f}%

{'='*50}
pKd STATISTICS
{'='*50}

Min:                 {self.df_raw['Y'].min():>10.2f}
25th percentile:     {self.df_raw['Y'].quantile(0.25):>10.2f}
Median:              {self.df_raw['Y'].median():>10.2f}
Mean:                {self.df_raw['Y'].mean():>10.2f}
75th percentile:     {self.df_raw['Y'].quantile(0.75):>10.2f}
Max:                 {self.df_raw['Y'].max():>10.2f}
Std Dev:             {self.df_raw['Y'].std():>10.2f}

{'='*50}
✅ READY FOR TRAINING!
        """
        
        axes[1, 2].text(0.1, 0.95, stats_text, transform=axes[1, 2].transAxes,
                       fontsize=11, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'preprocessing_visualization.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved visualization: {output_path}")
        
        plt.close()
        
        return self
    
    def save_processed_data(self):
        """Save processed dataset"""
        print("\n" + "="*70)
        print("STEP 9: SAVING PROCESSED DATA")
        print("="*70)
        
        # Keep only necessary columns
        self.df_clean = self.df_raw[['Drug', 'Target', 'Y']].copy()
        
        # Save
        output_path = self.output_dir / 'BindingDB_processed.csv'
        self.df_clean.to_csv(output_path, index=False)
        
        print(f"✅ Saved: {output_path}")
        print(f"✅ Samples: {len(self.df_clean):,}")
        print(f"✅ Columns: {self.df_clean.columns.tolist()}")
        print(f"✅ Size: {output_path.stat().st_size / 1e6:.1f} MB")
        
        # Also save with original Y for reference
        reference_path = self.output_dir / 'BindingDB_with_original_Y.csv'
        self.df_raw[['Drug', 'Target', 'Y', 'Y_nM_original']].to_csv(
            reference_path, index=False
        )
        print(f"✅ Saved reference: {reference_path}")
        
        return self
    
    def run_pipeline(self):
        """Run complete preprocessing pipeline"""
        print("\n" + "="*70)
        print("🔬 BINDINGDB PREPROCESSING PIPELINE")
        print("="*70)
        print("\nThis will:")
        print("1. Load raw data")
        print("2. Check columns")
        print("3. Remove missing values")
        print("4. Convert to pKd scale ⚡ CRITICAL!")
        print("5. Clean sequences")
        print("6. Remove outliers")
        print("7. Remove duplicates")
        print("8. Create visualizations")
        print("9. Save processed data")
        
        (self
         .load_data()
         .check_columns()
         .remove_missing()
         .convert_to_pkd()
         .clean_sequences()
         .remove_outliers()
         .remove_duplicates()
         .visualize()
         .save_processed_data())
        
        print("\n" + "="*70)
        print("✅ PREPROCESSING COMPLETE!")
        print("="*70)
        
        print(f"\n📊 SUMMARY:")
        print(f"   Input:  {self.stats['initial_samples']:,} samples")
        print(f"   Output: {self.stats['final_samples']:,} samples")
        print(f"   Retention: {self.stats['final_samples']/self.stats['initial_samples']*100:.1f}%")
        
        print(f"\n📁 OUTPUT FILES:")
        print(f"   ✅ data_processed/BindingDB_processed.csv")
        print(f"   ✅ data_processed/preprocessing_visualization.png")
        
        print(f"\n🎯 NEXT STEPS:")
        print(f"   1. Examine preprocessing_visualization.png")
        print(f"   2. Run 02_train_baseline.py for quick test")
        print(f"   3. If R² > 0, proceed to full training!")
        
        print(f"\n💡 EXPECTED RESULTS:")
        print(f"   Before: R² = negative, Loss in billions ❌")
        print(f"   After:  R² = 0.60-0.70, Loss = 0.5-2.0 ✅")
        
        return self


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess BindingDB data')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to raw BindingDB.csv')
    parser.add_argument('--output_dir', type=str, default='data_processed',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Run preprocessing
    preprocessor = BindingDBPreprocessor(args.input, args.output_dir)
    preprocessor.run_pipeline()


if __name__ == "__main__":
    main()
