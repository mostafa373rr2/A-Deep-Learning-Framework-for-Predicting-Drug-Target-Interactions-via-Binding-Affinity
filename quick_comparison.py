"""
QUICK MODEL COMPARISON - Add Your Own Results
Run this after you reproduce DeepDTA and DeepPurpose
"""

from model_comparison_tool import ModelComparison

# ============================================================================
# STEP 1: Create comparison object
# ============================================================================

comp = ModelComparison(output_dir='E:/DTI_env/comparison_results')

# ============================================================================
# STEP 2: Add your model (automatic from Excel)
# ============================================================================

comp.add_your_model_from_excel()

# ============================================================================
# STEP 3: Add DeepDTA results
# ============================================================================

# Option A: Use paper baseline (default)
comp.add_deepdta_baseline()

# Option B: Add YOUR OWN DeepDTA reproduction results
# Uncomment and update with your actual results:
"""
comp.add_model(
    name='DeepDTA (Reproduced)',
    metrics_dict={
        'R2': 0.875,        # YOUR RESULT
        'RMSE': 0.495,      # YOUR RESULT
        'MAE': 0.375,       # YOUR RESULT
        'Pearson_R': 0.93,  # YOUR RESULT
        'Spearman_R': 0.82, # YOUR RESULT
        'C_Index': 0.83     # YOUR RESULT
    },
    training_time=6.5,      # YOUR TRAINING TIME
    params=2100000,
    notes='Reproduced on same hardware (RTX 3060, 16GB RAM)'
)
"""

# ============================================================================
# STEP 4: Add DeepPurpose results
# ============================================================================

# Option A: Use paper baseline (default)
comp.add_deeppurpose_baseline()

# Option B: Add YOUR OWN DeepPurpose reproduction results
# Uncomment and update with your actual results:
"""
comp.add_model(
    name='DeepPurpose (Reproduced)',
    metrics_dict={
        'R2': 0.885,        # YOUR RESULT
        'RMSE': 0.480,      # YOUR RESULT
        'MAE': 0.360,       # YOUR RESULT
        'Pearson_R': 0.94,  # YOUR RESULT
        'Spearman_R': 0.83, # YOUR RESULT
        'C_Index': 0.84     # YOUR RESULT
    },
    training_time=8.5,      # YOUR TRAINING TIME
    params=3500000,
    notes='Reproduced using DeepPurpose framework'
)
"""

# ============================================================================
# STEP 5: Generate comparison
# ============================================================================

print("\nGenerating comparison visualizations...")
comp.plot_comparison()
comp.plot_radar()
comp.export_to_excel()
comp.print_summary()

print("\n" + "=" * 80)
print("DONE! Check E:/DTI_env/comparison_results/")
print("=" * 80)
