"""
RUN THIS NOW - Immediate Comparison with Paper Baselines
=========================================================

This will create comparison using:
- Your model (from training results)
- DeepDTA (from paper - typical results)
- DeepPurpose (from paper - typical results)

Later, you can update with your own reproduced results!
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_comparison_tool import ModelComparison

def main():
    print("\n" + "=" * 80)
    print("CREATING MODEL COMPARISON - USING PAPER BASELINES")
    print("=" * 80)
    
    # Create comparison
    comp = ModelComparison(output_dir='E:/DTI_env/comparison_results')
    
    print("\n[1/5] Loading your model results from Excel...")
    comp.add_your_model_from_excel()
    
    print("\n[2/5] Adding DeepDTA baseline (from paper)...")
    comp.add_deepdta_baseline(
        r2=0.870,
        rmse=0.502,
        mae=0.380
    )
    
    print("\n[3/5] Adding DeepPurpose baseline (from paper)...")
    comp.add_deeppurpose_baseline(
        r2=0.883,
        rmse=0.485,
        mae=0.365
    )
    
    print("\n[4/5] Generating visualizations...")
    comp.plot_comparison()
    comp.plot_radar()
    
    print("\n[5/5] Exporting to Excel...")
    excel_file = comp.export_to_excel()
    
    print("\n" + "=" * 80)
    print("PRINTING SUMMARY")
    print("=" * 80)
    comp.print_summary()
    
    print("\n" + "=" * 80)
    print("SUCCESS! COMPARISON COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: E:/DTI_env/comparison_results/")
    print("\nGenerated files:")
    print("  1. model_comparison.png - 6-panel bar chart comparison")
    print("  2. radar_comparison.png - Multi-metric radar plot")
    print(f"  3. {os.path.basename(excel_file)} - Detailed Excel report")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("\n1. Check the comparison plots to see your performance")
    print("2. Review the Excel file for detailed metrics")
    print("\n3. [OPTIONAL] Reproduce baselines with your own runs:")
    print("   → See baseline_reproduction_guide.py")
    print("   → Update quick_comparison.py with your results")
    print("   → Re-run for updated comparison")
    
    print("\n" + "=" * 80)
    
    return comp


if __name__ == "__main__":
    comp = main()
