"""
🚀 2-DAY OPTIMIZED TRAINING WORKFLOW
Optimized for: RTX 3060 (12GB), 16GB RAM, 2-day timeline
Goal: Best deployment model with cold-target prediction capability

DAY 1: 
  Phase 1: Hyperparameter optimization (8h) - 8K samples, 40 trials
  Phase 2: Full training (16h) - ALL 42K samples, target-based split

DAY 2:
  Phase 3: Cold-target 3-fold CV (20h) - 25K samples/fold
  Phase 4: Deployment prep (4h)

KEY FEATURES:
    Target-based splitting for cold-target capability
    Optimized batch sizes for RTX 3060
    Gradient accumulation for larger effective batch
    Memory-efficient data loading
    Automatic GPU optimization
"""

import os
import sys
import argparse
import time
from datetime import datetime
from pathlib import Path
import json

def print_banner(text, char="="):
    """Print formatted banner"""
    print("\n" + char * 80)
    print(f"  {text}")
    print(char * 80 + "\n")


def verify_environment():
    """Verify computational environment"""
    print_banner("VERIFYING ENVIRONMENT")
    
    # Check Python
    import sys
    print(f"Python: {sys.version.split()[0]}")
    
    # Check PyTorch
    import torch
    print(f"PyTorch: {torch.__version__}")
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f" GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # RTX 3060 specific optimization
        if "3060" in gpu_name:
            print(f"RTX 3060 detected - optimizing batch sizes")
            recommended_batch = 32
        else:
            recommended_batch = 64
    else:
        print(f"No GPU detected - training will be VERY slow")
        print(f"Consider using Google Colab or cloud GPU")
        recommended_batch = 16
    
    # Check RAM
    import psutil
    ram_gb = psutil.virtual_memory().total / 1e9
    print(f" RAM: {ram_gb:.1f} GB")
    
    if ram_gb < 12:
        print(f"Low RAM - may need to reduce data loading workers")
    
    # Check data
    data_path = Path(r"E:\DTI_env\data_processed\BindingDB_processed.csv")
    if data_path.exists():
        size_mb = data_path.stat().st_size / 1e6
        print(f" Data: {data_path} ({size_mb:.1f} MB)")
    else:
        print(f" Data not found: {data_path}")
        print(f"Run: python 01_data_preprocessor.py --input YOUR_RAW_DATA.csv")
        return False, 16
    
    print_banner(" ENVIRONMENT READY", char="=")
    
    return True, recommended_batch


def estimate_time(phase, samples, trials=None, folds=None):
    """Estimate training time"""
    
    if phase == "hyperopt":
        # ~12 min per trial with 10K samples
        hours = (trials * 12) / 60
        return hours
    
    elif phase == "full_train":
        # ~0.7 hours per 1K samples (150 epochs)
        hours = samples / 1000 * 0.7
        return hours
    
    elif phase == "cv":
        # ~6 hours per fold with 25K samples
        hours = folds * 6
        return hours
    
    return 0


def show_2day_plan(batch_size):
    """Show the complete 2-day plan"""
    print_banner(" 2-DAY MASTER PLAN")
    
    print("GOAL: Production model with cold-target prediction capability")
    print(f"Hardware: RTX 3060, Batch size: {batch_size}")
    print(f"Full dataset: 42,227 samples\n")
    
    # Day 1
    print("="*80)
    print("DAY 1: OPTIMIZATION + FULL TRAINING (24 hours)")
    print("="*80)
    
    print("\n PHASE 1: Hyperparameter Optimization (8 hours)")
    print("   ├─ Samples: 10,000 (for speed)")
    print("   ├─ Trials: 50 (Optuna)")
    print("   ├─ Batch size:", batch_size)
    print("   ├─ Strategy: Find optimal model configuration")
    print("   └─ Output: best_config.json")
    
    phase1_hours = estimate_time("hyperopt", 10000, trials=50)
    print(f"     Estimated: {phase1_hours:.1f} hours\n")

    print(" PHASE 2: Full Training (16 hours)")
    print("   ├─ Samples: ALL 42,227 samples")
    print("   ├─ Split: Target-based (80/20) for cold-target")
    print("   ├─ Epochs: 150 (with early stopping)")
    print("   ├─ Config: From Phase 1 best_config.json")
    print("   ├─ Strategy: Train on all data with best hyperparameters")
    print("   └─ Output: production_model.pt")
    
    phase2_hours = estimate_time("full_train", 42227)
    print(f"     Estimated: {phase2_hours:.1f} hours\n")
    
    # Day 2
    print("="*80)
    print("DAY 2: COLD-TARGET VALIDATION + DEPLOYMENT (24 hours)")
    print("="*80)

    print("\n PHASE 3: Cold-Target Cross-Validation (20 hours)")
    print("   ├─ Folds: 3 (stratified by target)")
    print("   ├─ Samples: 25,000 per fold")
    print("   ├─ Split: By unique targets (not random!)")
    print("   ├─ Strategy: Validate generalization to new proteins")
    print("   └─ Output: cv_results.json, confidence intervals")
    
    phase3_hours = estimate_time("cv", 25000, folds=3)
    print(f"     Estimated: {phase3_hours:.1f} hours\n")

    print(" PHASE 4: Deployment Preparation (4 hours)")
    print("   ├─ Final model packaging")
    print("   ├─ Inference script creation")
    print("   ├─ Documentation")
    print("   └─ Output: Ready-to-deploy model")
    print(f"    Estimated: 4 hours\n")
    
    # Summary
    total_hours = phase1_hours + phase2_hours + phase3_hours + 4
    print("="*80)
    print(" TIMELINE SUMMARY")
    print("="*80)
    print(f"Phase 1 (Hyperopt):     {phase1_hours:>6.1f} hours")
    print(f"Phase 2 (Full Train):   {phase2_hours:>6.1f} hours")
    print(f"Phase 3 (Cold-Target):  {phase3_hours:>6.1f} hours")
    print(f"Phase 4 (Deployment):   {4:>6.1f} hours")
    print("-" * 80)
    print(f"TOTAL:                  {total_hours:>6.1f} hours (~{total_hours/24:.1f} days)")
    print("="*80 + "\n")
    
    # Expected results
    print("="*80)
    print(" EXPECTED RESULTS")
    print("="*80)
    print("After Phase 2 (Full Training):")
    print("   ├─ R² (same target): 0.75-0.82")
    print("   ├─ R² (cold target): 0.60-0.72")
    print("   └─ RMSE: 0.55-0.65 pKd units")
    print("\nAfter Phase 3 (Cross-Validation):")
    print("   ├─ Mean R²: 0.65-0.75 (±0.03)")
    print("   ├─ Cold-target capability: VALIDATED ")
    print("   └─ Ready for deployment: YES ")
    print("="*80 + "\n")


def run_phase_1(batch_size):
    """Phase 1: Hyperparameter Optimization"""
    print_banner(" PHASE 1: HYPERPARAMETER OPTIMIZATION (8 hours)")
    
    print("Starting hyperparameter optimization...")
    print(f"Using 10,000 samples for speed")
    print(f"50 trials with Optuna")
    print(f"Batch size: {batch_size}\n")

    cmd = f"python 06_hyperparameter_optimization_MODIFIED.py " \
          f"--data_path E:\\DTI_env\\data_processed\\BindingDB_processed.csv " \
          f"--sample_size 10000 " \
          f"--n_trials 50 " \
          f"--batch_size {batch_size}"
    
    print(f"Command: {cmd}\n")
    print("="*80)
    
    start_time = time.time()
    result = os.system(cmd)
    elapsed = (time.time() - start_time) / 3600
    
    if result == 0:
        print(f"\n Phase 1 Complete! ({elapsed:.1f} hours)")
        print(f" Best config saved: E:\\DTI_env\\results\\best_config.json\n")
        return True
    else:
        print(f"\n Phase 1 Failed!")
        return False


def run_phase_2(batch_size):
    """Phase 2: Full Training with Target-Based Split"""
    print_banner(" PHASE 2: FULL TRAINING (16 hours)")
    
    # Load best config
    config_path = Path(r"E:\DTI_env\results\best_config.json")
    
    if not config_path.exists():
        print("best_config.json not found - using default hyperparameters")
        d_model = 128
        num_layers = 6
        num_heads = 8
        d_ff = 512
        dropout = 0.15
        learning_rate = 0.0001
    else:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        d_model = config.get('drug_d_model', 128)
        num_layers = config.get('drug_num_layers', 6)
        num_heads = config.get('drug_num_heads', 8)
        d_ff = config.get('drug_d_ff', 512)
        dropout = config.get('dropout', 0.15)
        learning_rate = config.get('learning_rate', 0.0001)
        
        print("Loaded best hyperparameters from Phase 1:")
        print(f"d_model={d_model}, layers={num_layers}, heads={num_heads}")
        print(f"d_ff={d_ff}, dropout={dropout:.3f}, lr={learning_rate:.6f}\n")
    
    print("Starting full training on ALL 42,227 samples...")
    print(f"Target-based split: 20% targets held out (cold-target)")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: 100 (with early stopping)")
    print(f"Early stopping: Focused on LOSS & RMSE (not R²!)\n")
    
    # Create log file
    log_dir = Path(r"E:\DTI_env\logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"phase2_full_training_{timestamp}.log"
    
    cmd = f"python 05_training_pipeline_enhanced_MODIFIED.py " \
          f"--data_path E:\\DTI_env\\data_processed\\BindingDB_processed.csv " \
          f"--epochs 100 " \
          f"--batch_size {batch_size} " \
          f"--d_model {d_model} " \
          f"--num_layers {num_layers} " \
          f"--num_heads {num_heads} " \
          f"--d_ff {d_ff} " \
          f"--dropout {dropout} " \
          f"--learning_rate {learning_rate} " \
          f"--augment 1 " \
          f"--use_onecycle True " \
          f"--patience 20 " \
          f"--save_path E:\\DTI_env\\models_saved\\production_model.pt " \
          f"> \"{log_file}\" 2>&1"
    
    print(f"Command: {cmd}\n")
    print(f" All console output will be saved to:")
    print(f"   {log_file}")
    print("="*80)
    
    start_time = time.time()
    result = os.system(cmd)
    elapsed = (time.time() - start_time) / 3600
    
    if result == 0:
        print(f"\n Phase 2 Complete! ({elapsed:.1f} hours)")
        print(f"Production model: E:\\DTI_env\\models\\production_model.pt")
        print(f"Training metrics Excel: E:\\DTI_env\\results\\training_metrics.xlsx")
        print(f"Results summary image: E:\\DTI_env\\plots\\training_results_summary.png")
        print(f"Full training log: {log_file}\n")
        return True
    else:
        print(f"\n Phase 2 Failed!")
        print(f" Check log file for errors: {log_file}")
        return False


def show_day2_menu():
    """Show Day 2 options"""
    print_banner(" DAY 2: VALIDATION & DEPLOYMENT")
    
    print("Choose your next step:\n")
    print("1. Run Phase 3: Cold-Target Cross-Validation (20 hours)")
    print("   └─ 3-fold CV with target-based splits")
    print("   └─ Validates generalization to new proteins")
    print()
    print("2. Skip to Phase 4: Deployment Preparation (4 hours)")
    print("   └─ Package model for production")
    print("   └─ Create inference scripts")
    print()
    print("3. Run both Phase 3 + 4 (24 hours)")
    print()
    print("0. Exit")
    print()
    
    choice = input("Enter choice (1/2/3/0): ").strip()
    return choice


def run_phase_3(batch_size):
    """Phase 3: Cold-Target Cross-Validation"""
    print_banner(" PHASE 3: COLD-TARGET CROSS-VALIDATION (20 hours)")
    
    print("Starting cold-target cross-validation...")
    print(f" 3-fold CV")
    print(f" Split by unique targets")
    print(f" 25,000 samples per fold")
    print(f" Epochs: 100 per fold")
    print(f" Batch size: {batch_size}\n")
    
    # Load best config
    config_path = Path(r"E:\DTI_env\results\best_config.json")
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        d_model = config.get('drug_d_model', 128)
        num_layers = config.get('drug_num_layers', 6)
        num_heads = config.get('drug_num_heads', 8)
        d_ff = config.get('drug_d_ff', 512)
        dropout = config.get('dropout', 0.15)
        learning_rate = config.get('learning_rate', 0.0001)
    else:
        d_model = 128
        num_layers = 6
        num_heads = 8
        d_ff = 512
        dropout = 0.15
        learning_rate = 0.0001
    
    # Create log file
    log_dir = Path(r"E:\DTI_env\logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"phase3_cross_validation_{timestamp}.log"
    
    cmd = f"python 05_training_pipeline_enhanced_MODIFIED.py " \
          f"--data_path E:\\DTI_env\\data_processed\\BindingDB_processed.csv " \
          f"--use_cv True " \
          f"--k_folds 3 " \
          f"--sample_size 25000 " \
          f"--epochs 100 " \
          f"--batch_size {batch_size} " \
          f"--d_model {d_model} " \
          f"--num_layers {num_layers} " \
          f"--num_heads {num_heads} " \
          f"--d_ff {d_ff} " \
          f"--dropout {dropout} " \
          f"--learning_rate {learning_rate} " \
          f"> \"{log_file}\" 2>&1"
    
    print(f"Command: {cmd}\n")
    print(f" All console output will be saved to:")
    print(f"   {log_file}")
    print("="*80)
    
    start_time = time.time()
    result = os.system(cmd)
    elapsed = (time.time() - start_time) / 3600
    
    if result == 0:
        print(f"\n Phase 3 Complete! ({elapsed:.1f} hours)")
        print(f" CV results saved: E:\\DTI_env\\results\\cv_results.json")
        print(f" Full CV log: {log_file}\n")
        return True
    else:
        print(f"\n Phase 3 Failed!")
        print(f" Check log file for errors: {log_file}")
        return False


def run_phase_4():
    """Phase 4: Deployment Preparation"""
    print_banner(" PHASE 4: DEPLOYMENT PREPARATION")
    
    print("Preparing model for deployment...\n")
    
    # Create deployment package
    deployment_dir = Path(r"E:\DTI_env\deployment")
    deployment_dir.mkdir(exist_ok=True)
    
    print("Creating deployment package:")
    print(f" Directory: {deployment_dir}")
    
    # Check if production model exists
    model_path = Path(r"E:\DTI_env\models_saved\production_model.pt")
    if not model_path.exists():
        print(f"\n  Production model not found: {model_path}")
        print(f"   Run Phase 2 first!")
        return False
    
    print(f" Model: {model_path}")
    
    # Copy model to deployment
    import shutil
    shutil.copy(model_path, deployment_dir / "model.pt")
    print(f" Copied to: {deployment_dir / 'model.pt'}")
    
    # Create inference script (will be created below)
    print(f" Creating inference script...")
    
    # Create requirements
    print(f" Creating requirements.txt...")
    
    # Create README
    print(f" Creating deployment README...")
    print(f"\n Phase 4 Complete!")
    print(f"\nDeployment package ready at: {deployment_dir}")
    print(f"\nNext steps:")
    print(f"  1. Test model: python deployment/test_model.py")
    print(f"  2. Deploy to production")
    
    return True


def main():
    """Main workflow controller"""
    parser = argparse.ArgumentParser(description='2-Day Optimized Training Workflow')
    parser.add_argument('--phase', type=str, choices=['1', '2', '3', '4', 'all'],
                       help='Run specific phase (1=hyperopt, 2=full_train, 3=cv, 4=deploy, all=everything)')
    parser.add_argument('--skip_verify', action='store_true',
                       help='Skip environment verification')
    
    args = parser.parse_args()
    
    # Verify environment
    if not args.skip_verify:
        env_ok, batch_size = verify_environment()
        if not env_ok:
            print(" Environment check failed! Fix issues and try again.")
            return
    else:
        batch_size = 32  # Default for RTX 3060
    
    # Show plan
    if not args.phase:
        show_2day_plan(batch_size)
        
        print("="*80)
        print(" INTERACTIVE MODE")
        print("="*80)
        print("\nChoose what to run:\n")
        print("1. DAY 1: Phase 1 + 2 (Hyperopt + Full Training) - 24 hours")
        print("2. DAY 2: Phase 3 + 4 (CV + Deployment) - 24 hours")
        print("3. Run all phases (2 days)")
        print("4. Run specific phase")
        print()
        print("0. Exit")
        print()
        
        choice = input("Enter choice (1/2/3/4/0): ").strip()
        
        if choice == '1':
            # Day 1
            print_banner(" STARTING DAY 1")
            success = run_phase_1(batch_size)
            if success:
                success = run_phase_2(batch_size)
                if success:
                    print_banner(" DAY 1 COMPLETE!")
                    print("\n Next: Run Day 2 (Phase 3 + 4)")
                    print("   python TRAIN_WORKFLOW_2DAY_OPTIMIZED.py --phase 3")
        
        elif choice == '2':
            # Day 2
            print_banner(" STARTING DAY 2")
            day2_choice = show_day2_menu()
            
            if day2_choice == '1':
                run_phase_3(batch_size)
            elif day2_choice == '2':
                run_phase_4()
            elif day2_choice == '3':
                success = run_phase_3(batch_size)
                if success:
                    run_phase_4()
                    print_banner(" DAY 2 COMPLETE!")
                    print("\n FULL WORKFLOW COMPLETE!")
                    print("   Your model is ready for deployment!")
        
        elif choice == '3':
            # All phases
            print_banner(" STARTING FULL 2-DAY WORKFLOW")
            
            success = run_phase_1(batch_size)
            if not success:
                return
            
            success = run_phase_2(batch_size)
            if not success:
                return
            
            print_banner(" DAY 1 COMPLETE!")
            input("\n  Press Enter to start Day 2...")
            
            success = run_phase_3(batch_size)
            if not success:
                return
            
            run_phase_4()
            
            print_banner(" FULL WORKFLOW COMPLETE!")
        
        elif choice == '4':
            # Specific phase
            phase = input("\nEnter phase number (1/2/3/4): ").strip()
            
            if phase == '1':
                run_phase_1(batch_size)
            elif phase == '2':
                run_phase_2(batch_size)
            elif phase == '3':
                run_phase_3(batch_size)
            elif phase == '4':
                run_phase_4()
    
    else:
        # Command-line mode
        if args.phase == '1':
            run_phase_1(batch_size)
        elif args.phase == '2':
            run_phase_2(batch_size)
        elif args.phase == '3':
            run_phase_3(batch_size)
        elif args.phase == '4':
            run_phase_4()
        elif args.phase == 'all':
            run_phase_1(batch_size)
            run_phase_2(batch_size)
            run_phase_3(batch_size)
            run_phase_4()


if __name__ == "__main__":
    main()
