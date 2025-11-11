"""
Quick test of MambaTSR training script
Test with small model and 1 epoch to verify everything works
"""

import os
import sys
import torch

# Add MambaTSR to path
sys.path.insert(0, '/mnt/g/Dataset/MambaTSR')

# Import training script
sys.path.insert(0, '/mnt/g/Dataset')
from train_mambatsr_plantvillage import (
    MambaTSRConfig, 
    prepare_dataset, 
    build_model, 
    train_one_epoch,
    validate,
    get_optimizer_and_scheduler,
    set_seed
)
import torch.nn as nn


class TestConfig(MambaTSRConfig):
    """Test configuration - small and fast"""
    # Small model for testing
    depths = [2, 2]  # Only 2 layers
    dims = [96, 192]
    
    # Fast training
    batch_size = 8
    num_epochs = 1
    learning_rate = 1e-4
    
    # Small dataset sample
    train_ratio = 0.8
    val_ratio = 0.2


def test_training():
    """Test the training pipeline"""
    print("="*80)
    print("Testing MambaTSR Training Pipeline")
    print("="*80)
    
    # Create config
    config = TestConfig()
    
    # Check GPU
    print(f"Device: {config.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("WARNING: No GPU available!")
        return False
    
    print("\n" + "-"*80)
    
    # Set seed
    set_seed(config.seed)
    
    # Prepare dataset
    print("Step 1: Loading dataset...")
    try:
        train_loader, val_loader, class_names = prepare_dataset(config)
        print(f"✓ Dataset loaded: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val")
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return False
    
    print("\n" + "-"*80)
    
    # Build model
    print("Step 2: Building model...")
    try:
        model = build_model(config)
        print(f"✓ Model built: {sum(p.numel() for p in model.parameters()):,} parameters")
    except Exception as e:
        print(f"✗ Model building failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "-"*80)
    
    # Setup training
    print("Step 3: Setting up optimizer and scheduler...")
    try:
        criterion = nn.CrossEntropyLoss()
        optimizer, scheduler = get_optimizer_and_scheduler(model, config, len(train_loader))
        print("✓ Optimizer and scheduler ready")
    except Exception as e:
        print(f"✗ Setup failed: {e}")
        return False
    
    print("\n" + "-"*80)
    
    # Test forward pass
    print("Step 4: Testing forward pass...")
    try:
        model.eval()
        test_input = torch.randn(2, 3, 224, 224).to(config.device)
        with torch.no_grad():
            test_output = model(test_input)
        print(f"✓ Forward pass OK: input {test_input.shape} -> output {test_output.shape}")
        
        # Check GPU memory
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**2
            memory_reserved = torch.cuda.memory_reserved() / 1024**2
            print(f"  GPU memory: {memory_allocated:.1f} MB allocated, {memory_reserved:.1f} MB reserved")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "-"*80)
    
    # Test training one epoch
    print("Step 5: Testing training for 1 epoch...")
    try:
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, config, epoch=0
        )
        print(f"✓ Training epoch completed:")
        print(f"  Loss: {train_loss:.4f}")
        print(f"  Accuracy: {train_acc:.2f}%")
    except Exception as e:
        print(f"✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "-"*80)
    
    # Test validation
    print("Step 6: Testing validation...")
    try:
        val_loss, val_acc = validate(model, val_loader, criterion, config, epoch=0)
        print(f"✓ Validation completed:")
        print(f"  Loss: {val_loss:.4f}")
        print(f"  Accuracy: {val_acc:.2f}%")
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)
    print("\nTraining pipeline is working correctly on RTX 5060 Ti!")
    print("Ready to start full training with: python train_mambatsr_plantvillage.py")
    print("="*80)
    
    return True


if __name__ == '__main__':
    success = test_training()
    sys.exit(0 if success else 1)
