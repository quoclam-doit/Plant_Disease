"""
MambaTSR Training Script for PlantVillage Dataset
RTX 5060 Ti (sm_120) Compatible
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
import json
from datetime import datetime

# Auto-detect base directory (Windows vs WSL)
if os.name == 'nt':  # Windows
    BASE_DIR = r'G:\Dataset'
else:  # Linux/WSL
    BASE_DIR = '/mnt/g/Dataset'

MAMBATSR_DIR = os.path.join(BASE_DIR, 'MambaTSR')

# Add MambaTSR to path
sys.path.insert(0, MAMBATSR_DIR)

# Import MambaTSR model
import importlib.util
vmamba_path = os.path.join(MAMBATSR_DIR, 'models', 'vmamba.py')
spec = importlib.util.spec_from_file_location("vmamba_module", vmamba_path)
vmamba_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vmamba_module)
VSSM = vmamba_module.VSSM


class MambaTSRConfig:
    """Configuration for MambaTSR training"""
    # Dataset
    data_root = os.path.join(BASE_DIR, 'Data', 'PlantVillage', 'PlantVillage-Dataset-master')
    exclude_folder = 'x_Removed_from_Healthy_leaves'
    
    # Model
    patch_size = 4
    in_chans = 3
    num_classes = 38  # PlantVillage has 38 disease classes
    depths = [2, 2, 9, 2]  # VSSM-Tiny architecture
    dims = [96, 192, 384, 768]
    drop_path_rate = 0.1
    
    # Training
    batch_size = 32  # Increased from 16 - RTX 5060 Ti has 16GB VRAM
    num_epochs = 50
    learning_rate = 1e-4
    weight_decay = 0.05
    warmup_epochs = 5
    num_workers = 4  # DataLoader workers
    
    # Data split
    train_ratio = 0.8
    val_ratio = 0.2
    
    # Image size
    img_size = 64  # Reduced from 224 for faster training (16x speedup!)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Checkpoints
    save_dir = os.path.join(BASE_DIR, 'models', 'MambaTSR')
    log_interval = 50
    save_interval = 5  # Save every 5 epochs
    
    # Random seed
    seed = 42


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_data_transforms(img_size=224):
    """Get data augmentation transforms"""
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def prepare_dataset(config):
    """Prepare PlantVillage dataset"""
    print("Loading PlantVillage dataset...")
    
    # Get transforms
    train_transform, val_transform = get_data_transforms(config.img_size)
    
    # Load full dataset
    full_dataset = datasets.ImageFolder(config.data_root, transform=train_transform)
    
    # Filter out excluded folder
    if config.exclude_folder:
        indices = [i for i, (path, _) in enumerate(full_dataset.samples) 
                   if config.exclude_folder not in path]
        full_dataset = torch.utils.data.Subset(full_dataset, indices)
    
    # Get class names and count
    class_names = full_dataset.dataset.classes if hasattr(full_dataset, 'dataset') else full_dataset.classes
    num_classes = len(class_names)
    
    print(f"Total images: {len(full_dataset)}")
    print(f"Number of classes: {num_classes}")
    
    # Split into train and validation
    train_size = int(config.train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed)
    )
    
    # Update validation transform
    val_dataset.dataset.transform = val_transform
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True if config.num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,  # Larger batch for validation
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True if config.num_workers > 0 else False
    )
    
    return train_loader, val_loader, class_names


def build_model(config):
    """Build MambaTSR model"""
    print("Building MambaTSR model...")
    
    model = VSSM(
        patch_size=config.patch_size,
        in_chans=config.in_chans,
        num_classes=config.num_classes,
        depths=config.depths,
        dims=config.dims,
        drop_path_rate=config.drop_path_rate
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    model = model.to(config.device)
    
    return model


def get_optimizer_and_scheduler(model, config, steps_per_epoch):
    """Setup optimizer and learning rate scheduler"""
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Cosine annealing with warmup
    warmup_steps = config.warmup_epochs * steps_per_epoch
    total_steps = config.num_epochs * steps_per_epoch
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265359))))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    return optimizer, scheduler


def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, config, epoch):
    """Train for one epoch"""
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Train]")
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(config.device), targets.to(config.device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        if (batch_idx + 1) % config.log_interval == 0:
            avg_loss = running_loss / (batch_idx + 1)
            accuracy = 100. * correct / total
            lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{accuracy:.2f}%',
                'lr': f'{lr:.6f}'
            })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, val_loader, criterion, config, epoch):
    """Validate the model"""
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Val]  ")
    
    for inputs, targets in pbar:
        inputs, targets = inputs.to(config.device), targets.to(config.device)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    print(f"Validation - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
    
    return epoch_loss, epoch_acc


def save_checkpoint(model, optimizer, epoch, val_acc, config, is_best=False):
    """Save model checkpoint"""
    os.makedirs(config.save_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'config': {
            'num_classes': config.num_classes,
            'depths': config.depths,
            'dims': config.dims,
            'patch_size': config.patch_size,
            'in_chans': config.in_chans,
            'drop_path_rate': config.drop_path_rate
        }
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(config.save_dir, f'mambatsr_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save best model
    if is_best:
        best_path = os.path.join(config.save_dir, 'mambatsr_best.pth')
        torch.save(checkpoint, best_path)
        print(f"Best model saved: {best_path}")


def train(config):
    """Main training function"""
    print("="*80)
    print("MambaTSR Training on PlantVillage Dataset")
    print("RTX 5060 Ti (sm_120) Compatible")
    print("="*80)
    print(f"Device: {config.device}")
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Learning rate: {config.learning_rate}")
    print("="*80)
    
    # Set seed
    set_seed(config.seed)
    
    # Prepare dataset
    train_loader, val_loader, class_names = prepare_dataset(config)
    
    # Save class names
    os.makedirs(config.save_dir, exist_ok=True)
    with open(os.path.join(config.save_dir, 'class_names.json'), 'w') as f:
        json.dump(class_names, f, indent=2)
    
    # Build model
    model = build_model(config)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = get_optimizer_and_scheduler(model, config, len(train_loader))
    
    # Training loop
    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    print("\nStarting training...")
    start_time = datetime.now()
    
    for epoch in range(config.num_epochs):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, config, epoch
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, config, epoch)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{config.num_epochs} Summary:")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        # Save checkpoint
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            print(f"  New best validation accuracy: {best_val_acc:.2f}%")
        
        if (epoch + 1) % config.save_interval == 0 or is_best:
            save_checkpoint(model, optimizer, epoch + 1, val_acc, config, is_best)
        
        print("-"*80)
    
    # Training complete
    elapsed_time = datetime.now() - start_time
    print("\n" + "="*80)
    print("Training Complete!")
    print(f"Time elapsed: {elapsed_time}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print("="*80)
    
    # Save final history
    history_path = os.path.join(config.save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved: {history_path}")
    
    return model, history


if __name__ == '__main__':
    # Create config
    config = MambaTSRConfig()
    
    # Check GPU
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available! Training on CPU will be very slow.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            exit()
    else:
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Train
    model, history = train(config)
    
    print("\n" + "="*80)
    print("📊 Generating Training Plots...")
    print("="*80)
    
    # Plot training curves
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('MambaTSR Training Results - PlantVillage', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Plot 1: Training & Validation Loss
        axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Training & Validation Accuracy
        axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
        axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[0, 1].set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Loss comparison
        axes[1, 0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2, alpha=0.7)
        axes[1, 0].plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2, alpha=0.7)
        axes[1, 0].fill_between(epochs, history['train_loss'], history['val_loss'], 
                                alpha=0.2, color='purple', label='Gap')
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Loss', fontsize=12)
        axes[1, 0].set_title('Train-Val Loss Gap (Overfitting Check)', fontsize=14, fontweight='bold')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Learning Rate Schedule
        axes[1, 1].plot(epochs, history['lr'], 'g-', linewidth=2)
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
        axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(config.save_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training plots saved: {plot_path}")
        
        # Also save individual plots for easy viewing
        # Loss only
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Loss Curves', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        loss_path = os.path.join(config.save_dir, 'loss_curve.png')
        plt.savefig(loss_path, dpi=300, bbox_inches='tight')
        print(f"✓ Loss curve saved: {loss_path}")
        plt.close()
        
        # Accuracy only
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
        plt.plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('Accuracy Curves', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        acc_path = os.path.join(config.save_dir, 'accuracy_curve.png')
        plt.savefig(acc_path, dpi=300, bbox_inches='tight')
        print(f"✓ Accuracy curve saved: {acc_path}")
        plt.close()
        
        print("\n✅ All plots generated successfully!")
        
    except Exception as e:
        print(f"⚠️ Error generating plots: {e}")
        print("Training history is still saved in JSON format.")
    
    print("\n" + "="*80)
    print("🎉 Training Complete! All results saved.")
    print("="*80)
    print(f"\n📁 Results location: {config.save_dir}")
    print(f"  - Best model: mambatsr_best.pth")
    print(f"  - Training curves: training_curves.png")
    print(f"  - Loss curve: loss_curve.png")
    print(f"  - Accuracy curve: accuracy_curve.png")
    print(f"  - Training history: training_history.json")
    print("\n✅ Training script finished successfully! 🎉")
