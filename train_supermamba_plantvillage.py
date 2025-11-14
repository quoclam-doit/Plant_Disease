"""
Train Super_Mamba on PlantVillage Dataset
- Model: Super_Mamba (4M parameters) from MambaTSR/models/VSSBlock_utils.py
- Dataset: PlantVillage (54,304 images, 39 classes)
- Image size: 64×64
- Architecture: 100% ORIGINAL (no modifications)
"""

import os
import sys
import time
import json
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

# Auto-detect platform (Windows vs WSL)
if os.name == 'nt':  # Windows
    BASE_DIR = r'G:\Dataset'
else:  # WSL/Linux
    BASE_DIR = '/mnt/g/Dataset'

MAMBATSR_DIR = os.path.join(BASE_DIR, 'MambaTSR')
sys.path.insert(0, MAMBATSR_DIR)

# Import Super_Mamba model (100% ORIGINAL - NO MODIFICATIONS)
import importlib.util
vssblock_utils_path = os.path.join(MAMBATSR_DIR, 'models', 'VSSBlock_utils.py')
spec = importlib.util.spec_from_file_location("vssblock_utils_module", vssblock_utils_path)
vssblock_utils_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vssblock_utils_module)
Super_Mamba = vssblock_utils_module.Super_Mamba


class SuperMambaConfig:
    """Configuration for Super_Mamba training"""
    # Dataset
    data_root = os.path.join(BASE_DIR, 'Data', 'PlantVillage', 'PlantVillage-Dataset-master')
    
    # Model architecture (KEEP ORIGINAL)
    dims = 3           # Starting dimension (default from original)
    depth = 6          # Number of layers (default from original)
    num_classes = 39   # PlantVillage has 39 classes
    
    # Training hyperparameters
    batch_size = 32
    num_epochs = 50
    learning_rate = 1e-4
    weight_decay = 0.05
    
    # Image size
    img_size = 64      # Optimized for speed
    
    # Data augmentation
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Checkpointing
    checkpoint_dir = os.path.join(BASE_DIR, 'models', 'SuperMamba')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def prepare_dataset(config):
    """Prepare PlantVillage dataset"""
    print(f"\n{'='*70}")
    print("LOADING PLANTVILLAGE DATASET")
    print(f"{'='*70}")
    
    # Load full dataset
    full_dataset = datasets.ImageFolder(root=config.data_root)
    print(f"Total images: {len(full_dataset)}")
    print(f"Number of classes: {len(full_dataset.classes)}")
    
    # Filter out grayscale images (keep only RGB)
    rgb_indices = []
    print("\nFiltering RGB images...")
    for idx in tqdm(range(len(full_dataset)), desc="Checking images"):
        img, _ = full_dataset[idx]
        if img.mode == 'RGB':
            rgb_indices.append(idx)
    
    print(f"RGB images: {len(rgb_indices)}")
    print(f"Filtered out: {len(full_dataset) - len(rgb_indices)} grayscale images")
    
    # Create subset with only RGB images
    full_dataset = torch.utils.data.Subset(full_dataset, rgb_indices)
    
    # Split into train/val (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Apply transforms
    train_dataset.dataset.dataset.transform = config.train_transforms
    val_dataset.dataset.dataset.transform = config.val_transforms
    
    print(f"\nDataset split:")
    print(f"   Training: {len(train_dataset)} images")
    print(f"   Validation: {len(val_dataset)} images")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Get class names
    class_names = datasets.ImageFolder(root=config.data_root).classes
    
    return train_loader, val_loader, class_names


def build_model(config):
    """Build Super_Mamba model (100% ORIGINAL ARCHITECTURE)"""
    print(f"\n{'='*70}")
    print("BUILDING SUPER_MAMBA MODEL")
    print(f"{'='*70}")
    
    # Create Super_Mamba with ORIGINAL architecture
    model = Super_Mamba(
        dims=config.dims,           # Keep original: 3
        depth=config.depth,         # Keep original: 6
        num_classes=config.num_classes  # Change only num_classes: 39 for PlantVillage
    )
    
    model = model.to(config.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model: Super_Mamba")
    print(f"Architecture: dims={config.dims}, depth={config.depth}")
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
    print(f"Device: {config.device}")
    
    return model


def train_one_epoch(model, train_loader, criterion, optimizer, config, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Train]")
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(config.device), labels.to(config.device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, config, epoch):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Val]")
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(config.device), labels.to(config.device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def save_checkpoint(model, optimizer, epoch, train_acc, val_acc, config, filename):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_acc': train_acc,
        'val_acc': val_acc,
        'config': {
            'dims': config.dims,
            'depth': config.depth,
            'num_classes': config.num_classes,
            'img_size': config.img_size,
        }
    }
    filepath = os.path.join(config.checkpoint_dir, filename)
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def plot_training_history(history, config):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(config.checkpoint_dir, 'training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training plot saved: {plot_path}")
    plt.close()


def main():
    """Main training function"""
    print("\n" + "="*70)
    print("SUPER_MAMBA PLANTVILLAGE TRAINING")
    print("="*70)
    
    # Configuration
    config = SuperMambaConfig()
    
    print(f"\nConfiguration:")
    print(f"   Model: Super_Mamba (dims={config.dims}, depth={config.depth})")
    print(f"   Dataset: PlantVillage (39 classes)")
    print(f"   Image size: {config.img_size}x{config.img_size}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Epochs: {config.num_epochs}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Device: {config.device}")
    
    # Prepare dataset
    train_loader, val_loader, class_names = prepare_dataset(config)
    
    # Build model (100% ORIGINAL ARCHITECTURE)
    model = build_model(config)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    start_time = time.time()
    
    print(f"\n{'='*70}")
    print("TRAINING START")
    print(f"{'='*70}\n")
    
    # Training loop
    for epoch in range(config.num_epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, config, epoch
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, config, epoch
        )
        
        # Update learning rate
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{config.num_epochs} Summary:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"   Time: {epoch_time:.2f}s | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch, train_acc, val_acc, config,
                'supermamba_best.pth'
            )
            print(f"   New best validation accuracy: {best_val_acc:.2f}%")
        
        print()
    
    # Training complete
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Final train accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"Final val accuracy: {history['val_acc'][-1]:.2f}%")
    
    # Save final model
    save_checkpoint(
        model, optimizer, config.num_epochs-1,
        history['train_acc'][-1], history['val_acc'][-1],
        config, 'supermamba_final.pth'
    )
    
    # Save training history
    history_path = os.path.join(config.checkpoint_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved: {history_path}")
    
    # Plot training curves
    plot_training_history(history, config)
    
    # Save class mappings
    class_mapping_path = os.path.join(config.checkpoint_dir, 'class_mappings.json')
    with open(class_mapping_path, 'w') as f:
        json.dump({i: name for i, name in enumerate(class_names)}, f, indent=4)
    print(f"Class mappings saved: {class_mapping_path}")
    
    print(f"\n{'='*70}")
    print("ALL DONE!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
