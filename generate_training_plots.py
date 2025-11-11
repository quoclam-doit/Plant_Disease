#!/usr/bin/env python3
"""
Generate training visualization plots from training_history.json
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 11

# Load training history
history_path = Path('models/MambaTSR/training_history.json')
with open(history_path, 'r') as f:
    history = json.load(f)

# Extract data
epochs = list(range(1, len(history['train_loss']) + 1))
train_loss = history['train_loss']
val_loss = history['val_loss']
train_acc = history['train_acc']
val_acc = history['val_acc']

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('MambaTSR Training on PlantVillage Dataset\n64×64 Images, 50 Epochs, Best Val Acc: 98.96%', 
             fontsize=16, fontweight='bold')

# Plot 1: Training & Validation Loss
ax1 = axes[0, 0]
ax1.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2, marker='o', markersize=4, markevery=5)
ax1.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=4, markevery=5)
ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, 51])

# Add best epoch marker on loss plot
best_epoch = np.argmax(val_acc) + 1
best_val_loss = val_loss[best_epoch - 1]
ax1.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, linewidth=2, label=f'Best Epoch: {best_epoch}')
ax1.plot(best_epoch, best_val_loss, 'g*', markersize=20, label=f'Best Val Loss: {best_val_loss:.4f}')
ax1.legend(fontsize=10, loc='upper right')

# Plot 2: Training & Validation Accuracy
ax2 = axes[0, 1]
ax2.plot(epochs, train_acc, 'b-', label='Training Accuracy', linewidth=2, marker='o', markersize=4, markevery=5)
ax2.plot(epochs, val_acc, 'r-', label='Validation Accuracy', linewidth=2, marker='s', markersize=4, markevery=5)
ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11, loc='lower right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim([0, 51])
ax2.set_ylim([0, 105])

# Add 99% line
ax2.axhline(y=99, color='purple', linestyle='--', alpha=0.5, linewidth=1.5, label='99% Target')

# Add best epoch marker
best_val_acc = max(val_acc)
ax2.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, linewidth=2)
ax2.plot(best_epoch, best_val_acc, 'g*', markersize=20, label=f'Best: {best_val_acc:.2f}% @ Epoch {best_epoch}')
ax2.legend(fontsize=10, loc='lower right')

# Plot 3: Overfitting Gap (Train - Val Accuracy)
ax3 = axes[1, 0]
gap = [t - v for t, v in zip(train_acc, val_acc)]
ax3.plot(epochs, gap, 'purple', linewidth=2, marker='D', markersize=4, markevery=5)
ax3.fill_between(epochs, 0, gap, alpha=0.3, color='purple')
ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax3.set_ylabel('Accuracy Gap (%)', fontsize=12, fontweight='bold')
ax3.set_title('Overfitting Analysis (Train - Val Accuracy)', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_xlim([0, 51])
ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)

# Add ideal zone
ax3.axhspan(0, 2, alpha=0.2, color='green', label='Ideal Zone (< 2%)')
ax3.axhspan(2, 5, alpha=0.2, color='yellow', label='Warning Zone (2-5%)')
ax3.legend(fontsize=10, loc='upper right')

# Final gap
final_gap = gap[-1]
ax3.text(50, final_gap, f'Final: {final_gap:.2f}%', fontsize=11, fontweight='bold', 
         ha='right', va='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 4: Learning Rate Schedule & Key Milestones
ax4 = axes[1, 1]

# Key milestones
milestones = [
    (1, 63.60, "Start: 63.60%"),
    (10, 92.74, "Epoch 10: 92.74%"),
    (20, 95.75, "Epoch 20: 95.75%"),
    (30, 97.91, "Epoch 30: 97.91%"),
    (40, 98.58, "Epoch 40: 98.58%"),
    (best_epoch, best_val_acc, f"Best: {best_val_acc:.2f}%"),
    (50, val_acc[-1], f"Final: {val_acc[-1]:.2f}%")
]

milestone_epochs = [m[0] for m in milestones]
milestone_accs = [m[1] for m in milestones]
milestone_labels = [m[2] for m in milestones]

# Plot validation accuracy curve
ax4.plot(epochs, val_acc, 'b-', linewidth=3, alpha=0.6, label='Validation Accuracy')
ax4.scatter(milestone_epochs, milestone_accs, s=200, c='red', marker='*', 
           zorder=5, edgecolors='black', linewidth=1.5, label='Key Milestones')

# Add annotations
for i, (ep, acc, label) in enumerate(milestones):
    if i < len(milestones) - 1:  # Not the last one
        ax4.annotate(label, xy=(ep, acc), xytext=(10, -15 if i % 2 == 0 else 15),
                    textcoords='offset points', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=1.5))
    else:  # Last milestone
        ax4.annotate(label, xy=(ep, acc), xytext=(-50, 15),
                    textcoords='offset points', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=1.5))

ax4.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax4.set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
ax4.set_title('Training Progress & Key Milestones', fontsize=14, fontweight='bold')
ax4.legend(fontsize=10, loc='lower right')
ax4.grid(True, alpha=0.3)
ax4.set_xlim([0, 51])
ax4.set_ylim([60, 100])

# Add 99% target line
ax4.axhline(y=99, color='purple', linestyle='--', alpha=0.5, linewidth=2, label='99% Target')

plt.tight_layout()

# Save plot
output_path = Path('models/MambaTSR/training_curves_complete.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_path}")

# Create individual plots
print("\nGenerating individual plots...")

# Individual Loss Plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2, marker='o', markersize=3, markevery=5)
plt.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=3, markevery=5)
plt.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, linewidth=2)
plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.ylabel('Loss', fontsize=12, fontweight='bold')
plt.title('MambaTSR Training Loss\n64×64 Images, PlantVillage Dataset', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xlim([0, 51])
output_path = Path('models/MambaTSR/loss_curve.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_path}")
plt.close()

# Individual Accuracy Plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_acc, 'b-', label='Training Accuracy', linewidth=2, marker='o', markersize=3, markevery=5)
plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy', linewidth=2, marker='s', markersize=3, markevery=5)
plt.axhline(y=99, color='purple', linestyle='--', alpha=0.5, linewidth=1.5, label='99% Target')
plt.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, linewidth=2)
plt.plot(best_epoch, best_val_acc, 'g*', markersize=20, label=f'Best: {best_val_acc:.2f}%')
plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
plt.title('MambaTSR Training Accuracy\n64×64 Images, Best: 98.96%', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xlim([0, 51])
plt.ylim([0, 105])
output_path = Path('models/MambaTSR/accuracy_curve.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_path}")
plt.close()

print("\n" + "="*60)
print("📊 TRAINING SUMMARY:")
print("="*60)
print(f"Total Epochs: {len(epochs)}")
print(f"Best Epoch: {best_epoch}")
print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
print(f"Final Training Accuracy: {train_acc[-1]:.2f}%")
print(f"Final Validation Accuracy: {val_acc[-1]:.2f}%")
print(f"Final Overfitting Gap: {gap[-1]:.2f}%")
print(f"Final Training Loss: {train_loss[-1]:.4f}")
print(f"Final Validation Loss: {val_loss[-1]:.4f}")
print("="*60)
print("✅ All plots generated successfully!")
print("="*60)
