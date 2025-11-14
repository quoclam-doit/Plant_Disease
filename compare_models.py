#!/usr/bin/env python3
"""
Compare VSSM and Super_Mamba training results
Tạo biểu đồ so sánh giữa 2 models
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Load training histories
def load_history(model_dir, model_name):
    history_path = os.path.join('models', model_dir, 'training_history.json')
    if not os.path.exists(history_path):
        print(f"Warning: {history_path} not found!")
        return None
    
    with open(history_path, 'r') as f:
        data = json.load(f)
    
    # Add model info
    data['model_name'] = model_name
    return data

# Load both models
vssm_history = load_history('MambaTSR', 'VSSM')
supermamba_history = load_history('SuperMamba', 'Super_Mamba')

if vssm_history is None or supermamba_history is None:
    print("Error: Cannot load training histories!")
    exit(1)

# Extract metrics
vssm_train_loss = vssm_history['train_loss']
vssm_train_acc = vssm_history['train_acc']
vssm_val_loss = vssm_history['val_loss']
vssm_val_acc = vssm_history['val_acc']
vssm_epochs = range(1, len(vssm_train_loss) + 1)

sm_train_loss = supermamba_history['train_loss']
sm_train_acc = supermamba_history['train_acc']
sm_val_loss = supermamba_history['val_loss']
sm_val_acc = supermamba_history['val_acc']
sm_epochs = range(1, len(sm_train_loss) + 1)

# Create comparison plots
fig = plt.figure(figsize=(20, 12))

# 1. Training Loss Comparison
plt.subplot(2, 3, 1)
plt.plot(vssm_epochs, vssm_train_loss, 'b-', label='VSSM (77M params)', linewidth=2)
plt.plot(sm_epochs, sm_train_loss, 'r-', label='Super_Mamba (1.3M params)', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Training Loss', fontsize=12)
plt.title('Training Loss Comparison', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# 2. Validation Loss Comparison
plt.subplot(2, 3, 2)
plt.plot(vssm_epochs, vssm_val_loss, 'b-', label='VSSM (77M params)', linewidth=2)
plt.plot(sm_epochs, sm_val_loss, 'r-', label='Super_Mamba (1.3M params)', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Validation Loss', fontsize=12)
plt.title('Validation Loss Comparison', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# 3. Training Accuracy Comparison
plt.subplot(2, 3, 4)
plt.plot(vssm_epochs, vssm_train_acc, 'b-', label='VSSM (77M params)', linewidth=2)
plt.plot(sm_epochs, sm_train_acc, 'r-', label='Super_Mamba (1.3M params)', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Training Accuracy (%)', fontsize=12)
plt.title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# 4. Validation Accuracy Comparison
plt.subplot(2, 3, 5)
plt.plot(vssm_epochs, vssm_val_acc, 'b-', label='VSSM (77M params)', linewidth=2)
plt.plot(sm_epochs, sm_val_acc, 'r-', label='Super_Mamba (1.3M params)', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Validation Accuracy (%)', fontsize=12)
plt.title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# 5. Final Performance Bar Chart
plt.subplot(2, 3, 3)
models = ['VSSM\n(77M)', 'Super_Mamba\n(1.3M)']
final_val_acc = [max(vssm_val_acc), max(sm_val_acc)]
colors = ['#2E86C1', '#E74C3C']
bars = plt.bar(models, final_val_acc, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

# Add value labels on bars
for bar, acc in zip(bars, final_val_acc):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{acc:.2f}%',
             ha='center', va='bottom', fontsize=14, fontweight='bold')

plt.ylabel('Best Validation Accuracy (%)', fontsize=12)
plt.title('Final Performance Comparison', fontsize=14, fontweight='bold')
plt.ylim([min(final_val_acc) - 5, 100])
plt.grid(True, axis='y', alpha=0.3)

# 6. Model Efficiency (Accuracy per Million Parameters)
plt.subplot(2, 3, 6)
vssm_params = 77.0  # Million
sm_params = 1.3    # Million
vssm_efficiency = max(vssm_val_acc) / vssm_params
sm_efficiency = max(sm_val_acc) / sm_params

models = ['VSSM\n(77M)', 'Super_Mamba\n(1.3M)']
efficiency = [vssm_efficiency, sm_efficiency]
colors = ['#2E86C1', '#E74C3C']
bars = plt.bar(models, efficiency, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

# Add value labels on bars
for bar, eff in zip(bars, efficiency):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{eff:.2f}',
             ha='center', va='bottom', fontsize=14, fontweight='bold')

plt.ylabel('Accuracy per Million Parameters', fontsize=12)
plt.title('Model Efficiency Comparison', fontsize=14, fontweight='bold')
plt.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved: model_comparison.png")

# Print summary statistics
print("\n" + "="*70)
print("MODEL COMPARISON SUMMARY")
print("="*70)

print("\n📊 VSSM (77M parameters):")
print(f"   Best Training Accuracy: {max(vssm_train_acc):.2f}%")
print(f"   Best Validation Accuracy: {max(vssm_val_acc):.2f}%")
print(f"   Final Training Loss: {vssm_train_loss[-1]:.4f}")
print(f"   Final Validation Loss: {vssm_val_loss[-1]:.4f}")
print(f"   Total Epochs: {len(vssm_train_loss)}")

print("\n📊 Super_Mamba (1.3M parameters):")
print(f"   Best Training Accuracy: {max(sm_train_acc):.2f}%")
print(f"   Best Validation Accuracy: {max(sm_val_acc):.2f}%")
print(f"   Final Training Loss: {sm_train_loss[-1]:.4f}")
print(f"   Final Validation Loss: {sm_val_loss[-1]:.4f}")
print(f"   Total Epochs: {len(sm_train_loss)}")

print("\n🔍 COMPARISON:")
acc_diff = max(sm_val_acc) - max(vssm_val_acc)
param_ratio = vssm_params / sm_params
efficiency_ratio = sm_efficiency / vssm_efficiency

print(f"   Accuracy Difference: {acc_diff:+.2f}%")
print(f"   Parameter Reduction: {param_ratio:.1f}x smaller (Super_Mamba)")
print(f"   Efficiency Gain: {efficiency_ratio:.2f}x better (Accuracy/Param)")

if acc_diff > -2.0:  # Less than 2% accuracy drop
    print(f"\n✅ Super_Mamba achieved comparable accuracy with {param_ratio:.1f}x fewer parameters!")
else:
    print(f"\n⚠️ Super_Mamba has {abs(acc_diff):.2f}% lower accuracy but {param_ratio:.1f}x fewer parameters")

print("\n" + "="*70)
