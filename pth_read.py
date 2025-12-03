from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
import re
import random
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import time
import torch
# Import everything you need from your training module
from train_unet_gpu import (
    UNet,
    build_slice_index,
    split_by_patient,
    BrainMetSlices,
    resize_collate,
    bce_dice_loss,
    evaluate_loader,
    eval_tumor_presence,
    visualize_random_predictions,
    ROOT,
    BATCH_SIZE,
    NUM_WORKERS,
    TEST_FRACTION,
    VAL_FRACTION,
    RNG_SEED,
)
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib-config'  # Fix matplotlib warning - makes sure matplotlib has a writable directory instead of a temporary one
'''
This file reads in our neural network with the best parameters and provides plots and images showing the output. 
'''
# ================== LOAD MODEL + DATA ==================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1) Recreate model architecture
model = UNet(in_channels=1, num_classes=1, base_ch=64, use_bn=True, dropout=0.1).to(device)

# 2) Load existing best checkpoint (early-stopped)
best_model_path = Path("./unet_brain_tumor_earlystopped.pth")
ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])

# 3) Restore histories from checkpoint
train_loss_history = ckpt["train_loss_history"]
val_loss_history   = ckpt["val_loss_history"]
train_acc_history  = ckpt["train_acc_history"]
val_acc_history    = ckpt["val_acc_history"]
train_dice_history = ckpt["train_dice_history"]
val_dice_history   = ckpt["val_dice_history"]
lr_history         = ckpt["lr_history"]

criterion = bce_dice_loss  # for evaluate_loader

# 4) Rebuild val/test datasets + loaders with the SAME split logic
index = build_slice_index(ROOT)
split = split_by_patient(
    index,
    test_fraction=TEST_FRACTION,
    val_fraction=VAL_FRACTION,
    seed=RNG_SEED,
)

val_ds  = BrainMetSlices(split["val"],  augment=False)
test_ds = BrainMetSlices(split["test"], augment=False)

val_loader = torch.utils.data.DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    collate_fn=resize_collate,
)

test_loader = torch.utils.data.DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    collate_fn=resize_collate,
)

# ================== TEST EVALUATION ==================

print("\nReloading best model (early-stopped weights)...")
# (Already loaded above; this is just informational)

print("\n" + "="*60)
print("TEST SET EVALUATION (BEST MODEL)")
print("="*60)
test_metrics = evaluate_loader(model, test_loader, device, criterion, desc="TEST (BEST)")
test_loss = test_metrics["loss"]
test_acc  = test_metrics["accuracy"]
test_dice = test_metrics["dice"]
print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test Dice: {test_dice:.4f}")

# ================== VALIDATION TUMOR PRESENCE ==================

val_metrics = eval_tumor_presence(
    model,
    val_loader,
    device,
    prob_thresh=0.5,
    min_positive_pixels=1,
)
print("\nValidation metrics (tumor presence):")
for k, v in val_metrics.items():
    print(f"{k}: {v}")

# ================== VISUALIZATION ==================

print("\nVisualizing random validation examples...")
fig = visualize_random_predictions(
    model,
    val_ds,
    device,
    num_examples=6,
    prob_thresh=0.5,
    seed=123,
    save_path="./results/validation_visualization.png",
)

print("\nVisualizing random TEST examples...")
fig_test = visualize_random_predictions(
    model,
    test_ds,
    device,
    num_examples=6,
    prob_thresh=0.5,
    seed=999,  # different seed for test examples
    save_path="./results/test_visualization.png",
)

# ================== FINAL PLOTS ==================

# 1) LR and Loss vs Epoch
fig, ax1 = plt.subplots(figsize=(7, 4))
ax1.plot(val_loss_history,   color="tab:blue",   label="Val Loss")
ax1.plot(train_loss_history, color="tab:purple", label="Train Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")

ax2 = ax1.twinx()
ax2.plot(lr_history, color="tab:red", linestyle="--", label="Learning Rate")
ax2.set_ylabel("Learning Rate", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")

fig.suptitle("Training Loss and Learning Rate vs. Epoch")
fig.legend()
fig.tight_layout()
plt.savefig("./results/loss_lr_plot.png", dpi=300, bbox_inches="tight")
plt.show()

# ---------- LOSS VS EPOCH ----------
fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(val_loss_history,   color="tab:blue",   label="Val Loss")
ax.plot(train_loss_history, color="tab:purple", label="Train Loss")

# Test loss horizontal line
ax.axhline(test_loss, color="tab:green", linestyle="--",
           label=f"Test Loss = {test_loss:.3f}")

ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Loss vs Epoch")
ax.grid(True, alpha=0.3)
ax.legend(loc="best")

fig.tight_layout()
plt.savefig("./results/loss_plot.png", dpi=300, bbox_inches="tight")
plt.show()

# ---------- ACCURACY VS EPOCH ----------
fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(val_acc_history,   color="tab:red",  linestyle="--", label="Val Accuracy")
ax.plot(train_acc_history, color="tab:pink", linestyle="--", label="Train Accuracy")

# Test accuracy horizontal line
ax.axhline(test_acc, color="tab:orange", linestyle=":",
           label=f"Test Acc = {test_acc:.3f}")

ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy vs Epoch")
ax.grid(True, alpha=0.3)
ax.legend(loc="best")

fig.tight_layout()
plt.savefig("./results/accuracy_plot.png", dpi=300, bbox_inches="tight")
plt.show()


# 3) Dice vs Epoch (with test horizontal line)
fig, ax1 = plt.subplots(figsize=(7, 4))
ax1.plot(val_dice_history,   color="tab:blue", label="Val Dice")
ax1.plot(train_dice_history, color="tab:red",  label="Train Dice")
ax1.axhline(test_dice, color="tab:green", linestyle="--",
            label=f"Test Dice = {test_dice:.3f}")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Dice", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")
fig.suptitle("Dice vs. Epoch")
fig.legend()
fig.tight_layout()
plt.savefig("./results/dice_plot.png", dpi=300, bbox_inches="tight")
plt.show()