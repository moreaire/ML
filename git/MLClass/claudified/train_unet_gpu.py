#!/usr/bin/env python3
"""
Brain Tumor Segmentation with U-Net - GPU Optimized
====================================================
Training script for brain metastases segmentation using U-Net with RTX 4090 GPU acceleration.

Expected performance:
- Training time: ~6-10 minutes for 120 epochs (vs 5.5 hours on CPU)
- GPU utilization: ~80-90% during training
- System remains responsive during training

Author: Claude + User
Date: 2025-11-25
"""

# ============= CPU THREAD CONTROL & GPU SETUP =============
import os
os.environ['OMP_NUM_THREADS'] = '4' #Prevents uncontolled usage of CPU to 4 threads with OpenMP (Used by PyTorch)
os.environ['MKL_NUM_THREADS'] = '4' #Avoids resource overload of the Intel MKL (Math Kernel Library) by limiting to 4 threads
os.environ['NUMEXPR_NUM_THREADS'] = '4' #Similar as above. Set for consistency
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib-config'  # Fix matplotlib warning - makes sure matplotlib has a writable directory instead of a temporary one

import torch
torch.set_num_threads(4) #Controls the number of threads each operation (matrix multiplication, convolution) in pytorch can use
torch.set_num_interop_threads(2)
# ==========================================================

from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
import re
import random
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import time
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

# ==================== CONFIGURATION ====================
# UPDATE THIS PATH TO YOUR DATA LOCATION!
ROOT = Path("/config/workspace/git/MLClass/ML project")  # ← CHANGE THIS!
# Alternative paths you might need:
# ROOT = Path("/config/workspace/projects/local/MLproject/ML project")
# ROOT = Path("../ML project")

RNG_SEED = 42
VAL_FRACTION = 0.15   # of TRAIN (patient-wise)
TEST_FRACTION = 0.20  # patient-wise
TARGET_SIZE = (256, 256)  # (H, W). Standardizes the input image size

# GPU Training Parameters
BATCH_SIZE = 16        # Optimized for RTX 4090 (can increase to 32-64)
NUM_WORKERS = 4        # Parallel data loading
NUM_EPOCHS = 120       # created from total data / batch size
LEARNING_RATE = 1e-3   # Starting learning rate. Commonly used for ADAM
WEIGHT_DECAY = 1e-4    # Common starting weight decay for regularization
# =======================================================

def print_system_info():
    """Print system configuration and GPU status"""
    print("=" * 60)
    print("SYSTEM CONFIGURATION")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if hasattr(torch.version, 'cuda') and torch.version.cuda:
        print(f"CUDA version: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    else:
        print("⚠️  GPU NOT AVAILABLE - Running on CPU")
    print("=" * 60)

def print_gpu_memory():
    """Monitor GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

# ==================== DATA UTILITIES ====================

def is_patient_dir(p: Path) -> bool:
    return p.is_dir() and re.match(r"^p\d+$", p.name, flags=re.I) is not None

def find_cleaned_dirs(patient_dir: Path):
    """Find MRI and Mask directories under patient/cleaned/"""
    cleaned = patient_dir / "cleaned"
    if not cleaned.is_dir():
        return None, None

    img_dir = None
    mask_dir = None
    for d in cleaned.iterdir():
        if not d.is_dir():
            continue
        name = d.name.lower()
        if name.endswith("_mri_slices"):
            img_dir = d
        elif name.endswith("_mask_slices"):
            mask_dir = d
    return img_dir, mask_dir

def list_dicom_sorted(d: Path):
    """List DICOM files sorted by InstanceNumber"""
    files = [f for f in d.iterdir() if f.is_file()]
    def sort_key(p: Path):
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
            inst = getattr(ds, "InstanceNumber", None)
            return (inst if inst is not None else 10**9, p.name)
        except Exception:
            return (10**9, p.name)
    return sorted(files, key=sort_key)

def map_by_uid(img_files, mask_files):
    """Pair images and masks by SOPInstanceUID"""
    def uid_map(files):
        m = {}
        for f in files:
            try:
                ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
                uid = str(getattr(ds, "SOPInstanceUID", ""))
                if uid:
                    m[uid] = f
            except Exception:
                pass
        return m

    img_uids = uid_map(img_files)
    mask_uids = uid_map(mask_files)

    pairs = []
    if img_uids and mask_uids:
        common = [u for u in img_uids.keys() if u in mask_uids]
        if len(common) >= min(len(img_files), len(mask_files)) * 0.9:
            for u in sorted(common):
                pairs.append((img_uids[u], mask_uids[u]))
            if len(common) != len(img_files) or len(common) != len(mask_files):
                print(f"[WARN] UID pairing dropped some slices: {len(common)} pairs, "
                      f"{len(img_files)} images, {len(mask_files)} masks")
            return pairs

    # Fallback: index pairing
    n = min(len(img_files), len(mask_files))
    if len(img_files) != len(mask_files):
        print(f"[WARN] Index pairing with unequal counts: images={len(img_files)} masks={len(mask_files)}")
    for i in range(n):
        pairs.append((img_files[i], mask_files[i]))
    return pairs

def dicom_to_image(ds):
    """Convert DICOM to numpy array with RescaleSlope/Intercept"""
    arr = ds.pixel_array.astype(np.float32, copy=False)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    if slope != 1.0 or intercept != 0.0:
        arr = arr * slope + intercept
    return arr

def read_dicom(path: Path, stop_before_pixels=False):
    return pydicom.dcmread(str(path), stop_before_pixels=stop_before_pixels, force=True)

def build_slice_index(root: Path):
    """Build index of all image-mask pairs"""
    patients = [p for p in root.iterdir() if is_patient_dir(p)]
    index = []
    for pd in sorted(patients, key=lambda x: int(re.findall(r"\d+", x.name)[0])):
        img_dir, mask_dir = find_cleaned_dirs(pd)
        if not img_dir or not mask_dir:
            print(f"[SKIP] {pd.name}: cleaned MRI/mask dirs not found")
            continue
        img_files = list_dicom_sorted(img_dir)
        mask_files = list_dicom_sorted(mask_dir)
        if not img_files or not mask_files:
            print(f"[SKIP] {pd.name}: no DICOM files in cleaned dirs")
            continue
        pairs = map_by_uid(img_files, mask_files)
        if not pairs:
            print(f"[SKIP] {pd.name}: no slice pairs")
            continue
        for img_p, mask_p in pairs:
            index.append({"patient": pd.name, "img": img_p, "mask": mask_p})
    return index

def split_by_patient(index, test_fraction=0.2, val_fraction=0.15, seed=RNG_SEED):
    """Patient-wise split to avoid data leakage"""
    rng = random.Random(seed)
    patients = sorted({row["patient"] for row in index})
    rng.shuffle(patients)
    n = len(patients)
    n_test = max(1, int(round(n * test_fraction)))
    test_patients = set(patients[:n_test])
    remaining = patients[n_test:]
    n_val = max(1, int(round(len(remaining) * val_fraction))) if remaining else 0
    val_patients = set(remaining[:n_val])
    train_patients = set(remaining[n_val:])

    split = {"train": [], "val": [], "test": []}
    for row in index:
        if row["patient"] in test_patients:
            split["test"].append(row)
        elif row["patient"] in val_patients:
            split["val"].append(row)
        else:
            split["train"].append(row)
    print(f"Patients total={n} | train={len(train_patients)} val={len(val_patients)} test={len(test_patients)}")
    print(f"Slices   train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}")
    return split

def resize_collate(batch):
    """Collate function for DataLoader"""
    imgs, msks, metas = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    msks = torch.stack(msks, dim=0)
    return imgs, msks, list(metas)

def _resize_2d(t: torch.Tensor, size_hw: tuple[int,int], mode: str, align_corners: bool | None):
    """Resize 2D tensor"""
    if t.ndim == 2:
        t = t.unsqueeze(0)
    t = t.unsqueeze(0)
    out = F.interpolate(t, size=size_hw, mode=mode, align_corners=align_corners)
    return out.squeeze(0).squeeze(0)

# ==================== DATA AUGMENTATION ====================

@dataclass
class AugmentConfig:
    """Data augmentation parameters"""
    p_flip_lr: float = 0.5
    contrast_range: tuple[float, float] = (0.9, 1.1)
    brightness_range: tuple[float, float] = (-0.1, 0.1)
    p_gamma: float = 0.3
    gamma_range: tuple[float, float] = (0.9, 1.1)
    p_gauss_noise: float = 0.3
    gauss_sigma: float = 0.05

def _rand_uniform(a: float, b: float, rng: np.random.Generator) -> float:
    return float(rng.uniform(a, b))

def augment_flip_lr(img: np.ndarray, msk: np.ndarray,
                    rng: np.random.Generator, p: float) -> tuple[np.ndarray, np.ndarray]:
    """Random horizontal flip"""
    if rng.random() < p:
        img = np.ascontiguousarray(img[:, ::-1])
        msk = np.ascontiguousarray(msk[:, ::-1])
    return img, msk

def augment_intensity(img: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig) -> np.ndarray:
    """Random intensity augmentation"""
    c = _rand_uniform(*cfg.contrast_range, rng)
    b = _rand_uniform(*cfg.brightness_range, rng)
    img = img * c + b

    if rng.random() < cfg.p_gamma:
        vmin, vmax = float(img.min()), float(img.max())
        if vmax > vmin:
            x = (img - vmin) / (vmax - vmin)
            gamma = _rand_uniform(*cfg.gamma_range, rng)
            x = np.power(x, gamma)
            img = x * (vmax - vmin) + vmin

    if rng.random() < cfg.p_gauss_noise:
        img = img + rng.normal(0.0, cfg.gauss_sigma, size=img.shape).astype(img.dtype)

    return img

# ==================== DATASET ====================

class BrainMetSlices(Dataset):
    """Brain metastases dataset"""
    def __init__(self, rows, binarize_threshold=0.5, zscore=True, target_size=(256,256),
                 augment: bool = False, aug_cfg: AugmentConfig | None = None, seed: int = 1234):
        self.rows = rows
        self.binarize_threshold = binarize_threshold
        self.zscore = zscore
        self.target_size = target_size
        self.augment = augment
        self.aug_cfg = aug_cfg if aug_cfg is not None else AugmentConfig()
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        rec = self.rows[i]
        ds_img = read_dicom(rec["img"])
        ds_msk = read_dicom(rec["mask"])

        img = dicom_to_image(ds_img).astype(np.float32)
        msk = ds_msk.pixel_array.astype(np.float32, copy=False)

        # Binarize mask
        vmax = float(msk.max()) if msk.size > 0 else 1.0
        if vmax > 0:
            msk = msk / vmax
        msk = (msk >= self.binarize_threshold).astype(np.int64)

        # Z-score normalization
        if self.zscore:
            mu = float(img.mean())
            sd = float(img.std())
            img = (img - mu) / sd if sd > 0 else (img - mu)

        # Augmentation
        if self.augment:
            rng = np.random.default_rng((self._seed + i) & 0xFFFFFFFF)
            img, msk = augment_flip_lr(img, msk, rng, self.aug_cfg.p_flip_lr)
            img = augment_intensity(img, rng, self.aug_cfg)

        # To tensors
        img_t = torch.from_numpy(img).to(torch.float32)
        msk_t = torch.from_numpy(msk).to(torch.int64)

        # Resize
        if self.target_size is not None:
            img_t = _resize_2d(img_t, self.target_size, mode="bilinear", align_corners=False)
            msk_t = _resize_2d(msk_t.to(torch.float32), self.target_size, mode="nearest", align_corners=None).to(torch.int64)

        # Channel-first
        img_t = img_t.unsqueeze(0)
        msk_t = msk_t.unsqueeze(0)

        meta = {"patient": rec["patient"], "img_path": str(rec["img"]), "mask_path": str(rec["mask"])}
        return img_t, msk_t, meta

# ==================== U-NET MODEL ====================

class DoubleConv(nn.Module):
    """(Conv -> BN -> ReLU) x 2"""
    def __init__(self, in_ch, out_ch, use_bn=True, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=not use_bn),
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))

        layers += [
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=not use_bn),
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))

        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class Down(nn.Module):
    """Downscale: MaxPool -> DoubleConv"""
    def __init__(self, in_ch, out_ch, use_bn=True, dropout=0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch, use_bn, dropout)

    def forward(self, x):
        return self.conv(self.pool(x))

class Up(nn.Module):
    """Upscale: ConvTranspose2d -> concat skip -> DoubleConv"""
    def __init__(self, in_ch, out_ch, use_bn=True, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch, use_bn, dropout)

    @staticmethod
    def _center_crop(skip, target_spatial):
        _, _, H, W = skip.shape
        h, w = target_spatial
        dh = (H - h) // 2
        dw = (W - w) // 2
        return skip[:, :, dh:dh + h, dw:dw + w]

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            skip = self._center_crop(skip, x.shape[-2:])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    """U-Net for medical image segmentation"""
    def __init__(self, in_channels=1, num_classes=1, base_ch=64, use_bn=True, dropout=0.0):
        super().__init__()
        # Encoder
        self.inc = DoubleConv(in_channels, base_ch, use_bn, dropout=0.0)
        self.down1 = Down(base_ch, base_ch*2, use_bn, dropout=dropout)
        self.down2 = Down(base_ch*2, base_ch*4, use_bn, dropout=dropout)
        self.down3 = Down(base_ch*4, base_ch*8, use_bn, dropout=dropout)
        self.down4 = Down(base_ch*8, base_ch*16, use_bn, dropout=dropout)

        # Decoder
        self.up1 = Up(base_ch*16, base_ch*8, use_bn, dropout=dropout)
        self.up2 = Up(base_ch*8,  base_ch*4, use_bn, dropout=dropout)
        self.up3 = Up(base_ch*4,  base_ch*2, use_bn, dropout=dropout)
        self.up4 = Up(base_ch*2,  base_ch,   use_bn, dropout=dropout)

        self.outc = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)
        logits = self.outc(x)
        return logits

# ==================== LOSS FUNCTION ====================

def dice_coeff(pred, target, eps=1e-6):
    """Dice coefficient for binary segmentation"""
    pred = pred.squeeze(1)
    target = target.squeeze(1).float()
    intersection = (pred * target).sum(dim=(1,2))
    union = pred.sum(dim=(1,2)) + target.sum(dim=(1,2))
    dice = (2*intersection + eps) / (union + eps)
    return dice.mean()

def bce_dice_loss(logits, target):
    """Combined BCE + Dice loss"""
    bce = nn.BCEWithLogitsLoss()
    bce_loss = bce(logits, target.float())
    probs = torch.sigmoid(logits)
    dice = dice_coeff(probs, target)
    return bce_loss + (1 - dice)

# ==================== MAIN TRAINING ====================

def main():
    """Main training function"""

    # Print system info
    print_system_info()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Build dataset
    print("\n" + "="*60)
    print("BUILDING DATASET")
    print("="*60)
    index = build_slice_index(ROOT)
    if not index:
        raise RuntimeError("No slice pairs found. Check folder names and DICOM contents.")

    split = split_by_patient(index, test_fraction=TEST_FRACTION, val_fraction=VAL_FRACTION, seed=RNG_SEED)

    # Create datasets
    train_ds = BrainMetSlices(split["train"], augment=True, aug_cfg=AugmentConfig(), seed=42)
    val_ds   = BrainMetSlices(split["val"],   augment=False)
    test_ds  = BrainMetSlices(split["test"],  augment=False)

    # Create GPU-optimized DataLoaders
    print("\n" + "="*60)
    print("CREATING GPU-OPTIMIZED DATALOADERS")
    print("="*60)
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Num workers: {NUM_WORKERS}")
    print(f"Pin memory: True")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=resize_collate
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=resize_collate
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=resize_collate
    )

    # Create model
    print("\n" + "="*60)
    print("CREATING MODEL")
    print("="*60)
    model = UNet(in_channels=1, num_classes=1, base_ch=32, use_bn=True, dropout=0.1).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Optimizer and scheduler
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(opt, T_max=int(0.75*NUM_EPOCHS))
    criterion = bce_dice_loss

    # Training loop
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)

    # Loss tracking
    train_loss_history = []
    val_loss_history = []

    # Accuracy tracking
    train_acc_history = []
    val_acc_history = []

    
    start_time = time.time()

    plt.ion()
    fig, ax = plt.subplots(figsize = (12,5))

    for epoch in range(NUM_EPOCHS):
        # ----- TRAIN ------
        model.train()
        epoch_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for imgs, msks, _ in train_loader:
            imgs, msks = imgs.to(device), msks.to(device)
            opt.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, msks.float())
            loss.backward()
            opt.step()
            epoch_train_loss += loss.item() * imgs.size(0)
            # For accuracy: 
            preds = (torch.sigmoid(logits).detach().cpu().numpy() > 0.5)
            labels = msks.detach().cpu().numpy()
            correct_train += (preds == labels).sum()
            total_train += labels.size

        avg_train_loss = epoch_train_loss / len(train_loader.dataset)
        train_loss_history.append(avg_train_loss)
        train_acc_history.append(correct_train / total_train)

        # ----- VALIDATION ------
        model.eval()
        epoch_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for imgs, msks, _ in val_loader:
                imgs, msks = imgs.to(device), msks.to(device)
                logits = model(imgs)
                loss = criterion(logits, msks.float())
                epoch_val_loss += loss.item() * imgs.size(0)
                # For accuracy:
                preds = (torch.sigmoid(logits).detach().cpu().numpy() > 0.5)
                labels = msks.detach().cpu().numpy()
                correct_val += (preds == labels).sum()
                total_val += labels.size

        avg_val_loss = epoch_val_loss / len(val_loader.dataset)
        val_loss_history.append(avg_val_loss)
        val_acc_history.append(correct_val / total_val)

        # ---- Optional: print progress every epoch

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f} | "
            f"Train Acc: {train_acc_history[-1]:.3f}, Val Acc: {val_acc_history[-1]:.3f}")
        
        # --- Live PLOT (for notebook/shell, to update after each epoch)
        
        # update plot
        ax.clear()
        ax.plot(train_loss_history, label="Train Loss")
        ax.plot(val_loss_history, label="Val Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training and Validation Loss per Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.pause(0.01)  # let the UI update                    

    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Total training time: {total_time/60:.2f} minutes")
    print(f"Average time per epoch: {total_time/NUM_EPOCHS:.2f} seconds")

    # Save model
    save_path = Path("./claudified/unet_brain_tumor_gpu.pth")
    save_path.parent.mkdir(exist_ok=True)
    torch.save({
        'epoch': NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'loss_history': train_loss_history,
        'lr_history': lr_history,
    }, save_path)
    print(f"\nModel saved to: {save_path}")

# print model parameters. # params directly relates to memory 
# Read the .pth file and visualize what it looks like (what does each image and mask look like?)
# Plot loss and accuracy 
# Make sure to output the weights just in case things go bad 
# model.summary()

'''
    # Plot training curves
    epochs_range = range(1, epochs + 1)

    fig, ax1 = plt.subplots(figsize=(10,6))
    ax1.plot(epochs_range, train_loss_history, color='tab:blue', label='Training Loss')
    ax1.plot(epochs_range, val_loss_history, color='tab:cayan', label='Validatin Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(lr_history, color='tab:red', linestyle='--', label='Learning Rate')
    ax2.set_ylabel('Learning Rate', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.suptitle("Training Loss and Learning Rate vs. Epoch")
    fig.tight_layout()
    
    print(f"Training curves saved to: ./claudified/training_curves.png")

'''
plt.savefig('./claudified/training_curves.png', dpi=150, bbox_inches='tight')    

if __name__ == "__main__":
    main()
