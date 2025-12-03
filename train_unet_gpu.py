#!/usr/bin/env python3
"""
Brain Tumor Segmentation with U-Net - GPU Optimized
====================================================
Training script for brain metastases segmentation using U-Net with RTX 4090 GPU acceleration.

Expected performance:
- Training time: ~20 minutes for 120 epochs (vs 5.5 hours on CPU)
- GPU utilization: ~80-90% during training
- System remains responsive during training

Author: Olivia Magneson and Morgan Aire
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
import torch
from torch.nn.functional import sigmoid
from torch.optim import AdamW
from torch.optim import RAdam




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

# Training Parameters
BATCH_SIZE = 32       # Optimized for RTX 4090 (can increase to 32-64)
NUM_WORKERS = 2    # Parallel data loading
NUM_EPOCHS = 100      # created from total data / batch size
LEARNING_RATE = 1e-3   # Starting learning rate. Commonly used for ADAM
WEIGHT_DECAY = 1e-4    # Common starting weight decay for regularization
PATIENCE = 50       # epochs with no improvement before stopping
MIN_DELTA = 1e-4    # minimum change in val loss to count as improvement 0.0001
BCE_WEIGHT = 0.3 #change loss so that dice score impacts more
DICE_WEIGHT = 0.7
POS_WEIGHT = 30 #make tumor positions weigh more in bce score

# =======================================================

################################################################################################
## GPU Settup ##################################################################################
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

################################################################################################


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
    """
    Patient-wise split to avoid leakage:
      - Pick TEST_FRACTION of patients for test
      - From remaining, pick VAL_FRACTION for val
    Returns dict with 'train', 'val', 'test' lists of index rows.
    """
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
    # batch is a list of (img[1,H,W], mask[1,H,W], meta_dict)
    imgs, msks, metas = zip(*batch)  # metas stays as tuple of dicts
    imgs = torch.stack(imgs, dim=0)  # [B,1,H,W]
    msks = torch.stack(msks, dim=0)  # [B,1,H,W]
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
    """Holds parameters controlling data augmentation strength."""
    p_flip_lr: float = 0.5  # probability of left-right flip (mirror across Y-axis)
    contrast_range: tuple[float, float] = (0.9, 1.1)
    brightness_range: tuple[float, float] = (-0.1, 0.1)
    p_gamma: float = 0.3
    gamma_range: tuple[float, float] = (0.9, 1.1)
    p_gauss_noise: float = 0.3
    gauss_sigma: float = 0.05  # std dev of additive Gaussian noise (z-score units)

def _rand_uniform(a: float, b: float, rng: np.random.Generator) -> float:
    return float(rng.uniform(a, b))

def augment_flip_lr(img: np.ndarray, msk: np.ndarray,
                    rng: np.random.Generator, p: float) -> tuple[np.ndarray, np.ndarray]:
    """Random horizontal (Y-axis) flip applied to both image and mask."""
    if rng.random() < p:
        img = np.ascontiguousarray(img[:, ::-1])
        msk = np.ascontiguousarray(msk[:, ::-1])
    return img, msk

def augment_intensity(img: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig) -> np.ndarray:
    """Apply mild random intensity changes to an already z-scored image."""
    # Contrast (multiply) and brightness (add)
    c = _rand_uniform(*cfg.contrast_range, rng)
    b = _rand_uniform(*cfg.brightness_range, rng)
    img = img * c + b

    # Optional gamma (nonlinear brightness curve)
    if rng.random() < cfg.p_gamma:
        vmin, vmax = float(img.min()), float(img.max())
        if vmax > vmin:
            x = (img - vmin) / (vmax - vmin)
            gamma = _rand_uniform(*cfg.gamma_range, rng)
            x = np.power(x, gamma)
            img = x * (vmax - vmin) + vmin

    # Optional Gaussian noise
    if rng.random() < cfg.p_gauss_noise:
        img = img + rng.normal(0.0, cfg.gauss_sigma, size=img.shape).astype(img.dtype)

    return img

# ==================== DATASET ====================

class BrainMetSlices(Dataset):
    """
    Returns (image, mask, meta) per 2D slice.
    image: torch.float32 [1,H,W], z-scored and resized
    mask : torch.int64  [1,H,W], binary
    """
    def __init__(self, rows, binarize_threshold=0.5, zscore=True, target_size=(256,256),
                 augment: bool = False, aug_cfg: AugmentConfig | None = None, seed: int = 1234):
        self.rows = rows
        self.binarize_threshold = binarize_threshold
        self.zscore = zscore
        self.target_size = target_size
        self.augment = augment
        self.aug_cfg = aug_cfg if aug_cfg is not None else AugmentConfig()
        # per-dataset RNG (so workers get different streams even with same seed)
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

        # Robust mask binarization
        vmax = float(msk.max()) if msk.size > 0 else 1.0
        if vmax > 0:
            msk = msk / vmax
        msk = (msk >= self.binarize_threshold).astype(np.int64)

        # We apply z-score first → jitter operates in a consistent scale.
        if self.zscore:
            mu = float(img.mean())
            sd = float(img.std())
            img = (img - mu) / sd if sd > 0 else (img - mu)

        # --- AUGMENTATIONS (train only) ---
        if self.augment:
            # Make a per-sample RNG by mixing index and base seed (stable across epochs)
            rng = np.random.default_rng((self._seed + i) & 0xFFFFFFFF)
            # Spatial flip (apply to image and mask)
            img, msk = augment_flip_lr(img, msk, rng, self.aug_cfg.p_flip_lr)
            # Intensity jitter (image only)
            img = augment_intensity(img, rng, self.aug_cfg)

        # To tensors
        img_t = torch.from_numpy(img).to(torch.float32)
        msk_t = torch.from_numpy(msk).to(torch.int64)

        # Resize to common size (image bilinear, mask nearest)
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
    def __init__(self, in_ch, out_ch, use_bn=True, dropout=0.0):  #no droppout 
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=not use_bn), # 3x3 kernal with padding = 1 so that maintains the same HxW
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch)) # stabilizes training and prevent exploding gradients
        layers.append(nn.ReLU(inplace=True)) # ReLu acivation to save GPU memory

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
        x = self.up(x)              # [B, C/2, 2H, 2W]
        if x.shape[-2:] != skip.shape[-2:]:
            skip = self._center_crop(skip, x.shape[-2:])
        x = torch.cat([skip, x], dim=1)         # channel concat
        return self.conv(x)

class UNet(nn.Module):
    """
    U-Net (2D).
    - in_channels: image channels (1)
    - num_classes: 1 for binary (logit); >1 for multi-class (logits per class)
    - base_ch: width of first stage
    """
    def __init__(self, in_channels=1, num_classes=1, base_ch=32, use_bn=True, dropout=0.0):
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
        x1 = self.inc(x)     # [B, 64, H, W]
        x2 = self.down1(x1)  # [B, 128, H/2, W/2]
        x3 = self.down2(x2)  # [B, 256, H/4, W/4]
        x4 = self.down3(x3)  # [B, 512, H/8, W/8]
        x5 = self.down4(x4)  # [B, 1024,H/16,W/16]
        # Decoder with skips
        x = self.up1(x5, x4)
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)
        logits = self.outc(x)
        return logits

# ==================== Model Test Function ====================
@torch.no_grad()
def evaluate_loader(model, loader, device, criterion, desc: str = "EVAL"):
    """
    Run one full pass over a DataLoader and compute:
      - avg loss
      - pixel accuracy
      - mean Dice (per batch)
    Uses the same logic as your train/val loops.
    """
    model.eval()
    epoch_loss = 0.0
    correct = 0
    total = 0
    epoch_dice = []

    for imgs, msks, _ in loader:
        imgs, msks = imgs.to(device), msks.to(device)

        logits = model(imgs)
        loss = criterion(logits, msks.float())
        epoch_loss += loss.item() * imgs.size(0)

        # Pixel accuracy (same as in your train/val)
        preds = (torch.sigmoid(logits).detach().cpu().numpy() > 0.5)
        labels = msks.detach().cpu().numpy()
        correct += (preds == labels).sum()
        total += labels.size

        # Dice per batch
        probs = torch.sigmoid(logits)
        epoch_dice.append(dice_coeff(probs, msks).item())

    avg_loss = epoch_loss / len(loader.dataset)
    avg_acc = correct / total if total > 0 else 0.0
    avg_dice = float(np.mean(epoch_dice)) if epoch_dice else 0.0

    print(f"\n[{desc}]  Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f} | Dice: {avg_dice:.4f}")

    return {
        "loss": avg_loss,
        "accuracy": avg_acc,
        "dice": avg_dice,
    }

# ==================== LOSS FUNCTION ====================
'''
def dice_coeff(pred, target, eps=1e-6):
    """Dice coefficient for binary segmentation"""
    pred = pred.squeeze(1)
    target = target.squeeze(1).float()
    intersection = (pred * target).sum(dim=(1,2))
    union = pred.sum(dim=(1,2)) + target.sum(dim=(1,2))
    dice = (2*intersection + eps) / (union + eps)
    return dice.mean()
'''
def dice_coeff(pred_probs, target, thresh=0.5, eps=1e-6):
    """
    Dice for binary segmentation.
    - pred_probs: [B,1,H,W] probabilities
    - target:     [B,1,H,W] {0,1}
    Only computed on slices that contain tumor in the ground truth.
    """
    pred = (pred_probs >= thresh).float()   # [B,1,H,W]
    target = target.float()
    pred = pred.squeeze(1)
    target = target.squeeze(1)
    fg_mask = (target.view(target.size(0), -1).sum(dim=1) > 0)  # [B] bool

    if not fg_mask.any():
        # No tumor slices in this batch
        return torch.tensor(0.0, device=pred_probs.device)

    pred_fg = pred[fg_mask]
    target_fg = target[fg_mask]

    intersection = (pred_fg * target_fg).sum(dim=(1, 2))
    union = pred_fg.sum(dim=(1, 2)) + target_fg.sum(dim=(1, 2))
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean()

def bce_dice_loss(logits, target):
    """Combined BCE + Dice loss"""
    pos_weight = logits.new_tensor(POS_WEIGHT)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    bce_loss = bce(logits, target.float())
    probs = torch.sigmoid(logits)
    dice = dice_coeff(probs, target)
    loss = BCE_WEIGHT*bce_loss + DICE_WEIGHT*(1 - dice)
    return loss


''' Visualization of Generated Masks '''
@torch.no_grad()
def visualize_random_predictions(
    model,
    dataset,
    device,
    num_examples: int = 6,
    prob_thresh: float = 0.5,
    seed: int | None = None,
    save_path: str | None = None,
):
    """
    Visualize a random mix of tumor and non-tumor slices.
    Displays MR image, ground-truth mask, and predicted mask.
    """
    model.eval()
    # Optional: true randomness if seed=None
    if seed is not None:
        random.seed(seed)
    # Choose random indices from entire dataset
    all_indices = list(range(len(dataset)))
    random.shuffle(all_indices)
    # Pick the first N after shuffling
    selected = all_indices[:num_examples]
    fig, axes = plt.subplots(num_examples, 3, figsize=(10, 3 * num_examples))
    if num_examples == 1:
        axes = np.expand_dims(axes, axis=0)
    for row, idx in enumerate(selected):
        img_t, msk_t, meta = dataset[idx]

        # Move to GPU/CPU for inference
        logits = model(img_t.unsqueeze(0).to(device))   # [1,1,H,W]
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
        pred_mask = (probs >= prob_thresh).astype(np.uint8)

        true_mask = msk_t[0].numpy()
        image_np = img_t[0].numpy()

        ax_img, ax_true, ax_pred = axes[row]

        # (1) Original MRI
        ax_img.imshow(image_np, cmap="gray")
        ax_img.set_title(f"Image\n{meta['patient']}")
        ax_img.axis("off")

        # (2) Ground-truth mask overlay
        ax_true.imshow(image_np, cmap="gray")
        ax_true.imshow(true_mask, alpha=0.4)
        ax_true.set_title("Ground Truth Mask")
        ax_true.axis("off")

        # (3) Predicted mask overlay
        ax_pred.imshow(image_np, cmap="gray")
        ax_pred.imshow(pred_mask, alpha=0.4)
        ax_pred.set_title("Predicted Mask")
        ax_pred.axis("off")

    plt.tight_layout()

    # Optional saving
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    #plt.show()
    return fig


    """ Validation Metrics """
@torch.no_grad()
def eval_tumor_presence(model, loader, device, prob_thresh=0.5, min_positive_pixels=1):
    """
    Accuracy, Specificity, Sensitivity
    TP FP TN FN
    """
    model.eval()
    tp = tn = fp = fn = 0

    for imgs, msks, _ in loader:
        imgs = imgs.to(device)
        msks = msks.to(device)  # [B,1,H,W], int64 0/1

        logits = model(imgs)
        probs  = torch.sigmoid(logits)     # [B,1,H,W]
        preds  = (probs >= prob_thresh).to(torch.int64)

        # Flatten per-slice
        preds_flat  = preds.view(preds.size(0), -1)
        masks_flat  = msks.view(msks.size(0), -1)

        # Per-slice "has tumor" flags
        pred_has_tumor = (preds_flat.sum(dim=1) >= min_positive_pixels)  # bool
        true_has_tumor = (masks_flat.sum(dim=1)  > 0)                    # bool

        for p, t in zip(pred_has_tumor, true_has_tumor):
            if t and p:
                tp += 1
            elif (not t) and (not p):
                tn += 1
            elif (not t) and p:
                fp += 1
            elif t and (not p):
                fn += 1

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else 0.0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "accuracy": acc,
        "sensitivity": sens,
        "specificity": spec,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "total_slices": total,
    }
    return metrics
# ==================== MAIN TRAINING ====================

def main():
    """Main training function"""

    # SYSTEM AND DATA SET UP # --------------------
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

    # Make sure that the test, validation, and training are split early by patient to avoid bleed through 
    split = split_by_patient(index, test_fraction=TEST_FRACTION, val_fraction=VAL_FRACTION, seed=RNG_SEED)

    # Create datasets
    train_ds = BrainMetSlices(split["train"], augment=True, aug_cfg=AugmentConfig(), seed=42) #With data augmentation
    val_ds   = BrainMetSlices(split["val"],   augment=False) # No augmentation
    test_ds  = BrainMetSlices(split["test"],  augment=False) # No augmentation

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


    # MODEL / OPTIM # ---------------------------------------------------
    # Create model
    print("\n" + "="*60)
    print("CREATING MODEL")
    print("="*60)
    model = UNet(in_channels=1, num_classes=1, base_ch=64, use_bn=True, dropout=0.1).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Optimizer and scheduler
    opt = RAdam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(opt, T_max=int(NUM_EPOCHS)) #decrease learning rate via cos not step (smoother)
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

    # Dice tracking
    train_dice_history = []
    val_dice_history = []

    lr_history = []

    ### Early Stopping Initializations ###
    best_val_loss = float('inf')
    best_epoch = -1
    epochs_no_improve = 0
    #best_model_path = Path("./claudified/unet_brain_tumor_best.pth")
    best_model_path = Path("./unet_brain_tumor_earlystopped.pth")
    best_model_path.parent.mkdir(exist_ok=True)

    start_time = time.time()

    plt.ion()
    fig, ax = plt.subplots(figsize = (12,5))
    plt.show(block = False)

    for epoch in range(NUM_EPOCHS):
        # ----- TRAIN ------
        model.train()
        epoch_train_loss = 0.0
        correct_train = 0
        total_train = 0
        epoch_train_dice = [] #initialize so that can find average per epoch

        for imgs, msks, _ in train_loader:
            imgs, msks = imgs.to(device), msks.to(device)
            opt.zero_grad() #zero gradiants from previoius batch

            logits = model(imgs)
            loss = criterion(logits, msks.float())
            loss.backward()
            opt.step() #update model parameters

            epoch_train_loss += loss.item() * imgs.size(0)

            # For accuracy: 
            preds = (torch.sigmoid(logits).detach().cpu().numpy() > 0.5) #create preditiction mask with 0.5 threshold
            labels = msks.detach().cpu().numpy()
            correct_train += (preds == labels).sum() #correct number of pixels in prediction mask to ground truth
            total_train += labels.size

            #dice per batch
            probs = torch.sigmoid(logits)
            epoch_train_dice.append(dice_coeff(probs, msks).item()) # record train dice scores

        #Update training loss, accuracy, dice score  and add to ongoing list for plotting
        avg_train_loss = epoch_train_loss / len(train_loader.dataset)
        avg_train_acc = correct_train / total_train
        avg_train_dice = float(np.mean(epoch_train_dice)) if epoch_train_dice else 0.0

        train_loss_history.append(avg_train_loss)
        train_acc_history.append(avg_train_acc)
        train_dice_history.append(avg_train_dice)
        
        # ----- VALIDATION ------
        model.eval()
        epoch_val_loss = 0.0
        correct_val = 0
        total_val = 0
        epoch_val_dice = []


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

                 # Dice per batch
                probs = torch.sigmoid(logits)
                epoch_val_dice.append(dice_coeff(probs, msks).item())

        # Calculate val loss, accuracy, dice sore and add to the primary array of histories
        avg_val_loss = epoch_val_loss / len(val_loader.dataset)
        avg_val_acc = correct_val / total_val
        avg_val_dice = float(np.mean(epoch_val_dice)) if epoch_val_dice else 0.0

        val_loss_history.append(avg_val_loss)
        val_acc_history.append(avg_val_acc)
        val_dice_history.append(avg_val_dice)

        scheduler.step() # update learning rate for plotting
        lr_history.append(scheduler.get_last_lr()[0])

        # ---- Optional: print progress every epoch

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f} | "
            f"Train Acc: {avg_train_acc:.3f}, Val Acc: {avg_val_acc:.3f} | "
            f"Train Dice: {avg_train_dice:.3f}, Val Dice: {avg_val_dice:.3f}"
        )
        
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
        
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)  # let the UI update                    
        
        ###Early Stopping ###
        if avg_val_loss < best_val_loss - MIN_DELTA: #Min delta is amount allowed to deviate 
            # Improvement
            best_val_loss = avg_val_loss
            best_epoch = epoch
            epochs_no_improve = 0

            # Save best model so far; keep for final save
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'train_loss_history': train_loss_history,
                'val_loss_history': val_loss_history,
                'train_acc_history': train_acc_history,
                'val_acc_history': val_acc_history,
                'train_dice_history': train_dice_history,
                'val_dice_history': val_dice_history,
                'lr_history': lr_history,
                'best_val_loss': best_val_loss,
            }, best_model_path)
            print(f"✓ New best model saved at epoch {epoch+1} with val_loss={best_val_loss:.4f}")

        else:
            # No sufficient improvement
            epochs_no_improve += 1
            print(f"No improvement in val loss for {epochs_no_improve} epoch(s)")
            if epochs_no_improve >= PATIENCE:
                print(f"⏹ Early stopping triggered at epoch {epoch+1}. "
                      f"Best epoch was {best_epoch+1} with val_loss={best_val_loss:.4f}")

                break

    '''            
    globals().update({
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
        "train_acc_history": train_acc_history,
        "val_acc_history": val_acc_history,
        "train_dice_history": train_dice_history,
        "val_dice_history": val_dice_history,
        "lr_history": lr_history,
        "model": model,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "device": device,
        "val_ds": val_ds
    })
    '''

    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Total training time: {total_time/60:.2f} minutes")

    '''
    #Save model
    save_path = Path("./unet_brain_tumor_notstopped.pth")
    save_path.parent.mkdir(exist_ok=True)
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'loss_history': train_loss_history,
        'val_loss_history': val_loss_history,
        'train_acc_history': train_acc_history,
        'val_acc_history': val_acc_history,
        'train_dice_history': train_dice_history,
        'val_dice_history': val_dice_history,
        'lr_history': lr_history,
    }, save_path)
    print(f"\nModel saved to: {save_path}")
    '''
    # ----- NEW: RELOAD BEST AND TEST  -------------------------
    # Load best early-stopped model before test evaluation (optional)
    print("\nReloading best model (early-stopped weights)...")
    ckpt = torch.load(
        best_model_path,
        map_location= device,
        weights_only = False,
    )
    model.load_state_dict(ckpt["model_state_dict"])

    print("\n" + "="*60)
    print("TEST SET EVALUATION (BEST MODEL)")
    print("="*60)
    test_metrics = evaluate_loader(model, test_loader, device, criterion, desc="TEST (BEST)")
    test_loss = test_metrics["loss"]
    test_acc = test_metrics["acc"]
    test_dice = test_metrics["dice"]


    # ----- VALIDATION TUMOR PRESENCE ----------
    # ------------------ VALIDATION TUMOR PRESENCE ------------------
    val_metrics = eval_tumor_presence(model, val_loader, device,
                                      prob_thresh=0.5, min_positive_pixels=1)
    print("\nValidation metrics (tumor presence):")
    for k, v in val_metrics.items():
        print(f"{k}: {v}")

    # ------------------ VISUALIZATION EXAMPLES ------------------
    print("\nVisualizing random validation examples...")
    visualize_random_predictions(model, val_ds, device,
                                 num_examples=6, prob_thresh=0.5, seed=123)

    # ------------------ FINAL PLOTS ------------------
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
    fig.legend(loc="best")
    fig.tight_layout()
    plt.show()

    # 2) Loss + Accuracy vs Epoch (with test lines)
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(val_loss_history,   color="tab:blue",   label="Val Loss")
    ax1.plot(train_loss_history, color="tab:purple", label="Train Loss")
    ax1.axhline(test_loss, color="tab:green", linestyle="--",
                label=f"Test Loss = {test_loss:.3f}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(val_acc_history,   color="tab:red",  linestyle="--", label="Val Accuracy")
    ax2.plot(train_acc_history, color="tab:pink", linestyle="--", label="Train Accuracy")
    ax2.axhline(test_acc, color="tab:orange", linestyle=":",
                label=f"Test Acc = {test_acc:.3f}")
    ax2.set_ylabel("Accuracy", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    fig.suptitle("Loss and Accuracy vs. Epoch")
    fig.legend(loc="best")
    fig.tight_layout()
    plt.show()

    # 3) Dice vs Epoch (with test line)
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(val_dice_history,   color="tab:blue", label="Val Dice")
    ax1.plot(train_dice_history, color="tab:red",  label="Train Dice")
    ax1.axhline(test_dice, color="tab:green", linestyle="--",
                label=f"Test Dice = {test_dice:.3f}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Dice", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    fig.suptitle("Dice vs. Epoch")
    fig.legend(loc="best")
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


'''
################## TESTING ##################################
 # Load best early-stopped model before test evaluation (optional)
print("\nReloading best model (early-stopped weights)...")
ckpt = torch.load(
    best_model_path,
    map_location= device,
)
model.load_state_dict(ckpt["model_state_dict"])

print("\n" + "="*60)
print("TEST SET EVALUATION (BEST MODEL)")
print("="*60)
test_metrics = evaluate_loader(model, test_loader, device, criterion, desc="TEST (BEST)")
test_loss = test_metrics["loss"]
test_acc = test_metrics["acc"]
test_dice = test_metrics["dice"]



####################################################################################
# MODEL ANALYSIS ###################################################################
####################################################################################

""" Plot LR and Training Loss vs # EPOCHS """
fig, ax1 = plt.subplots(figsize=(7,4))
ax1.plot(val_loss_history, color='tab:blue', label='Val Loss')
ax1.plot(train_loss_history, color = 'tab:purple', label = 'Train Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.plot(lr_history, color='tab:red', linestyle='--', label='Learning Rate')
ax2.set_ylabel('Learning Rate', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

fig.suptitle("Training Loss and Learning Rate vs. Epoch")
fig.legend()
fig.tight_layout()
plt.show()






device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
val_metrics = eval_tumor_presence(model, val_loader, device, prob_thresh=0.5, min_positive_pixels=1)
print("Validation metrics (tumor presence):")
for k, v in val_metrics.items():
    print(f"{k}: {v}")




visualize_random_predictions(model, val_ds, device, num_examples=6, prob_thresh=0.5)



Loss and accuracy vs accuracy
fig, ax1 = plt.subplots(figsize=(7,4))
ax1.plot(val_loss_history, color='tab:blue', label='Val Loss')
ax1.plot(train_loss_history, color = 'tab:purple', label = 'Train Loss')

# test loss horizontal line
ax1.axhline(test_loss, color='tab:green', linestyle='--', label=f'Test Loss = {test_loss:.3f}')


ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.plot(val_acc_history, color='tab:red', linestyle='--', label='Val Accuracy')
ax2.plot(train_acc_history, color='tab:pink', linestyle='--', label='Train Accuracy')

# test accuracy horizontal line
ax2.axhline(test_acc, color='tab:orange', linestyle=':', label=f'Test Acc = {test_acc:.3f}')


ax2.set_ylabel('Accuracy', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

fig.suptitle(" Loss and Accuracy vs. Epoch")
fig.legend()
fig.tight_layout()
plt.show()
'''



''' Dice vs Epoch 
fig, ax1 = plt.subplots(figsize=(7,4))
ax1.plot(val_dice_history, color='tab:blue', label='Val Dice')
ax1.plot(train_dice_history, color = 'tab:red', label = 'Train Dice')

# test dice horizontal line
ax1.axhline(test_dice, color='tab:green', linestyle='--', label=f'Test Dice = {test_dice:.3f}')


ax1.set_xlabel('Epoch')
ax1.set_ylabel('Dice', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
fig.suptitle(" Dice vs. Epoch")
fig.legend(loc='best')
fig.tight_layout()
plt.show()
'''


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
#plt.savefig('./claudified/training_curves.png', dpi=150, bbox_inches='tight')    
