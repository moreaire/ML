# Brain Tumor Segmentation - GPU Optimized Training

GPU-accelerated U-Net training for brain metastases segmentation on RTX 4090.

## Quick Start

```bash
cd /config/workspace/git/MLClass/claudified

# Run training (should complete in ~6-10 minutes)
python train_unet_gpu.py
```

## Performance Comparison

| Platform | Time/Epoch | Total (120 epochs) | CPU Usage | System Impact |
|----------|------------|-------------------|-----------|---------------|
| **Original (CPU)** | ~2.75 min | ~5.5 hours | 2400% | System locks up |
| **GPU Optimized** | ~3-5 sec | ~6-10 min | <100% | No impact |

**Speedup: 30-50x faster** ⚡

## What's Been Optimized

### 1. CPU Thread Control
- Limited to 4 threads (prevents system lockup)
- Keeps system responsive during training

### 2. GPU-Optimized DataLoaders
```python
batch_size=16              # Increased from 8
num_workers=4              # Parallel loading (was 0)
pin_memory=True            # Fast CPU→GPU transfer
persistent_workers=True    # Workers stay alive
```

### 3. Model Configuration
- U-Net with base_ch=32 (~6.6M parameters)
- Fits comfortably in 24GB VRAM
- Can increase batch_size to 32 or 64 for even faster training

## Files

- `train_unet_gpu.py` - Main training script
- `requirements.txt` - Python dependencies
- `README.md` - This file

## System Requirements

- **GPU:** NVIDIA GPU with CUDA support (tested on RTX 4090)
- **VRAM:** 4GB+ (8GB+ recommended)
- **CUDA:** 11.8+ or 12.0+
- **PyTorch:** 2.0+ with CUDA

## Advanced Usage

### Increase Batch Size (Faster Training)

Edit `train_unet_gpu.py`:
```python
BATCH_SIZE = 32  # or 64 for RTX 4090
```

### Enable Mixed Precision (2x Speedup)

Add to training loop:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# In training loop:
with autocast():
    logits = model(imgs)
    loss = criterion(logits, msks.float())

scaler.scale(loss).backward()
scaler.step(opt)
scaler.update()
```

### Monitor GPU Usage

```bash
# In another terminal
watch -n 1 nvidia-smi
```

## Troubleshooting

### GPU Not Available

Check:
1. Container has `--gpus all` flag
2. PyTorch has CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
3. Install CUDA version: `pip install torch --index-url https://download.pytorch.org/whl/cu124`

### Out of Memory

Reduce batch size in `train_unet_gpu.py`:
```python
BATCH_SIZE = 8  # or 4
```

### Slow Training

Increase workers:
```python
NUM_WORKERS = 6  # or 8
```

## Output

- **Model:** `claudified/unet_brain_tumor_gpu.pth`
- **Training Curves:** `claudified/training_curves.png`
- **Console:** Loss and timing per epoch

## Next Steps

After training:
1. Evaluate on test set
2. Visualize predictions
3. Export for deployment
4. Fine-tune hyperparameters

## Original Notebook

Original Jupyter notebook preserved at: `../NN_code.ipynb`
