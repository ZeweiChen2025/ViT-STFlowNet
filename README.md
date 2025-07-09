# ViT-STFlowNet: Geometry-Adaptive Spatiotemporal Flow Prediction Framework

![Framework Overview](doc/ViT_STFlowNet.png)

## 📌 Core Innovation
**ViT-STFlowNet** is a novel Vision Transformer-based architecture for predicting **non-periodic unsteady flows in deformable domains**, featuring:
- **Geometry-adaptive attention**: Dynamically adjusts to domain deformations via SDF-enhanced positional encoding
- **Spatiotemporal coherence**: Hybrid CNN-Transformer structure captures multiscale flow features
- **Physics-aware training**: Combined RMSE/Gradient/SSIM loss enforces physical consistency

## 🚀 Quick Start
### Prerequisites
```bash
conda create -n vitst python=3.9
conda install pytorch==2.0.1 torchvision==0.15.2 -c pytorch
pip install einops torchinfo matplotlib
```

## 🚀 Testing (Example)
Test mach number can be 0.31, 0.4, 0.54, 0.63
	When downloading the repository as a compressed package, the model file (.pth) may be corrupted during compression and thus cannot be used directly. To ensure proper execution of the test.py script, we recommend downloading the model file method2_2loss_minloss.pth separately and replacing the corresponding model file in the compressed folder with this standalone version. This will allow the test script to run as intended.
```bash
python test.py \
  --test_ma 0.31
python contour.py \
  --test_ma 0.31
```
