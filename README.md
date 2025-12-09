# CIFAR10 ConvNet — Training + Inference API

A complete PyTorch pipeline for CIFAR-10 image classification, featuring:

- Convolutional Neural Network (ConvNet)
- Data augmentation (flip, rotation, crop, color jitter)
- Checkpointing with resume training support
- Best-model saving
- FastAPI inference endpoint
- Confusion matrix visualization
- Clean and modular project structure

---

## Features
- **Training** with data augmentation and evaluation  
- **Automatic checkpoint resume** if a previous run exists  
- **Inference API** using FastAPI (`POST /predict`)  
- **Visualization** via confusion matrix (matplotlib + sklearn)

---

## Installation
```bash
pip install -r requirements.txt
