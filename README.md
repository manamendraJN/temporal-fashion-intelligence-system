# Body Measurement AI System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent AI-powered system that estimates body measurements from images, provides health insights, and recommends clothing sizes without requiring a clothing database.

## 🎯 Research Contributions

1. **Body Measurement AI**: Accurate dual-input CNN for measurement estimation
2. **Universal Size Recommendation**: Works without clothing database
3. **Health Insights**: WHO-standard validated wellness indicators

## 📊 Project Status

- [x] Project initialization
- [ ] Model training & comparison (3 architectures)
- [ ] Model selection & evaluation
- [ ] Backend API development
- [ ] Frontend development
- [ ] Deployment

## 🏗️ Architecture

### Models Under Evaluation
- **Model 1**: EfficientNet-B3 (Dual-input)
- **Model 2**: ResNet50 (Dual-input)
- **Model 3**: MobileNetV3 (Dual-input, lightweight)

### System Components
- Deep Learning: PyTorch, timm, albumentations
- Backend: Flask REST API
- Frontend: React.js
- Training: Google Colab (GPU)

## 📂 Project Structure
