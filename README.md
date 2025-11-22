# 🎯 Body Measurement AI System

AI-powered body measurement from silhouette images using deep learning.

## Features

- ✅ Automatic background removal
- ✅ Body measurement prediction (14+ measurements)
- ✅ Health insights (BMI, body proportions)
- ✅ Clothing size recommendations
- ✅ Beautiful web interface

## Tech Stack

- **Backend:** Flask, PyTorch, OpenCV, rembg
- **Frontend:** React, Vite, TailwindCSS
- **AI:** EfficientNet-B3, U2NET

## Quick Start

### Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

## API Endpoints

- `GET /health` - Health check
- `POST /predict` - Get measurements
- `POST /complete-analysis` - Full analysis
- `POST /preview-mask` - Preview mask

## Author

**manamendraJN**
Date: November 2025

## License

MIT License