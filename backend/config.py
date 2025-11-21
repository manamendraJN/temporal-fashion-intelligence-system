"""
Configuration file for Body Measurement API
Author: manamendraJN
Date: 2025-11-18
"""

import os
from pathlib import Path

class Config:
    # API Configuration
    API_TITLE = "Body Measurement AI API"
    API_VERSION = "1.0.0"
    DEBUG = os.getenv('DEBUG', 'True') == 'True'
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))
    
    # Model Configuration
    BASE_DIR = Path(__file__).parent
    MODEL_DIR = BASE_DIR / 'models'
    
    # Available models
    MODELS = {
        'efficientnet-b3': {
            'path': MODEL_DIR / 'efficientnet-b3_model.pth',
            'backbone': 'efficientnet_b3',
            'name': 'EfficientNet-B3',
            'speed': 'medium',
            'accuracy': 'high'
        },
        'resnet50': {
            'path': MODEL_DIR / 'resnet50_model.pth',
            'backbone': 'resnet50',
            'name': 'ResNet50',
            'speed': 'slow',
            'accuracy': 'medium'
        },
        'mobilenet': {
            'path': MODEL_DIR / 'mobilenetv3_model.pth',
            'backbone': 'mobilenetv3_large_100',
            'name': 'MobileNetV3',
            'speed': 'fast',
            'accuracy': 'medium'
        }
    }
    
    DEFAULT_MODEL = 'efficientnet-b3'
    
    # Image Configuration
    IMG_SIZE = (512, 384)  # height, width
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    
    # Measurement Configuration
    MEASUREMENT_COLUMNS = [
        'ankle', 'arm-length', 'bicep', 'calf', 'chest', 'forearm',
        'height', 'hip', 'leg-length', 'shoulder-breadth',
        'shoulder-to-crotch', 'thigh', 'waist', 'wrist'
    ]
    
    # Health Configuration
    BMI_CATEGORIES = {
        'underweight': (0, 18.5),
        'normal': (18.5, 25),
        'overweight': (25, 30),
        'obese': (30, 100)
    }
    
    # Size Recommendation Configuration
    SIZE_CHARTS = {
        'us': {
            'male': {
                'xs': {'chest': (81, 86), 'waist': (66, 71)},
                's': {'chest': (86, 91), 'waist': (71, 76)},
                'm': {'chest': (91, 97), 'waist': (76, 81)},
                'l': {'chest': (97, 102), 'waist': (81, 86)},
                'xl': {'chest': (102, 109), 'waist': (86, 94)},
                'xxl': {'chest': (109, 117), 'waist': (94, 102)}
            },
            'female': {
                'xs': {'chest': (76, 81), 'waist': (58, 63), 'hip': (84, 89)},
                's': {'chest': (81, 86), 'waist': (63, 68), 'hip': (89, 94)},
                'm': {'chest': (86, 91), 'waist': (68, 73), 'hip': (94, 99)},
                'l': {'chest': (91, 97), 'waist': (73, 79), 'hip': (99, 104)},
                'xl': {'chest': (97, 102), 'waist': (79, 84), 'hip': (104, 109)},
                'xxl': {'chest': (102, 109), 'waist': (84, 91), 'hip': (109, 117)}
            }
        }
    }
    
    # CORS Configuration
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*')