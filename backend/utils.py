"""
Utility functions for Body Measurement API
Author: manamendraJN
Date: 2025-11-18
"""

import cv2
import numpy as np
import torch
from PIL import Image
import io
import base64
from pathlib import Path
from config import Config

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def preprocess_image(image_bytes, target_size=(512, 384)):
    """
    Preprocess image for model inference
    
    Args:
        image_bytes: Raw image bytes
        target_size: (height, width) tuple
    
    Returns:
        Preprocessed image tensor
    """
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError("Invalid image data")
    
    # Convert grayscale to RGB
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # Resize
    img = cv2.resize(img, (target_size[1], target_size[0]))
    
    # Normalize
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    
    # Convert to tensor (C, H, W)
    img = torch.from_numpy(img.transpose(2, 0, 1)).float()
    
    return img

def decode_base64_image(base64_string):
    """Decode base64 image string to bytes"""
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    return base64.b64decode(base64_string)

def format_measurements(measurements_dict):
    """Format measurements with proper units"""
    formatted = {}
    for key, value in measurements_dict.items():
        formatted[key] = {
            'value': round(value, 2),
            'unit': 'cm',
            'display': f"{round(value, 1)} cm"
        }
    return formatted

def validate_measurements(measurements):
    """Validate measurement values are reasonable"""
    ranges = {
        'ankle': (15, 35),
        'arm-length': (40, 70),
        'bicep': (20, 50),
        'calf': (25, 55),
        'chest': (70, 140),
        'forearm': (15, 40),
        'height': (140, 210),
        'hip': (70, 140),
        'leg-length': (60, 110),
        'shoulder-breadth': (30, 60),
        'shoulder-to-crotch': (40, 90),
        'thigh': (35, 80),
        'waist': (50, 140),
        'wrist': (12, 25)
    }
    
    warnings = []
    for key, value in measurements.items():
        if key in ranges:
            min_val, max_val = ranges[key]
            if value < min_val or value > max_val:
                warnings.append(f"{key}: {value:.1f}cm seems unusual (normal range: {min_val}-{max_val}cm)")
    
    return warnings

def create_error_response(message, status_code=400):
    """Create standardized error response"""
    return {
        'success': False,
        'error': message,
        'status_code': status_code
    }, status_code

def create_success_response(data, message="Success"):
    """Create standardized success response"""
    return {
        'success': True,
        'message': message,
        'data': data,
        'timestamp': str(np.datetime64('now'))
    }, 200