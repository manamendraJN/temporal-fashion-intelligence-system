"""
Model Loading and Inference
Author: manamendraJN
Date: 2025-11-18
"""

import torch
import torch.nn as nn
import timm
import numpy as np
from pathlib import Path
import json
from config import Config
from utils import preprocess_image

class DualInputBodyModel(nn.Module):
    """Dual-input CNN model for body measurement prediction"""
    
    def __init__(self, backbone_name='efficientnet_b3', num_measurements=14, pretrained=False):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_measurements = num_measurements
        
        # Two encoders for front and side views
        self.front_encoder = timm.create_model(
            backbone_name, 
            pretrained=pretrained, 
            num_classes=0, 
            global_pool='avg'
        )
        self.side_encoder = timm.create_model(
            backbone_name, 
            pretrained=pretrained, 
            num_classes=0, 
            global_pool='avg'
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, 512, 384)
            feature_dim = self.front_encoder(dummy).shape[1]
        
        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_measurements)
        )
    
    def forward(self, front_img, side_img):
        front_features = self.front_encoder(front_img)
        side_features = self.side_encoder(side_img)
        combined = torch.cat([front_features, side_features], dim=1)
        return self.regression_head(combined)

class ModelInference:
    """Handle model loading and inference"""
    
    def __init__(self, model_name='efficientnet-b3', device='cpu'):
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_config = Config.MODELS.get(model_name)
        
        if not self.model_config:
            raise ValueError(f"Model {model_name} not found in config")
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Load normalization stats
        self.load_normalization_stats()
        
        print(f"✅ Model loaded: {self.model_config['name']} on {self.device}")
    
    def _load_model(self):
        """Load trained model from checkpoint"""
        model_path = self.model_config['path']
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model
        model = DualInputBodyModel(
            backbone_name=self.model_config['backbone'],
            num_measurements=len(Config.MEASUREMENT_COLUMNS),
            pretrained=False
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        return model
    
    def load_normalization_stats(self):
        """Load mean and std for denormalization"""
        # Try to load from model checkpoint first
        model_path = self.model_config['path']
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'target_mean' in checkpoint and 'target_std' in checkpoint:
            self.target_mean = torch.FloatTensor(checkpoint['target_mean']).to(self.device)
            self.target_std = torch.FloatTensor(checkpoint['target_std']).to(self.device)
        else:
            # Fallback: try to load from normalization_stats.json
            stats_path = Config.MODEL_DIR / 'normalization_stats.json'
            if stats_path.exists():
                with open(stats_path, 'r') as f:
                    stats = json.load(f)
                self.target_mean = torch.FloatTensor(stats['target_mean']).to(self.device)
                self.target_std = torch.FloatTensor(stats['target_std']).to(self.device)
            else:
                # Default values (not recommended, should have proper stats)
                print("⚠️ Warning: Using default normalization (may affect accuracy)")
                self.target_mean = torch.zeros(len(Config.MEASUREMENT_COLUMNS)).to(self.device)
                self.target_std = torch.ones(len(Config.MEASUREMENT_COLUMNS)).to(self.device)
    
    def denormalize(self, normalized_tensor):
        """Convert normalized predictions back to real values"""
        return normalized_tensor * self.target_std + self.target_mean
    
    def predict(self, front_image_bytes, side_image_bytes):
        """
        Predict body measurements from front and side images
        
        Args:
            front_image_bytes: Front view image bytes
            side_image_bytes: Side view image bytes
        
        Returns:
            Dictionary of measurements
        """
        # Preprocess images
        front_img = preprocess_image(front_image_bytes, Config.IMG_SIZE)
        side_img = preprocess_image(side_image_bytes, Config.IMG_SIZE)
        
        # Add batch dimension
        front_img = front_img.unsqueeze(0).to(self.device)
        side_img = side_img.unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            normalized_output = self.model(front_img, side_img)
            output = self.denormalize(normalized_output)
        
        # Convert to dictionary
        measurements = {}
        output_np = output.cpu().numpy()[0]
        
        for i, col in enumerate(Config.MEASUREMENT_COLUMNS):
            measurements[col] = float(output_np[i])
        
        return measurements
    
    def get_model_info(self):
        """Get model information"""
        return {
            'name': self.model_config['name'],
            'backbone': self.model_config['backbone'],
            'speed': self.model_config['speed'],
            'accuracy': self.model_config['accuracy'],
            'device': str(self.device),
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'measurements': Config.MEASUREMENT_COLUMNS
        }