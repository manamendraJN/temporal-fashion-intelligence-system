"""
Create dummy model for API testing (predictions won't be accurate!)
"""
import torch
import torch.nn as nn
from pathlib import Path
import json

# Model architecture (from your training)
class DualInputBodyModel(nn.Module):
    def __init__(self, num_measurements=14):
        super().__init__()
        import timm
        self.front_encoder = timm.create_model('efficientnet_b3', pretrained=False, num_classes=0, global_pool='avg')
        self.side_encoder = timm.create_model('efficientnet_b3', pretrained=False, num_classes=0, global_pool='avg')
        
        with torch.no_grad():
            dummy = torch.randn(1, 3, 512, 384)
            feature_dim = self.front_encoder(dummy).shape[1]
        
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

# Create models directory
models_dir = Path('models')
models_dir.mkdir(exist_ok=True)

# Measurement columns
MEASUREMENT_COLUMNS = [
    'ankle', 'arm-length', 'bicep', 'calf', 'chest', 'forearm',
    'height', 'hip', 'leg-length', 'shoulder-breadth',
    'shoulder-to-crotch', 'thigh', 'waist', 'wrist'
]

# Create dummy normalization stats (approximate averages)
normalization_stats = {
    'target_mean': [23.0, 50.0, 32.0, 38.0, 100.0, 28.0, 170.0, 100.0, 90.0, 45.0, 70.0, 55.0, 85.0, 17.0],
    'target_std': [3.0, 5.0, 4.0, 4.0, 10.0, 3.0, 10.0, 10.0, 8.0, 5.0, 8.0, 6.0, 10.0, 2.0],
    'measurement_columns': MEASUREMENT_COLUMNS
}

# Save normalization stats
with open(models_dir / 'normalization_stats.json', 'w') as f:
    json.dump(normalization_stats, f, indent=2)
print("✅ Created: normalization_stats.json")

# Create and save dummy models
configs = {
    'efficientnetb3.pth': 'efficientnet_b3',
    'resnet50.pth': 'resnet50',
    'mobilenetv3.pth': 'mobilenetv3_large_100'
}

for filename, backbone in configs.items():
    model = DualInputBodyModel(num_measurements=len(MEASUREMENT_COLUMNS))
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_name': filename.replace('.pth', '').replace('_', '-'),
        'backbone': backbone,
        'measurement_columns': MEASUREMENT_COLUMNS,
        'target_mean': normalization_stats['target_mean'],
        'target_std': normalization_stats['target_std'],
        'note': 'DUMMY MODEL - For API testing only, predictions are random!'
    }
    
    torch.save(checkpoint, models_dir / filename)
    print(f"✅ Created: {filename}")

print("\n⚠️ WARNING: These are DUMMY models with random weights!")
print("   Predictions will NOT be accurate.")
print("   Replace with your trained models for real predictions.")