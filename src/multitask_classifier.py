import torch
import torch.nn as nn
import torchvision.models as models

class MultiTaskTileClassifier(nn.Module):
    def __init__(self, num_tile_classes=7, max_crowns=3):
        super().__init__()
        
        # Shared backbone
        self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1').features
        
        # Adaptive pooling
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Task-specific heads
        self.tile_classifier = nn.Linear(1280, num_tile_classes)
        self.crown_counter = nn.Linear(1280, max_crowns + 1)  # 0-3 crowns
        
    def forward(self, x):
        features = self.backbone(x)
        features = self.pool(features)
        features = torch.flatten(features, 1)
        
        tile_class = self.tile_classifier(features)
        crown_count = self.crown_counter(features)
        
        return tile_class, crown_count