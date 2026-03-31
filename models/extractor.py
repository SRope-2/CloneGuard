import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class CloneExtractor(nn.Module):

    def __init__(self, msg_length=64, input_channels=3):
        super().__init__()
        
        # === 1. Shared Feature Extractor (Simple ResNet-like) ===
        # Input: [B, 3, 256, 256]
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 64, 7, 2, 3, bias=False), # -> 128x128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1) # -> 64x64
        )
        
        # Feature Blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2) # -> 32x32
        self.layer3 = self._make_layer(128, 256, 2, stride=2) # -> 16x16
        self.layer4 = self._make_layer(256, 512, 2, stride=2) # -> 8x8
        
        # === 2. Watermark Decoder Head ===
        # Global Average Pooling -> FC
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.wm_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, msg_length) # Output Logits
        )

    def _make_layer(self, in_ch, out_ch, blocks, stride=1):
        layers = []
        layers.append(BasicBlock(in_ch, out_ch, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Input: x [B, 3, 256, 256] (Generated Image)
        Output: logits [B, msg_length]
        """
        # Feature Extraction
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        f_map = self.layer4(x) # [B, 512, 8, 8] - Shared Feature Map
        
        # Watermark Decoding
        pool = self.avgpool(f_map)
        logits = self.wm_head(pool)
        
        return logits