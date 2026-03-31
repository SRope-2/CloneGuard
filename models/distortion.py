import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

class DistortionLayer(nn.Module):
    """
    可微攻击层 (Differentiable Distortion Layer)
    包含：Identity, Noise, Blur, Resize(Down+Up), Crop
    """
    def __init__(self, probability=0.8):
        super().__init__()
        self.prob = probability
        
        # --- 预定义模糊核 (Gaussian Kernel) ---
        kernel_size = 5
        sigma = 1.5
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
        
        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.
        gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                          torch.exp(
                              -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                              (2 * variance)
                          )
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        
        # [3, 1, 5, 5] 用于 Depthwise Conv
        self.register_buffer('blur_kernel', gaussian_kernel.view(1, 1, kernel_size, kernel_size).repeat(3, 1, 1, 1))
        self.blur_pad = kernel_size // 2

    def forward(self, x):
        """
        x: [B, 3, H, W] 范围 [0, 1]
        """
        if not self.training:
            return x
            
        # 按照概率决定是否攻击
        if torch.rand(1) > self.prob:
            return x
            
        # 随机选择一种攻击方式
        attack_type = random.choices(
            ['noise', 'blur', 'resize', 'dropout', 'combined'], 
            weights=[0.1, 0.5, 0.1, 0.1, 0.2], # 权重可调
            k=1
        )[0]
        
        if attack_type == 'noise':
            return self.gaussian_noise(x)
        elif attack_type == 'blur':
            return self.gaussian_blur(x)
        elif attack_type == 'resize':
            return self.resize_attack(x)
        elif attack_type == 'dropout':
            return self.random_dropout(x)
        elif attack_type == 'combined':
            # 混合双打：缩放 + 噪声
            x = self.resize_attack(x)
            return self.gaussian_noise(x)
            
        return x

    def gaussian_noise(self, x, std_min=0.01, std_max=0.1):
        """加性高斯噪声"""
        # 🔥 修复：std 是 float，直接乘即可，不需要 .to()
        std = torch.rand(1).item() * (std_max - std_min) + std_min
        noise = torch.randn_like(x) * std
        return (x + noise).clamp(0, 1)

    def gaussian_blur(self, x):
        """高斯模糊"""
        return F.conv2d(x, self.blur_kernel, padding=self.blur_pad, groups=3)

    def resize_attack(self, x, scale_min=0.25, scale_max=0.9):

            B, C, H, W = x.shape
            
            if random.random() < 0.5:
                scale = random.choice([0.25, 0.5]) # 针对性训练
            else:
                scale = torch.rand(1).item() * (scale_max - scale_min) + scale_min
                
            new_h = int(H * scale)
            new_w = int(W * scale)
            

            # 'area' 模式通常最难，因为它会平均化像素信息
            mode = random.choice(['bilinear', 'bicubic', 'area', 'nearest'])
            
            # 注意: 'area' 和 'nearest' 可能会有维度限制或对齐问题，通常 bilinear/bicubic 最稳
            # 如果报错，可以回退到只用 bilinear/bicubic
            if mode == 'area':
                # area 插值需要输入是 float
                small = F.interpolate(x, size=(new_h, new_w), mode='area')
            elif mode == 'nearest':
                small = F.interpolate(x, size=(new_h, new_w), mode='nearest')
            else:
                small = F.interpolate(x, size=(new_h, new_w), mode=mode, align_corners=False)
            
            # 恢复尺寸 (模拟 Extractor 接收到的输入)
            # 恢复时通常攻击者不知道原图算法，所以通常默认用 bilinear 或 bicubic
            restore_mode = random.choice(['bilinear', 'bicubic'])
            restored = F.interpolate(small, size=(H, W), mode=restore_mode, align_corners=False)
            
            return restored

    def random_dropout(self, x, ratio_min=0.1, ratio_max=0.3):
        """
        随机遮挡攻击 (模拟裁剪或涂抹)
        """
        B, C, H, W = x.shape
        mask = torch.ones_like(x)
        
        # 简单的矩形遮挡
        h_drop = int(H * (torch.rand(1).item() * (ratio_max - ratio_min) + ratio_min))
        w_drop = int(W * (torch.rand(1).item() * (ratio_max - ratio_min) + ratio_min))
        
        top = torch.randint(0, H - h_drop, (1,)).item()
        left = torch.randint(0, W - w_drop, (1,)).item()
        
        mask[:, :, top:top+h_drop, left:left+w_drop] = 0
        return x * mask