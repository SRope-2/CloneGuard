import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os

# 从同目录的 modules.py 导入基础组件
from models.modules import MessageEncoder, ExpandNet

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any


# ==================== Core Modules from WF-VAE ====================

class Conv2d(nn.Conv2d):
    """Standard 2D convolution wrapper"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Normalize(nn.Module):
    """Normalization layer supporting different norm types"""
    def __init__(self, in_channels: int, norm_type: str = "layernorm"):
        super().__init__()
        self.norm_type = norm_type
        if norm_type == "layernorm":
            self.norm = nn.GroupNorm(1, in_channels, eps=1e-6, affine=True)
        elif norm_type == "groupnorm":
            self.norm = nn.GroupNorm(32, in_channels, eps=1e-6, affine=True)
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")
    
    def forward(self, x):
        return self.norm(x)


def nonlinearity(x):
    """SiLU activation function"""
    return x * torch.sigmoid(x)


class ResnetBlock2D(nn.Module):
    """2D ResNet block with optional dropout"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        norm_type: str = "layernorm"
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.norm1 = Normalize(in_channels, norm_type=norm_type)
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.norm2 = Normalize(out_channels, norm_type=norm_type)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        if in_channels != out_channels:
            self.shortcut = Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)


class Upsample(nn.Module):
    """2x upsampling with optional convolution"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class InverseHaarWaveletTransform2D(nn.Module):
    """Inverse 2D Haar wavelet transform"""
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """
        Input: [B, 12, H, W] where channels are [LL_R, LL_G, LL_B, LH_R, LH_G, LH_B, HL_R, HL_G, HL_B, HH_R, HH_G, HH_B]
        Output: [B, 3, 2H, 2W]
        """
        B, C, H, W = x.shape
        assert C == 12, f"Expected 12 channels, got {C}"
        
        # Split into RGB components
        LL = x[:, 0:3]  # Low-Low
        LH = x[:, 3:6]  # Low-High
        HL = x[:, 6:9]  # High-Low
        HH = x[:, 9:12] # High-High
        
        # Reconstruct image from wavelet coefficients
        out = torch.zeros(B, 3, 2*H, 2*W, device=x.device, dtype=x.dtype)
        out[:, :, 0::2, 0::2] = (LL + LH + HL + HH) / 2.0
        out[:, :, 0::2, 1::2] = (LL + LH - HL - HH) / 2.0
        out[:, :, 1::2, 0::2] = (LL - LH + HL - HH) / 2.0
        out[:, :, 1::2, 1::2] = (LL - LH - HL + HH) / 2.0
        
        return out


class WFUpBlock(nn.Module):
    """Wavelet Flow Upsampling Block for 2D images"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        energy_flow_size: int = 128,
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        norm_type: str = "layernorm",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.energy_flow_size = energy_flow_size
        assert num_res_blocks >= 2, "num_res_blocks must be at least 2"
        
        # Branch convolution to split features into main path and energy flow
        self.branch_conv = ResnetBlock2D(
            in_channels=in_channels,
            out_channels=in_channels + energy_flow_size,
            dropout=dropout,
            norm_type=norm_type
        )
        
        # Energy flow path to wavelet coefficients
        self.out_flow_conv = nn.Sequential(
            ResnetBlock2D(
                in_channels=energy_flow_size,
                out_channels=energy_flow_size,
                dropout=dropout,
                norm_type=norm_type
            ),
            Conv2d(
                in_channels=energy_flow_size,
                out_channels=12,  # 12 channels for 2D Haar wavelet (4 subbands × 3 RGB)
                kernel_size=3,
                stride=1,
                padding=1
            )
        )
        
        self.inverse_wavelet_transform = InverseHaarWaveletTransform2D()
        
        # Main processing path
        self.res_block = nn.Sequential(
            *[ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                dropout=dropout,
                norm_type=norm_type
            ) for _ in range(num_res_blocks - 2)]
        )
        
        self.up = Upsample(in_channels=in_channels, out_channels=in_channels)
        self.out_res_block = ResnetBlock2D(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            norm_type=norm_type
        )
    
    def forward(self, x, w=None):
        # Branch into main path and energy flow
        x = self.branch_conv(x)
        
        # Energy flow to wavelet coefficients
        coeffs = self.out_flow_conv(x[:, -self.energy_flow_size:])
        
        # Add residual from previous layer if available
        if w is not None:
            coeffs[:, :3] = coeffs[:, :3] + w
        
        # Inverse wavelet transform
        w = self.inverse_wavelet_transform(coeffs)
        
        # Main path processing
        x = self.res_block(x[:, :-self.energy_flow_size])
        x = self.up(x)
        
        return self.out_res_block(x), w, coeffs


# ==================== WF-VAE Decoder ====================

class WFVAEDecoder(nn.Module):
    """WF-VAE Decoder for 2D images"""
    def __init__(
        self,
        latent_dim: int = 16,
        num_resblocks: int = 3,
        dropout: float = 0.0,
        energy_flow_size: int = 128,
        norm_type: str = "layernorm",
        base_channels: List[int] = [128, 256, 512],
        up_layer_type: List[str] = ["hw", "hw"],
    ):
        super().__init__()
        self.energy_flow_size = energy_flow_size
        
        # Input convolution
        self.conv_in = Conv2d(
            latent_dim,
            base_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        # Middle layers
        self.mid = nn.Sequential(
            ResnetBlock2D(
                in_channels=base_channels[-1],
                out_channels=base_channels[-1],
                dropout=dropout,
                norm_type=norm_type
            ),
            ResnetBlock2D(
                in_channels=base_channels[-1],
                out_channels=base_channels[-1],
                dropout=dropout,
                norm_type=norm_type
            ),
        )
        
        # ========== 🔧 修复：Upsampling blocks ==========
        self.up_blocks = nn.ModuleList()
        num_up_blocks = len(up_layer_type)
        
        # 验证 base_channels 长度
        # 需要：len(base_channels) = len(up_layer_type) + 1
        assert len(base_channels) >= num_up_blocks + 1, \
            f"base_channels must have at least {num_up_blocks + 1} elements for {num_up_blocks} upsampling layers, " \
            f"got {len(base_channels)}. Example: for 3 up layers, need [C1, C2, C3, C4]."
        
        # 正确的索引方式
        for idx in range(num_up_blocks):
            # 从后往前：base_channels[-1] -> base_channels[-2] -> ... -> base_channels[0]
            in_ch_idx = len(base_channels) - 1 - idx
            out_ch_idx = in_ch_idx - 1
            
            up_block = WFUpBlock(
                in_channels=base_channels[in_ch_idx],
                out_channels=base_channels[out_ch_idx],
                energy_flow_size=energy_flow_size,
                num_res_blocks=num_resblocks,
                dropout=dropout,
                norm_type=norm_type
            )
            self.up_blocks.append(up_block)
        
        # Output layers
        self.norm_out = Normalize(base_channels[0], norm_type=norm_type)
        self.conv_out = Conv2d(
            base_channels[0],
            12,  # 12 channels for 2D Haar wavelet output
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.inverse_wavelet_transform_out = InverseHaarWaveletTransform2D()
    
    def forward(self, z):
        # Input processing
        h = self.conv_in(z)
        h = self.mid(h)
        
        # Progressive upsampling with wavelet flows
        inter_coeffs = []
        w = None
        for up_block in self.up_blocks:
            h, w, coeffs = up_block(h, w)
            inter_coeffs.append(coeffs)
        
        # Final output
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        
        # Add final wavelet residual
        h[:, :3] = h[:, :3] + w
        
        # Inverse wavelet transform to get final image
        dec = self.inverse_wavelet_transform_out(h)
        
        return dec, inter_coeffs


# ==================== RAE-WF Decoder Wrapper ====================

class RAE_WF_Decoder(nn.Module):
    """
    Wrapper class to adapt WF-VAE decoder to RAE framework.
    
    This adapter:
    1. Takes RAE latent codes (typically 768 channels from DINOv2)
    2. Reduces dimensionality via a conv adapter
    3. Passes through WF-VAE decoder for high-quality reconstruction
    4. Outputs at target resolution (224 or 256)
    """
    def __init__(
        self,
        rae_latent_dim: int = 768,
        wf_vae_latent_dim: int = 16,
        decoder_body_config: Optional[Dict[str, Any]] = None,
        target_size: int = 256,
    ):
        super().__init__()
        
        self.target_size = target_size
        
        # Default configuration for decoder body
        if decoder_body_config is None:
            # 默认配置：输出 256×256
            decoder_body_config = {
                "latent_dim": wf_vae_latent_dim,
                "num_resblocks": 3,
                "dropout": 0.0,
                "energy_flow_size": 128,
                "norm_type": "layernorm",
                "base_channels": [128, 256, 512, 512],  # 4 个配置对应 3 层上采样
                "up_layer_type": ["hw", "hw", "hw"],    # 3 层
            }
        else:
            # Ensure latent_dim matches wf_vae_latent_dim
            decoder_body_config["latent_dim"] = wf_vae_latent_dim
        
        # Adapter: reduce RAE latent dimension to WF-VAE latent dimension
        self.adapter = nn.Conv2d(
            in_channels=rae_latent_dim,
            out_channels=wf_vae_latent_dim,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        # WF-VAE decoder body
        self.decoder_body = WFVAEDecoder(**decoder_body_config)
        
        print(f"[RAE_WF_Decoder] Target output size: {target_size}x{target_size}")
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: RAE latent codes, shape [B, rae_latent_dim, H, W]
               e.g., [B, 768, 16, 16] for DINOv2-Base
        
        Returns:
            decoded_image: Reconstructed image, shape [B, 3, target_size, target_size]
                          e.g., [B, 3, 256, 256] for target_size=256
        """
        # Step 1: Adapt from RAE latent space to WF-VAE latent space
        z_adapted = self.adapter(z)  # [B, rae_latent_dim, H, W] -> [B, wf_vae_latent_dim, H, W]
        
        # Step 2: Pass through WF-VAE decoder
        decoded_image, inter_coeffs = self.decoder_body(z_adapted)
        
        # Step 3: 🔧 验证/调整输出尺寸
        _, _, H, W = decoded_image.shape
        expected_size = self.target_size
        
        if H != expected_size or W != expected_size:
            # 这应该只在配置错误时发生 - 输出警告
            if not hasattr(self, '_resize_warned'):
                print(f"⚠️ [RAE_WF_Decoder] WARNING: Decoder output {H}x{W} != target {expected_size}x{expected_size}")
                print(f"   Applying resize - this may reduce quality!")
                print(f"   Consider adjusting decoder config:")
                print(f"   - For 256 output: base_channels=[128,256,512,512], up_layer_type=['hw','hw','hw']")
                print(f"   - For 224 output: base_channels=[128,256,512], up_layer_type=['hw','hw']")
                self._resize_warned = True
            
            decoded_image = torch.nn.functional.interpolate(
                decoded_image,
                size=(expected_size, expected_size),
                mode='bicubic',
                align_corners=False
            )
        elif not hasattr(self, '_size_confirmed'):
            print(f"✅ [RAE_WF_Decoder] Output size {H}x{W} matches target {expected_size}x{expected_size} - no resize needed")
            self._size_confirmed = True
        
        return decoded_image

class WFVAEDecoder_ParallelClone(WFVAEDecoder):
    """
    并行流 + 权重克隆机制
    """
    def __init__(self, *args, Clone_config=None, **kwargs):
        super().__init__(*args, **kwargs)


class RAE_WF_Decoder_Clone(RAE_WF_Decoder):
    """
    针对 CloneGuard 封装的 RAE 解码器
    """
    def __init__(self, *args, Clone_config=None, stats_path=None, **kwargs):
        # 显式定义 decoder_body 为我们的并行克隆流
        super().__init__(*args, **kwargs)
