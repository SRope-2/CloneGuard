"""
CloneGuard: Robust Watermarking via Parallel Stream Weight Cloning
Core Inference Pipeline for Double-Blind Review.

This script demonstrates the end-to-end forward pass of CloneGuard, 
including host image encoding, endogenous texture synthesis (watermark injection), 
and robust extraction.
"""

import argparse
import os
import math
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from typing import Tuple

from models.decoder import RAE_WF_Decoder_Clone
from models.extractor import CloneExtractor
from my_rae.src.stage1.rae import RAE 

class CloneGuardPipeline:
    """
    CloneGuard Inference Pipeline.
    Encapsulates the frozen DINOv2 encoder, Wavelet-Flow decoder with 
    Parallel Stream Weight Cloning, and the robust extractor.
    """
    def __init__(self, vae_ckpt_path: str, wm_ckpt_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.msg_length = 64
        
        print("[INFO] Initializing CloneGuard Pipeline...")
        self.encoder, self.decoder, self.extractor = self._build_models()
        
        # 渐进式双阶段权重加载
        self._load_weights(vae_ckpt_path, wm_ckpt_path)
        
        self.encoder.eval()
        self.decoder.eval()
        self.extractor.eval()

    def _build_models(self) -> Tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]:
        encoder = RAE(
            encoder_cls='Dinov2withNorm',
            encoder_config_path='dinov2-with-registers-base', 
            normalize=True
        ).to(self.device)
        encoder.requires_grad_(False) 

        decoder = RAE_WF_Decoder_Clone(
            rae_latent_dim=768,
            wf_vae_latent_dim=16,
            target_size=256,
            Clone_config={"msg_length": self.msg_length, "wm_dim": 128} 
        ).to(self.device)

        extractor = CloneExtractor(msg_length=self.msg_length).to(self.device)
        
        return encoder, decoder, extractor

    def _load_weights(self, vae_ckpt_path: str, wm_ckpt_path: str):
        if os.path.exists(vae_ckpt_path):
            vae_ckpt = torch.load(vae_ckpt_path, map_location='cpu')
            self.decoder.load_state_dict(vae_ckpt.get('decoder', vae_ckpt), strict=False)
            print(f"[INFO] Stage-1 WF-VAE base weights loaded from {vae_ckpt_path}")
        else:
            print(f"[WARNING] Stage-1 weights not found at {vae_ckpt_path}.")

        if os.path.exists(wm_ckpt_path):
            wm_ckpt = torch.load(wm_ckpt_path, map_location='cpu')
            self.decoder.load_state_dict(wm_ckpt.get('decoder', wm_ckpt), strict=False)
            self.extractor.load_state_dict(wm_ckpt.get('extractor', wm_ckpt), strict=False)
            print(f"[INFO] Stage-2 Watermark weights loaded from {wm_ckpt_path}")
        else:
            print(f"[WARNING] Stage-2 weights not found at {wm_ckpt_path}.")

    @torch.no_grad()
    def embed_watermark(self, img_tensor: torch.Tensor, msg_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds the watermark and also returns the clean reconstruction for comparison.
        Returns: (cover_image, stego_image)
        """
        # 1. Host Encoding
        img_resized = F.interpolate(img_tensor, size=(224, 224), mode='bicubic', align_corners=False)
        x = (img_resized - self.encoder.encoder_mean.to(self.device)) / self.encoder.encoder_std.to(self.device)
        z_raw = self.encoder.encoder(x)
        
        B_z, N_z, C_z = z_raw.shape
        H_feat = int(math.sqrt(N_z))
        z_raw = z_raw.transpose(1, 2).view(B_z, C_z, H_feat, H_feat)
        if H_feat != 16:
            z_raw = F.interpolate(z_raw, size=(16, 16), mode='bicubic', align_corners=False)
            
        z_norm = (z_raw - self.decoder.latent_mean) / (self.decoder.latent_std + 1e-5)
        
        # 2. Clean Reconstruction (Cover)
        raw_cover = self.decoder(z_norm, watermark_msg=None)
        cover_image = self._safe_normalize(raw_cover)

        # 3. Watermark Injection (Stego)
        raw_stego = self.decoder(z_norm, watermark_msg=msg_tensor)
        stego_image = self._safe_normalize(raw_stego)
        
        return cover_image, stego_image

    @torch.no_grad()
    def extract_watermark(self, stego_tensor: torch.Tensor) -> torch.Tensor:
        pred_logits = self.extractor(stego_tensor)
        pred_bits = (torch.sigmoid(pred_logits) > 0.5).float()
        return pred_bits

    def _safe_normalize(self, imgs: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(imgs)
        for i in range(imgs.shape[0]):
            img = imgs[i]
            if img.min() < -0.2: 
                out[i] = ((img + 1.0) / 2.0).clamp(0, 1)
            else:
                out[i] = img.clamp(0, 1)
        return out

def main():
    parser = argparse.ArgumentParser(description="CloneGuard Inference Tool")
    parser.add_argument('--input_image', type=str, required=True, help='Path to the host cover image')
    parser.add_argument('--msg', type=str, default="10101010" * 8, help='Binary message string (64 bits)')
    parser.add_argument('--vae_ckpt', type=str, default="./checkpoints/wf_vae.pt", help='Path to Stage-1 VAE weights')
    parser.add_argument('--wm_ckpt', type=str, default="./checkpoints/wm.pt", help='Path to Stage-2 Watermark weights')
    parser.add_argument('--output_dir', type=str, default="./results", help='Directory to save the images')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Initialize
    pipeline = CloneGuardPipeline(vae_ckpt_path=args.vae_ckpt, wm_ckpt_path=args.wm_ckpt, device=device)

    # 2. Process Input
    transform = transforms.Compose([transforms.ToTensor()])
    input_img = Image.open(args.input_image).convert('RGB')
    input_tensor = transform(input_img).unsqueeze(0).to(device)
    msg_tensor = torch.tensor([[int(b) for b in args.msg[:64]]], dtype=torch.float32).to(device)

    # 3. Forward Pass (Get both Cover and Stego)
    print(f"[INFO] Processing image: {args.input_image}...")
    cover_tensor, stego_tensor = pipeline.embed_watermark(input_tensor, msg_tensor)
    
    # 4. Save Cover Image
    cover_np = (cover_tensor.squeeze().cpu().permute(1, 2, 0).numpy() * 255).astype('uint8')
    cover_path = os.path.join(args.output_dir, "cover_output.png")
    Image.fromarray(cover_np).save(cover_path)
    
    # 5. Save Stego Image
    stego_np = (stego_tensor.squeeze().cpu().permute(1, 2, 0).numpy() * 255).astype('uint8')
    stego_path = os.path.join(args.output_dir, "stego_output.png")
    Image.fromarray(stego_np).save(stego_path)
    
    print(f"[SUCCESS] Clean Cover saved to: {cover_path}")
    print(f"[SUCCESS] Watermarked Stego saved to: {stego_path}")

    # 6. Verify Extraction
    print("[INFO] Verifying watermark extraction...")
    extracted_bits = pipeline.extract_watermark(stego_tensor)
    bit_acc = (extracted_bits == msg_tensor).float().mean().item()
    print(f"[SUCCESS] Extraction Accuracy: {bit_acc * 100:.2f}%")

if __name__ == "__main__":
    main()