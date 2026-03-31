"""
CloneGuard: Robust Watermarking via Parallel Stream Weight Cloning
Training Pipeline for Stage-2 (Watermark Injection & Extraction)
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import sys
import os
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==========================================
# 1. Environment & Imports
# ==========================================
# Ensure models/ can be imported
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(current_file_path)
if project_root not in sys.path:
    sys.path.append(project_root)

from models.decoder import RAE_WF_Decoder_Clone
from models.extractor import CloneExtractor
from models.distortion import DistortionLayer  
# Requires the RAE repository to be in your PYTHONPATH
from my_rae.src.stage1.rae import RAE

# ==========================================
# 2. Configuration Center
# ==========================================
class Config:
    # 维度与架构参数
    rae_latent_dim = 768
    wf_latent_dim = 16
    msg_length = 64 
    wm_dim = 128
    target_size = 256
    
    decoder_body_config = {
        "latent_dim": 16, "num_resblocks": 3, "dropout": 0.0,
        "energy_flow_size": 128, "norm_type": "layernorm",
        "base_channels": [128, 256, 512, 512], "up_layer_type": ["hw", "hw", "hw"]
    }
    
    # 训练超参数
    batch_size = 24
    epochs = 10
    num_workers = 8
    lr = 1e-5         
    
    # Loss 目标：让 Stego 尽可能接近 Cover (Clean Reconstruction)
    lambda_wm = 8.0
    lambda_img = 20.0    # 约束 Stego 和 Cover 的 MSE
    lambda_lpips = 15.0  # 约束 Stego 和 Cover 的感知差异
    
    distortion_prob = 0.6 

# ==========================================
# 3. Utility Functions
# ==========================================
def robust_normalize_batch(imgs):
    """Post-processing to map outputs safely to [0, 1]"""
    B = imgs.shape[0]
    out = torch.zeros_like(imgs)
    for i in range(B):
        img = imgs[i]
        min_v, max_v = img.min(), img.max()
        if min_v >= -0.1 and max_v <= 1.1:
            out[i] = img.clamp(0, 1)
        elif min_v >= -1.1 and max_v <= 1.1:
            out[i] = ((img + 1.0) / 2.0).clamp(0, 1)
        else:
            out[i] = (img - min_v) / (max_v - min_v + 1e-8)
    return out

def calculate_psnr(img1, img2):
    with torch.no_grad():
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0: return 100.0
        psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
    return psnr.item()

def calculate_accuracy(pred_logits, gt_msg):
    with torch.no_grad():
        pred_bits = (torch.sigmoid(pred_logits) > 0.5).float()
        correct = (pred_bits == gt_msg).sum().item()
        total = gt_msg.numel()
    return correct / total

def save_debug_images(real, cover, stego, epoch, step, output_dir):
    """Saves visual comparisons: Real vs Cover vs Stego vs Diff Heatmap"""
    with torch.no_grad():
        N = min(real.shape[0], 8)
        real, cover, stego = real[:N].cpu(), cover[:N].cpu(), stego[:N].cpu()
        diff = torch.abs(stego - cover) * 10.0
        
        rows = [torch.cat([real[i], cover[i], stego[i], diff[i]], dim=2) for i in range(N)]
        grid = torch.cat(rows, dim=1)
        
        save_path = os.path.join(output_dir, f"vis_epoch{epoch+1}_step{step}.png")
        save_image(grid, save_path, nrow=N)

# ==========================================
# 4. Main Training Loop
# ==========================================
def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Starting CloneGuard Training on {device}")

    # --- 1. Data Loader ---
    print(f"[INFO] Loading Dataset from: {args.data_dir}")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    
    try:
        dataset = datasets.ImageFolder(args.data_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True, 
                                num_workers=Config.num_workers, pin_memory=True)
        print(f"[SUCCESS] Found {len(dataset)} images.")
    except Exception as e:
        print(f"[ERROR] Dataset load failed: {e}")
        return

    # --- 2. Models Initialization ---
    print("[INFO] Initializing Models...")
    
    rae_encoder = RAE(
        encoder_cls='Dinov2withNorm',
        encoder_config_path=args.encoder_cfg,
        encoder_params={'dinov2_path': args.encoder_cfg, 'normalize': True},
        normalization_stat_path=args.stats_path,
        decoder_config={'target': 'torch.nn.Identity', 'params': {}}
    ).to(device).eval()
    rae_encoder.requires_grad_(False)
    
    decoder = RAE_WF_Decoder_Clone(
        rae_latent_dim=Config.rae_latent_dim,
        wf_vae_latent_dim=Config.wf_latent_dim,
        target_size=Config.target_size,
        Clone_config={"msg_length": Config.msg_length, "wm_dim": Config.wm_dim},
        decoder_body_config=Config.decoder_body_config,
        stats_path=args.stats_path
    ).to(device)
    
    extractor = CloneExtractor(msg_length=Config.msg_length).to(device)
    distorter = DistortionLayer(probability=Config.distortion_prob).to(device)
    
    # LPIPS Metric
    try:
        import lpips
        loss_fn_vgg = lpips.LPIPS(net='vgg').to(device).eval()
        has_lpips = True
    except ImportError:
        print("[WARNING] LPIPS library not found. Running without LPIPS loss.")
        has_lpips = False

    # --- 3. Weights Loading ---
    if args.resume_ckpt and os.path.exists(args.resume_ckpt):
        print(f"[INFO] Resuming from checkpoint: {args.resume_ckpt}")
        ckpt = torch.load(args.resume_ckpt, map_location='cpu')
        decoder.load_state_dict(ckpt['decoder'])
        extractor.load_state_dict(ckpt['extractor'])
    else:
        print("[INFO] Starting with Stage-1 Base WF-VAE weights.")
        if os.path.exists(args.vae_ckpt):
            decoder.load_base_weights(args.vae_ckpt)

    # --- 4. Optimizer ---
    trainable_params = []
    for name, param in decoder.named_parameters():
        if 'wm_' in name: 
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False
    for param in extractor.parameters():
        param.requires_grad = True
        trainable_params.append(param)
        
    optimizer = optim.AdamW(trainable_params, lr=Config.lr)
    scaler = torch.cuda.amp.GradScaler()
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_mse = nn.MSELoss()

    # --- 5. Training Epochs ---
    decoder.train()
    extractor.train()
    for module in decoder.modules():
        if isinstance(module, nn.BatchNorm2d) and not any(p.requires_grad for p in module.parameters()):
            module.eval()

    for epoch in range(Config.epochs):
        metrics = {'loss': 0, 'acc': 0, 'psnr': 0}
        steps = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{Config.epochs}")
        
        for step_idx, (real_imgs, _) in enumerate(pbar):
            real_imgs = real_imgs.to(device, non_blocking=True)
            B = real_imgs.shape[0]
            msg = torch.randint(0, 2, (B, Config.msg_length)).float().to(device)
            
            with torch.no_grad():
                x = (real_imgs - rae_encoder.encoder_mean.to(device)) / rae_encoder.encoder_std.to(device)
                z_raw = rae_encoder.encoder(x) 
                z_raw = z_raw.transpose(1, 2).view(B, 768, 16, 16)
                
                l_mean, l_std = decoder.latent_mean, decoder.latent_std
                z_norm = (z_raw - l_mean) / (l_std + 1e-5)

            with torch.cuda.amp.autocast():
                # Cover Image (Clean Reconstruction)
                with torch.no_grad():
                    raw_clean = decoder(z_norm, watermark_msg=None)
                    img_clean = robust_normalize_batch(raw_clean)
                
                # Stego Image (Watermarked)
                raw_wm = decoder(z_norm, watermark_msg=msg)
                img_wm = robust_normalize_batch(raw_wm)
                
                # Forward Extraction
                img_distorted = distorter(img_wm)
                pred_logits = extractor(img_distorted)
                
                loss_wm = criterion_bce(pred_logits, msg)
                loss_mse = criterion_mse(img_wm, img_clean) 
                
                loss_lpips = 0
                if has_lpips:
                    in_wm = (img_wm * 2 - 1).clamp(-1, 1)
                    in_cl = (img_clean * 2 - 1).clamp(-1, 1)
                    loss_lpips = loss_fn_vgg(in_wm, in_cl).mean()
                    
                total_loss = Config.lambda_wm * loss_wm + \
                             Config.lambda_img * loss_mse + \
                             Config.lambda_lpips * loss_lpips

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            acc = calculate_accuracy(pred_logits, msg)
            psnr = calculate_psnr(img_wm, img_clean)
            
            metrics['loss'] += total_loss.item()
            metrics['acc'] += acc
            metrics['psnr'] += psnr
            steps += 1
            
            pbar.set_postfix({'Acc': f"{acc:.1%}", 'PSNR': f"{psnr:.1f}"})
            
            if step_idx == 0 and (epoch % 5 == 0 or epoch == 0):
                real_imgs_256 = torch.nn.functional.interpolate(real_imgs, size=(256, 256), mode='bicubic')
                save_debug_images(real_imgs_256, img_clean, img_wm, epoch, step_idx, args.output_dir)

        avg = {k: v/steps for k,v in metrics.items()}
        print(f"[Epoch {epoch+1} Summary] Acc: {avg['acc']:.2%} | PSNR: {avg['psnr']:.2f} dB")
        
        save_name = f"cloneguard_ep{epoch+1}_acc{avg['acc']:.2f}_psnr{avg['psnr']:.1f}.pt"
        save_path = os.path.join(args.output_dir, save_name)
        torch.save({
            'decoder': decoder.state_dict(),
            'extractor': extractor.state_dict(),
            'optimizer': optimizer.state_dict()
        }, save_path)
        print(f"[SUCCESS] Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CloneGuard Training Script")
    parser.add_argument('--data_dir', type=str, default="./dataset/imagenet", help="Path to training dataset")
    parser.add_argument('--output_dir', type=str, default="./checkpoints/training_outputs", help="Directory to save checkpoints")
    parser.add_argument('--vae_ckpt', type=str, default="./checkpoints/wf_vae.pt", help="Path to Stage-1 WF-VAE weights")
    parser.add_argument('--resume_ckpt', type=str, default="", help="Path to resume training from a checkpoint")
    parser.add_argument('--stats_path', type=str, default="./configs/stat.pt", help="Path to DINOv2 normalization stats")
    parser.add_argument('--encoder_cfg', type=str, default="dinov2-with-registers-base", help="Encoder configuration")
    args = parser.parse_args()
    
    train(args)