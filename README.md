#  CloneGuard: Robust Watermarking via Parallel Stream Weight Cloning

![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)
![ICME 2026](https://img.shields.io/badge/Accepted-ICME_2026-success)
![License](https://img.shields.io/badge/License-MIT-green)

> **🎉 News:** Our paper **"CloneGuard"** has been accepted by **IEEE ICME 2026**! (To appear). 
> 
> *Note: Since the paper is currently in the pre-publication stage to help researchers and developers quickly verify our inference pipeline, we provide **Lightweight Demo Checkpoints** below.*


---

## 📖 Introduction 

**CloneGuard** is a novel watermarking framework designed for Artificial Intelligence Generated Content (AIGC) copyright protection. Unlike traditional noise injection methods, it establishes a new paradigm of endogenous texture synthesis. 


---

## 🚀 Pre-trained Demo Weights 

To run the inference pipeline, you need both the base Wavelet-Flow VAE weights (Stage 1) and the Watermark Injection/Extraction weights (Stage 2). 


| Model Component | Checkpoint Type | Download Link |
| :--- | :---: | :--- |
| **Stage 1**: WF-VAE Decoder | Base Autoencoder | [wf_rae.pt](https://drive.google.com/file/d/1LOEq2U7h2DJGmIANFMndw7DTSPtSb6a2/view?usp=drive_link) |
| **Stage 2**: CloneGuard (GenPTW) | Watermark Network | [wm.pt](https://drive.google.com/file/d/1M74H2cHpE4e9HK28eEZavvkJZ3h-rSQo/view?usp=drive_link) |

*Please download both weights and place them in the `checkpoints/` directory.*

---

## 🛠️ Quick Start 

### 1. Environment Setup 

We recommend using `conda` to construct the virtual environment. Core dependencies are minimal to ensure out-of-the-box evaluation.

```bash
conda create -n cloneguard python=3.9
conda activate cloneguard
pip install torch torchvision torchaudio
pip install -r requirements.txt
```


### 2. Watermark Embedding & Extraction 
We provide a clean, decoupled inference script to demonstrate the forward pass.
Ensure your weights are placed in the checkpoints/ folder
```bash
python inference.py \
    --input_image ./assets/sample_cover.jpg \
    --msg "1010101010101010" \
    --vae_ckpt ./checkpoints/wf_rae.pt \
    --wm_ckpt ./checkpoints/wm.pt \
    --output_dir ./results/
```
### 3. Training Pipeline 
For researchers aiming to understand or reproduce the training phase on custom datasets, we provide the core training script entry:
```bash
python train.py --data_dir /path/to/your/dataset
```





## Acknowledgement

This code is built upon the following repositories:

* [RAE](https://github.com/bytetriper/RAE/tree/main)
* [WF-VAE](https://github.com/PKU-YuanGroup/WF-VAE.git)
## Citation

If you find this code useful in your research, please cite our paper accepted by **ICME 2026**:

```bibtex
@inproceedings{wang2026cloneguard,
  title={CloneGuard: Robust Watermarking via Parallel Stream Weight Cloning},
  author={Wang, Huazhong and Sun, Linhao and Zhu, Zhiying and Hou, Zhenxuan and Jiang, Qingchao and Ba, Zhongjie and Gu, Zaiwang},
  booktitle={2026 IEEE International Conference on Multimedia and Expo (ICME)},
  year={2026},
  note={To appear},
}
Note: This work has been accepted to ICME 2026. The final publication details (page numbers, DOI) will be updated once available.
