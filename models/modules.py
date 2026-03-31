import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNSelu(nn.Module):
    """
    基础卷积块：Conv + BN + SELU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBNSelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # 使用普通 BatchNorm2d，单卡训练足够；如果多卡且 batch_size 很小，可能需要 SyncBatchNorm
        self.bn = nn.BatchNorm2d(out_channels) 
        self.selu = nn.SELU()

    def forward(self, x):
        return self.selu(self.bn(self.conv(x)))

class MessageEncoder(nn.Module):
    """
    将二进制消息 (B, 64) 编码为初始水印特征 (B, 16, 16, 16)
    """
    def __init__(self, input_length=64, output_channels=16, img_size=16):
        super(MessageEncoder, self).__init__()
        self.img_size = img_size
        self.output_channels = output_channels

        # 全连接层：将比特映射为高维特征
        self.fc_layers = nn.Sequential(
            nn.Linear(input_length, 256),
            nn.SELU(inplace=True),
            nn.Linear(256, 1024),
            nn.SELU(inplace=True),
            # 输出大小适配: output_channels * img_size * img_size (16*16*16 = 4096)
            nn.Linear(1024, output_channels * img_size * img_size) 
        )

        # 卷积层：平滑特征
        self.conv_layers = nn.Sequential(
            ConvBNSelu(output_channels, output_channels),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        )

    def forward(self, m):
        batch_size = m.shape[0]
        
        # 1. FC 处理
        x = self.fc_layers(m)
        
        # 2. Reshape 为图像特征格式 [B, C, H, W]
        x = x.view(batch_size, self.output_channels, self.img_size, self.img_size)
        
        # 3. Conv 处理
        mw = self.conv_layers(x)
        return mw

class ExpandNet(nn.Module):
    """
    用于将水印特征映射到中间层维度
    例如: 把 16 通道水印 -> 512 通道特征
    """
    def __init__(self, in_channels, out_channels):
        super(ExpandNet, self).__init__()
        self.layers = nn.Sequential(
            ConvBNSelu(in_channels, out_channels),
            ConvBNSelu(out_channels, out_channels),
            # 1x1 卷积做最终的通道融合
            nn.Conv2d(out_channels, out_channels, kernel_size=1) 
        )

    def forward(self, x):
        return self.layers(x)