import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


# 简化版的Mamba模型实现
class MambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, expand=2):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.expand = expand
        
        # 前馈扩展
        self.in_proj = nn.Linear(dim, dim * expand * 2)
        
        # 门控激活
        self.gate_proj = nn.Linear(dim, dim * expand)
        
        # 状态空间模型参数
        self.A_log = nn.Parameter(torch.log(torch.ones(d_state)))
        self.D = nn.Parameter(torch.ones(dim * expand))
        self.B_proj = nn.Linear(dim * expand, d_state)
        self.C_proj = nn.Linear(d_state, dim * expand)
        
        # 输出投影
        self.out_proj = nn.Linear(dim * expand, dim)
        
        # 层归一化
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, dim]
        residual = x
        x = self.norm(x)
        
        # 前馈扩展
        x = self.in_proj(x)
        x1, x2 = x.chunk(2, dim=-1)
        
        # 门控激活
        gate = F.silu(self.gate_proj(residual))
        x = F.silu(x1) * x2 * gate
        
        # 状态空间模型计算（更稳定的实现）
        seq_len = x.shape[1]
        # 确保A是正数且有界
        A = -torch.exp(self.A_log.clamp(max=5.0)).view(-1)  # 限制指数增长
        
        # 使用更稳定的B投影
        B = self.B_proj(x)
        B = torch.tanh(B)  # 增加稳定性
        
        # 简化的SSM计算（批处理方式，更高效）
        h = torch.zeros(B.shape[0], self.d_state, device=x.device)
        outputs = []
        
        # 添加数值稳定性保护
        for i in range(seq_len):
            # 应用指数衰减
            decay = torch.exp(A)  # 已经是正数且小于1
            # 防止数值下溢
            decay = decay.clamp(min=1e-6)
            
            # 更新状态
            h = h * decay + B[:, i]
            
            # 计算输出
            output = F.linear(h, self.C_proj.weight, self.C_proj.bias)
            outputs.append(output)
        
        y = torch.stack(outputs, dim=1)
        
        # 使用D作为缩放因子，添加数值稳定性
        D = torch.sigmoid(self.D) * 2.0  # 限制D的范围在0-2
        y = y * D.view(1, 1, -1)
        
        # 输出投影，添加数值裁剪
        y = self.out_proj(y)
        y = y.clamp(min=-1e5, max=1e5)  # 防止极端值
        
        # 残差连接
        y = residual + y
        
        # 最终数值检查
        if torch.isnan(y).any():
            print("Warning: NaN detected in MambaBlock output")
            y = torch.nan_to_num(y, nan=0.0)
        
        return y

class MambaModel(nn.Module):
    def __init__(self, 
                 image_size=224,
                 patch_size=16,
                 in_channels=3,
                 dim=512,
                 depth=6,
                 num_classes=7,
                 d_state=16):
        super().__init__()
        
        # 计算patch数量和嵌入维度
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # 图像到patch的投影
        self.patch_embedding = nn.Conv2d(
            in_channels=in_channels,
            out_channels=dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # 位置嵌入
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))
        
        # Mamba层
        self.mamba_layers = nn.ModuleList([
            MambaBlock(dim=dim, d_state=d_state)
            for _ in range(depth)
        ])
        
        # 分类头
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, num_classes)
    
    def forward(self, x):
        # x shape: [batch_size, channels, height, width]
        
        # 转换为patches
        x = self.patch_embedding(x)
        # x shape: [batch_size, dim, num_patches^(1/2), num_patches^(1/2)]
        
        # 展平
        batch_size = x.shape[0]
        x = x.flatten(2).transpose(1, 2)
        # x shape: [batch_size, num_patches, dim]
        
        # 添加位置嵌入
        x = x + self.pos_embedding
        
        # 通过Mamba层
        for layer in self.mamba_layers:
            x = layer(x)
        
        # 全局平均池化
        x = self.norm(x.mean(dim=1))
        
        # 分类
        x = self.classifier(x)
        
        return x

def create_mamba_model(num_classes=7, image_size=224):
    """创建Mamba模型实例
    
    Args:
        num_classes: 类别数量
        image_size: 输入图像大小
        
    Returns:
        配置好的Mamba模型
    """
    # 使用更小的模型尺寸以增加稳定性
    model = MambaModel(
        image_size=image_size,
        patch_size=16,
        in_channels=3,
        dim=256,  # 减小维度以增加稳定性
        depth=4,  # 减小深度以减少训练难度
        num_classes=num_classes,
        d_state=16
    )
    
    # 更稳定的权重初始化
    def init_weights(m):
        if isinstance(m, nn.Linear):
            # 使用Xavier初始化替代Kaiming，对Mamba更稳定
            nn.init.xavier_uniform_(m.weight, gain=0.1)  # 较小的gain避免大权重
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=0.1)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif hasattr(m, 'A_log'):
            # 特殊初始化A_log参数
            nn.init.constant_(m.A_log, math.log(0.5))  # 初始衰减率为0.5
        elif hasattr(m, 'D'):
            # 初始化D参数
            nn.init.constant_(m.D, 0.0)
    
    model.apply(init_weights)
    return model