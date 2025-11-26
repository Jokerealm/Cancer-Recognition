# 导入必要的库
import torch
import numpy as np
from model.mamba_model import create_mamba_model

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 创建一个简单的测试图像批次
batch_size = 2
image_size = 224
images = torch.randn(batch_size, 3, image_size, image_size)

print("初始化Mamba模型...")
model = create_mamba_model(num_classes=7, image_size=image_size)

# 打印模型结构摘要
print(f"模型结构: {model.__class__.__name__}")
print(f"输入图像大小: {image_size}x{image_size}")
print(f"类别数量: 7")
print(f"内部维度: 256")
print(f"深度: 4")

# 计算模型参数数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n总参数数量: {total_params:,}")
print(f"可训练参数数量: {trainable_params:,}")

# 测试前向传播
print("\n测试前向传播...")
with torch.no_grad():
    outputs = model(images)
    
print(f"前向传播输出形状: {outputs.shape}")
print(f"输出统计 - 最小值: {outputs.min().item():.4f}")
print(f"输出统计 - 最大值: {outputs.max().item():.4f}")
print(f"输出统计 - 平均值: {outputs.mean().item():.4f}")
print(f"输出统计 - 标准差: {outputs.std().item():.4f}")

# 检查是否有NaN值
if torch.isnan(outputs).any():
    print("\n警告: 输出中包含NaN值!")
else:
    print("\n输出中没有NaN值，前向传播正常。")

# 简单的损失计算测试
labels = torch.randint(0, 7, (batch_size,))
criterion = torch.nn.CrossEntropyLoss()
with torch.no_grad():
    loss = criterion(outputs, labels)
    print(f"\n示例损失值: {loss.item():.4f}")

print("\nMamba模型测试完成! (稳定版本)")