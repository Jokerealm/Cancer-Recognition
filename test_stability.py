import torch
import numpy as np
from model.mamba_model import create_mamba_model

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 创建简化的测试数据
batch_size = 2
image_size = 224
images = torch.randn(batch_size, 3, image_size, image_size)
labels = torch.randint(0, 7, (batch_size,))

print("Creating Mamba model...")
model = create_mamba_model(num_classes=7, image_size=image_size)

# 检查模型初始化是否正常
for name, param in model.named_parameters():
    if torch.isnan(param).any():
        print(f"Warning: NaN in parameter {name}")

print("Testing forward pass...")
with torch.no_grad():
    outputs = model(images)
    if torch.isnan(outputs).any():
        print("ERROR: NaN detected in forward pass!")
    else:
        print(f"Forward pass successful! Output shape: {outputs.shape}")
        print(f"Output stats - min: {outputs.min().item():.4f}, max: {outputs.max().item():.4f}, mean: {outputs.mean().item():.4f}")

print("Testing backward pass with gradient clipping...")
# 初始化优化器和损失函数
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

# 进行前向和反向传播
optimizer.zero_grad()
outputs = model(images)
loss = criterion(outputs, labels)

if torch.isnan(loss):
    print("ERROR: NaN loss detected!")
else:
    print(f"Initial loss: {loss.item():.4f}")
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    grad_norm = 0
    has_nan_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"Warning: NaN in gradient of {name}")
                has_nan_grad = True
            grad_norm += param.grad.norm().item() ** 2
    grad_norm = np.sqrt(grad_norm)
    print(f"Gradient norm before clipping: {grad_norm:.4f}")
    
    # 应用梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # 再次检查梯度
    clipped_grad_norm = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            clipped_grad_norm += param.grad.norm().item() ** 2
    clipped_grad_norm = np.sqrt(clipped_grad_norm)
    print(f"Gradient norm after clipping: {clipped_grad_norm:.4f}")
    
    # 更新参数
    optimizer.step()
    print("Parameter update completed successfully!")

print("\nStability test completed.")