import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler

# 设置全局device变量
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {DEVICE}')


# 设置随机种子以确保可重复性
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

set_seed()

# 皮肤癌数据集类
class SkinCancerDataset(Dataset):
    def __init__(self, metadata_path, image_dir, transform=None):
        """
        初始化皮肤癌数据集
        
        Args:
            metadata_path: metadata.csv文件路径
            image_dir: 图像文件夹路径
            transform: 图像变换
        """
        self.metadata = pd.read_csv(metadata_path)
        self.image_dir = image_dir
        self.transform = transform
        
        # 类别映射（7个类别）
        self.classes = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.idx_to_class = {i: cls for cls, i in self.class_to_idx.items()}
        
        # 过滤metadata中不存在的图像
        self.valid_entries = []
        for idx, row in self.metadata.iterrows():
            img_path = os.path.join(self.image_dir, f'{row["image_id"]}.jpg')
            if os.path.exists(img_path):
                self.valid_entries.append(idx)
        
        self.metadata = self.metadata.iloc[self.valid_entries].reset_index(drop=True)
        print(f"Dataset initialized with {len(self.metadata)} valid samples")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        """
        row = self.metadata.iloc[idx]
        img_path = os.path.join(self.image_dir, f'{row["image_id"]}.jpg')
        
        # 读取图像
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 返回一个空白图像作为替代
            image = Image.new('RGB', (600, 450), color='black')
        
        # 获取标签
        label = self.class_to_idx.get(row['dx'], -1)
        if label == -1:
            print(f"Unknown class: {row['dx']}")
            label = 0  # 默认为第一个类别
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, label

# 创建数据变换
def get_transforms(img_size=224):
    """
    创建训练和验证/测试的图像变换
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_test_transform

# 创建数据加载器
def create_dataloaders(metadata_path, image_dir, batch_size=32, val_split=0.2, test_split=0.1):
    """
    创建训练、验证和测试数据加载器
    """
    # 获取变换
    train_transform, val_test_transform = get_transforms()
    
    # 创建完整数据集
    full_dataset = SkinCancerDataset(metadata_path, image_dir, transform=None)
    
    # 分割数据集
    total_size = len(full_dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size - test_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # 应用相应的变换
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_test_transform
    test_dataset.dataset.transform = val_test_transform
    
    # 创建数据加载器
    pin_memory = DEVICE == 'cuda'
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=pin_memory
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader, full_dataset

# 创建模型架构
def create_model(model_name='resnet50', num_classes=7, pretrained=True):
    """
    创建预训练的分类模型
    Args:
        model_name: 模型名称 ('resnet50', 'densenet121', 'efficientnet_b0')
        num_classes: 类别数量
        pretrained: 是否使用预训练权重
    Returns:
        配置好的模型
    """
    if model_name == 'resnet50':
        if pretrained:
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            model = models.resnet50(weights=None)
        # 替换最后的全连接层
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'densenet121':
        if pretrained:
            model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        else:
            model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == 'efficientnet_b0':
        if pretrained:
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        else:
            model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50, use_amp=False):
    """
    训练模型
    Args:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器
        num_epochs: 训练轮数
        device: 训练设备
    Returns:
        训练后的模型和训练历史
    """
    model.to(DEVICE)
    
    # 记录训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # 混合精度训练
    scaler = GradScaler() if use_amp and DEVICE == 'cuda' else None
    
    best_val_acc = 0.0
    best_model_weights = None
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        for images, labels in tqdm(train_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            with torch.set_grad_enabled(True):
                if use_amp and DEVICE == 'cuda':
                    with autocast():
                        outputs = model(images)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                    
                    # 反向传播和优化（混合精度）
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # 反向传播和优化
                    loss.backward()
                    optimizer.step()
            
            # 统计
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += images.size(0)
        
        # 计算训练指标
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())
        
        # 验证阶段
        val_loss, val_acc = evaluate_model(model, val_loader, criterion)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 更新学习率
        if scheduler is not None:
            scheduler.step(val_loss)
        
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = model.state_dict()
            print(f'Best model updated: Val Acc = {best_val_acc:.4f}')
    
    # 加载最佳模型权重
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
    
    print(f'Best validation accuracy: {best_val_acc:.4f}')
    return model, history

# 评估函数
def evaluate_model(model, dataloader, criterion):
    """
    评估模型
    
    Args:
        model: 要评估的模型
        dataloader: 数据加载器
        criterion: 损失函数
        device: 评估设备
    
    Returns:
        损失值和准确率
    """
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # 前向传播
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            # 统计
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += images.size(0)
    
    # 计算指标
    loss = running_loss / total_samples
    acc = running_corrects.double() / total_samples
    
    return loss, acc.item()

# 测试函数并生成详细报告
def test_model(model, test_loader, class_names=None):
    """
    测试模型并生成详细评估报告
    Args:
        model: 要测试的模型
        test_loader: 测试数据加载器
        device: 测试设备
        class_names: 类别名称列表
    Returns:
        评估报告和混淆矩阵
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # 前向传播
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            # 收集预测结果和真实标签
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 生成分类报告
    report = classification_report(all_labels, all_preds, target_names=class_names)
    
    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    print('\nTest Results:')
    print(report)
    
    return report, cm

# 可视化训练历史
def plot_training_history(history, save_path='training_history.png'):
    """
    绘制训练历史曲线
    Args:
        history: 训练历史字典
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制损失曲线
    axes[0].plot(history['train_loss'], label='Training Loss')
    axes[0].plot(history['val_loss'], label='Validation Loss')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    
    # 绘制准确率曲线
    axes[1].plot(history['train_acc'], label='Training Accuracy')
    axes[1].plot(history['val_acc'], label='Validation Accuracy')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f'Training history plot saved to {save_path}')

# 可视化混淆矩阵
def plot_confusion_matrix(cm, class_names, save_path='confusion_matrix.png'):
    """
    绘制混淆矩阵
    
    Args:
        cm: 混淆矩阵
        class_names: 类别名称列表
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()
    print(f'Confusion matrix plot saved to {save_path}')

# 主函数
def main():
    """
    主函数，执行完整的训练和评估流程
    """
    # 设置随机种子，确保实验可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 数据集路径 - 使用相对路径以确保跨环境兼容性
    base_dir = os.path.dirname(os.path.abspath(__file__))
    metadata_path = os.path.join(base_dir, 'dataset', 'HAM10000_metadata.csv')
    image_dir = os.path.join(base_dir, 'dataset', 'Skin Cancer')
    
    # 类别映射
    class_mapping = {
        'nv': 0,    # 黑色素细胞痣
        'mel': 1,   # 黑色素瘤
        'bkl': 2,   # 良性角化病变
        'bcc': 3,   # 基底细胞癌
        'akiec': 4, # 光化性角化病
        'vasc': 5,  # 血管病变
        'df': 6     # 皮肤纤维瘤
    }
    
    # 类别名称
    class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
    
    # 已在全局设置DEVICE变量
    
    # 创建数据集和数据加载器
    print('Loading data...')
    train_loader, val_loader, test_loader, full_dataset = create_dataloaders(
        metadata_path=metadata_path,
        image_dir=image_dir,
        batch_size=32,
        val_split=0.15,
        test_split=0.15
    )
    
    print(f'Train samples: {len(train_loader.dataset)}')
    print(f'Validation samples: {len(val_loader.dataset)}')
    print(f'Test samples: {len(test_loader.dataset)}')
    
    # 创建模型
    print('Creating model...')
    model = create_model(model_name='resnet50', num_classes=7)
    
    # 损失函数（考虑类别不平衡）
    criterion = nn.CrossEntropyLoss()
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # 训练模型
    print('Starting training...')
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=50,
        use_amp=True  # 启用混合精度训练
    )
    
    # 保存模型
    torch.save(model.state_dict(), 'model/skin_cancer_model.pth')
    print('Model saved to model/skin_cancer_model.pth')
    
    # 可视化训练历史
    plot_training_history(history)
    
    # 测试模型
    print('Testing model...')
    report, cm = test_model(model, test_loader, class_names=class_names)
    
    # 可视化混淆矩阵
    plot_confusion_matrix(cm, class_names)
    
    print('Training and evaluation completed!')

# 运行主函数
if __name__ == '__main__':
    main()