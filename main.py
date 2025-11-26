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
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler
from model.mamba_model import create_mamba_model
# from model.mamba_model import create_mamba_model

# 设置全局device变量
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {DEVICE}')


# 设置随机种子以确保可重复性
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

set_seed()

# 添加必要的导入
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 皮肤癌数据集类
class SkinCancerDataset(Dataset):
    def __init__(self, metadata_path, image_dir, transform=None, train=True, scaler=None, gender_encoder=None, localization_encoder=None):
        """
        初始化皮肤癌数据集
        
        Args:
            metadata_path: metadata.csv文件路径
            image_dir: 图像文件夹路径
            transform: 图像变换
            train: 是否为训练集（用于确定是否拟合预处理器）
            scaler: 年龄标准化器（训练集时为None，验证/测试集时使用训练集的）
            gender_encoder: 性别编码器（训练集时为None，验证/测试集时使用训练集的）
            localization_encoder: 解剖部位编码器（训练集时为None，验证/测试集时使用训练集的）
        """
        self.metadata = pd.read_csv(metadata_path)
        self.image_dir = image_dir
        self.transform = transform
        self.train = train
        
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
        
        # 预处理表格特征
        self._preprocess_metadata_features(scaler, gender_encoder, localization_encoder)
        
        print(f"Dataset initialized with {len(self.metadata)} valid samples")
        print(f"表格特征维度: {self.metadata_feature_dim}")
    
    def __len__(self):
        return len(self.metadata)
    
    def _preprocess_metadata_features(self, scaler=None, gender_encoder=None, localization_encoder=None):
        """
        预处理表格特征
        
        Args:
            scaler: 年龄标准化器
            gender_encoder: 性别编码器
            localization_encoder: 解剖部位编码器
        """
        # 复制数据以避免修改原始数据
        processed = self.metadata.copy()
        
        # 处理年龄缺失值
        age_mean = processed['age'].mean()
        processed['age'] = processed['age'].fillna(age_mean)
        
        # 处理性别缺失值
        processed['sex'] = processed['sex'].fillna('unknown')
        
        # 处理解剖部位缺失值
        processed['localization'] = processed['localization'].fillna('unknown')
        
        # 性别编码
        if self.train:
            self.gender_encoder = LabelEncoder()
            gender_encoded = self.gender_encoder.fit_transform(processed['sex'])
        else:
            self.gender_encoder = gender_encoder
            # 使用编码器转换，如果遇到未知类别则设为0
            gender_encoded = []
            for sex in processed['sex']:
                try:
                    gender_encoded.append(self.gender_encoder.transform([sex])[0])
                except ValueError:
                    gender_encoded.append(0)
            gender_encoded = np.array(gender_encoded)
        
        # 解剖部位编码
        if self.train:
            self.localization_encoder = LabelEncoder()
            localization_encoded = self.localization_encoder.fit_transform(processed['localization'])
        else:
            self.localization_encoder = localization_encoder
            # 使用编码器转换，如果遇到未知类别则设为0
            localization_encoded = []
            for loc in processed['localization']:
                try:
                    localization_encoded.append(self.localization_encoder.transform([loc])[0])
                except ValueError:
                    localization_encoded.append(0)
            localization_encoded = np.array(localization_encoded)
        
        # 年龄标准化
        if self.train:
            self.scaler = StandardScaler()
            age_scaled = self.scaler.fit_transform(processed[['age']])
        else:
            self.scaler = scaler
            age_scaled = self.scaler.transform(processed[['age']])
        
        # 创建特征矩阵
        # 年龄(1维) + 性别(1维) + 解剖部位(1维) = 3维特征
        self.metadata_features = np.hstack([
            age_scaled,
            gender_encoded.reshape(-1, 1),
            localization_encoded.reshape(-1, 1)
        ])
        
        # 记录特征维度
        self.metadata_feature_dim = self.metadata_features.shape[1]
    
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
        
        # 获取表格特征
        metadata_features = torch.tensor(self.metadata_features[idx], dtype=torch.float)
        
        return image, metadata_features, label

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
    
    # 首先创建完整数据集以获取索引
    temp_dataset = SkinCancerDataset(metadata_path, image_dir, transform=None)
    
    # 分割数据集索引
    total_size = len(temp_dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size - test_size
    
    # 使用固定的随机种子进行分割以确保可重复性
    torch.manual_seed(42)
    # 创建索引列表
    indices = list(range(total_size))
    # 随机打乱索引
    np.random.shuffle(indices)
    # 分割索引
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    # 创建训练数据集（会拟合预处理器）
    train_dataset = SkinCancerDataset(
        metadata_path, 
        image_dir, 
        transform=train_transform,
        train=True
    )
    # 只保留训练集的样本
    train_dataset.metadata = train_dataset.metadata.iloc[train_indices].reset_index(drop=True)
    train_dataset.metadata_features = train_dataset.metadata_features[train_indices]
    
    # 创建验证数据集（使用训练集的预处理器）
    val_dataset = SkinCancerDataset(
        metadata_path, 
        image_dir, 
        transform=val_test_transform,
        train=False,
        scaler=train_dataset.scaler,
        gender_encoder=train_dataset.gender_encoder,
        localization_encoder=train_dataset.localization_encoder
    )
    # 只保留验证集的样本
    val_dataset.metadata = val_dataset.metadata.iloc[val_indices].reset_index(drop=True)
    val_dataset.metadata_features = val_dataset.metadata_features[val_indices]
    
    # 创建测试数据集（使用训练集的预处理器）
    test_dataset = SkinCancerDataset(
        metadata_path, 
        image_dir, 
        transform=val_test_transform,
        train=False,
        scaler=train_dataset.scaler,
        gender_encoder=train_dataset.gender_encoder,
        localization_encoder=train_dataset.localization_encoder
    )
    # 只保留测试集的样本
    test_dataset.metadata = test_dataset.metadata.iloc[test_indices].reset_index(drop=True)
    test_dataset.metadata_features = test_dataset.metadata_features[test_indices]
    
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
    return train_loader, val_loader, test_loader, train_dataset

# 多模态融合模型
class MultimodalFusionModel(nn.Module):
    def __init__(self, image_model, metadata_feature_dim, num_classes=7, fusion_type='concat'):
        """
        多模态融合模型，整合图像特征和表格特征
        
        Args:
            image_model: 图像特征提取模型
            metadata_feature_dim: 表格特征维度
            num_classes: 类别数量
            fusion_type: 融合类型 ('concat', 'attention', 'gated')
        """
        super(MultimodalFusionModel, self).__init__()
        self.image_model = image_model
        self.metadata_feature_dim = metadata_feature_dim
        self.fusion_type = fusion_type
        
        # 获取图像特征维度
        if hasattr(image_model, 'fc'):  # ResNet系列
            self.image_feature_dim = image_model.fc.in_features
            # 移除原始的全连接层
            self.image_model.fc = nn.Identity()
        elif hasattr(image_model, 'classifier') and isinstance(image_model.classifier, nn.Linear):  # DenseNet系列
            self.image_feature_dim = image_model.classifier.in_features
            # 移除原始的分类器
            self.image_model.classifier = nn.Identity()
        elif hasattr(image_model, 'classifier') and isinstance(image_model.classifier, nn.Sequential):  # EfficientNet系列
            self.image_feature_dim = image_model.classifier[1].in_features
            # 移除原始的分类器
            self.image_model.classifier = nn.Identity()
        elif hasattr(image_model, 'heads') and hasattr(image_model.heads, 'head'):  # ViT系列
            self.image_feature_dim = image_model.heads.head.in_features
            # 移除原始的分类头
            self.image_model.heads.head = nn.Identity()
        else:
            # 假设Mamba模型有一个固定的特征维度（需要根据实际实现调整）
            self.image_feature_dim = 1024  # 临时值
        
        # 表格特征处理层
        self.metadata_fc = nn.Sequential(
            nn.Linear(metadata_feature_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
        # 根据融合类型设置融合层
        if fusion_type == 'concat':
            # 简单连接特征
            combined_dim = self.image_feature_dim + 128
            self.fusion_layer = nn.Identity()
        elif fusion_type == 'attention':
            # 注意力机制融合
            self.attention_layer = nn.Sequential(
                nn.Linear(self.image_feature_dim + 128, self.image_feature_dim + 128),
                nn.Softmax(dim=1)
            )
            combined_dim = self.image_feature_dim + 128
        elif fusion_type == 'gated':
            # 门控机制融合
            self.gate_layer = nn.Sequential(
                nn.Linear(self.image_feature_dim + 128, 1),
                nn.Sigmoid()
            )
            combined_dim = self.image_feature_dim
        else:
            raise ValueError(f"Unsupported fusion type: {fusion_type}")
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, images, metadata_features):
        # 提取图像特征
        image_features = self.image_model(images)
        
        # 处理表格特征
        metadata_features = self.metadata_fc(metadata_features)
        
        # 融合特征
        if self.fusion_type == 'concat':
            combined_features = torch.cat([image_features, metadata_features], dim=1)
        elif self.fusion_type == 'attention':
            combined = torch.cat([image_features, metadata_features], dim=1)
            attention_weights = self.attention_layer(combined)
            combined_features = combined * attention_weights
        elif self.fusion_type == 'gated':
            combined = torch.cat([image_features, metadata_features], dim=1)
            gate = self.gate_layer(combined)
            # 使用门控权重控制图像特征的重要性
            combined_features = image_features * gate + metadata_features * (1 - gate)
        
        # 分类
        outputs = self.classifier(combined_features)
        return outputs

# 创建模型架构
def create_model(model_name='resnet50', num_classes=7, pretrained=True, metadata_feature_dim=None, fusion_type='concat', source='remote'):
    """
    创建预训练的分类模型或多模态融合模型
    Args:
        model_name: 模型名称 ('resnet50', 'densenet121', 'efficientnet_b0', 'vit_b_16', 'mamba')
        num_classes: 类别数量
        pretrained: 是否使用预训练权重（不适用于Mamba模型）
        metadata_feature_dim: 表格特征维度，如果提供则创建多模态融合模型
        fusion_type: 融合类型 ('concat', 'attention', 'gated')
        source: 预训练权重来源 ('remote' 或 'local')
    Returns:
        配置好的模型
    """
    # 创建图像特征提取模型
    if model_name == 'resnet50':
        if pretrained:
            if source == 'remote':
                image_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            else:
                image_model = models.resnet50(weights=None)
                # 尝试从本地加载预训练权重
                weights_path = os.path.join('~/.cache/torch/hub/checkpoints', f'{model_name}.pth')
                if os.path.exists(weights_path):
                    try:
                        print(f"Loading local weights from {weights_path}")
                        pretrained_weights = torch.load(weights_path)
                        # 加载权重，但跳过不匹配的键
                        model_dict = image_model.state_dict()
                        pretrained_dict = {k: v for k, v in pretrained_weights.items() if k in model_dict}
                        model_dict.update(pretrained_dict)
                        image_model.load_state_dict(model_dict)
                        print(f"Successfully loaded {len(pretrained_dict)} parameters from {weights_path}")
                    except Exception as e:
                        print(f"Error loading weights: {e}")
                        print("Using default initialization instead.")
                else:
                    print(f"Local weights file not found: {weights_path}")
                    print("Using default initialization instead.")
        else:
            image_model = models.resnet50(weights=None)
    elif model_name == 'resnet18':
        if pretrained:
            if source == 'remote':
                image_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            else:
                image_model = models.resnet18(weights=None)
                # 尝试从本地加载预训练权重
                weights_path = os.path.join('~/.cache/torch/hub/checkpoints', f'{model_name}.pth')
                if os.path.exists(weights_path):
                    try:
                        print(f"Loading local weights from {weights_path}")
                        pretrained_weights = torch.load(weights_path)
                        # 加载权重，但跳过不匹配的键
                        model_dict = image_model.state_dict()
                        pretrained_dict = {k: v for k, v in pretrained_weights.items() if k in model_dict}
                        model_dict.update(pretrained_dict)
                        image_model.load_state_dict(model_dict)
                        print(f"Successfully loaded {len(pretrained_dict)} parameters from {weights_path}")
                    except Exception as e:
                        print(f"Error loading weights: {e}")
                        print("Using default initialization instead.")
                else:
                    print(f"Local weights file not found: {weights_path}")
                    print("Using default initialization instead.")
        else:
            image_model = models.resnet18(weights=None)
    elif model_name == 'densenet121':
        if pretrained:
            if source == 'remote':
                image_model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            else:
                image_model = models.densenet121(weights=None)
                # 尝试从本地加载预训练权重
                weights_path = os.path.join('~/.cache/torch/hub/checkpoints', f'{model_name}.pth')
                if os.path.exists(weights_path):
                    try:
                        print(f"Loading local weights from {weights_path}")
                        pretrained_weights = torch.load(weights_path)
                        # 加载权重，但跳过不匹配的键
                        model_dict = image_model.state_dict()
                        pretrained_dict = {k: v for k, v in pretrained_weights.items() if k in model_dict}
                        model_dict.update(pretrained_dict)
                        image_model.load_state_dict(model_dict)
                        print(f"Successfully loaded {len(pretrained_dict)} parameters from {weights_path}")
                    except Exception as e:
                        print(f"Error loading weights: {e}")
                        print("Using default initialization instead.")
                else:
                    print(f"Local weights file not found: {weights_path}")
                    print("Using default initialization instead.")
        else:
            image_model = models.densenet121(weights=None)
    elif model_name == 'efficientnet_b0':
        if pretrained:
            if source == 'remote':
                image_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            else:
                image_model = models.efficientnet_b0(weights=None)
                # 尝试从本地加载预训练权重
                weights_path = os.path.join('~/.cache/torch/hub/checkpoints', f'{model_name}.pth')
                if os.path.exists(weights_path):
                    try:
                        print(f"Loading local weights from {weights_path}")
                        pretrained_weights = torch.load(weights_path)
                        # 加载权重，但跳过不匹配的键
                        model_dict = image_model.state_dict()
                        pretrained_dict = {k: v for k, v in pretrained_weights.items() if k in model_dict}
                        model_dict.update(pretrained_dict)
                        image_model.load_state_dict(model_dict)
                        print(f"Successfully loaded {len(pretrained_dict)} parameters from {weights_path}")
                    except Exception as e:
                        print(f"Error loading weights: {e}")
                        print("Using default initialization instead.")
                else:
                    print(f"Local weights file not found: {weights_path}")
                    print("Using default initialization instead.")
        else:
            image_model = models.efficientnet_b0(weights=None)
    elif model_name == 'vit_b_16':
        # Vision Transformer模型
        if pretrained:
            if source == 'remote':
                image_model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
            else:
                image_model = models.vit_b_16(weights=None)
                # 尝试从本地加载预训练权重
                weights_path = os.path.join('~/.cache/torch/hub/checkpoints', f'{model_name}.pth')
                if os.path.exists(weights_path):
                    try:
                        print(f"Loading local weights from {weights_path}")
                        pretrained_weights = torch.load(weights_path)
                        # 加载权重，但跳过不匹配的键
                        model_dict = image_model.state_dict()
                        pretrained_dict = {k: v for k, v in pretrained_weights.items() if k in model_dict}
                        model_dict.update(pretrained_dict)
                        image_model.load_state_dict(model_dict)
                        print(f"Successfully loaded {len(pretrained_dict)} parameters from {weights_path}")
                    except Exception as e:
                        print(f"Error loading weights: {e}")
                        print("Using default initialization instead.")
                else:
                    print(f"Local weights file not found: {weights_path}")
                    print("Using default initialization instead.")
        else:
            image_model = models.vit_b_16(weights=None)
    elif model_name == 'mamba':
        # 创建Mamba模型
        try:
            image_model = create_mamba_model(num_classes=num_classes)
            print("Created Mamba model for image classification")
        except Exception as e:
            print(f"Error creating Mamba model: {e}")
            # 回退到ResNet模型
            image_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # 如果提供了表格特征维度，则创建多模态融合模型
    if metadata_feature_dim is not None:
        model = MultimodalFusionModel(
            image_model=image_model,
            metadata_feature_dim=metadata_feature_dim,
            num_classes=num_classes,
            fusion_type=fusion_type
        )
        print(f"Created multimodal fusion model with {fusion_type} fusion")
        print(f"Image feature dimension: {model.image_feature_dim}")
        print(f"Metadata feature dimension: {metadata_feature_dim}")
    else:
        # 单模态模型，替换分类层
        if hasattr(image_model, 'fc'):  # ResNet系列
            image_model.fc = nn.Linear(image_model.fc.in_features, num_classes)
        elif hasattr(image_model, 'classifier') and isinstance(image_model.classifier, nn.Linear):  # DenseNet系列
            image_model.classifier = nn.Linear(image_model.classifier.in_features, num_classes)
        elif hasattr(image_model, 'classifier') and isinstance(image_model.classifier, nn.Sequential):  # EfficientNet系列
            image_model.classifier[1] = nn.Linear(image_model.classifier[1].in_features, num_classes)
        elif hasattr(image_model, 'heads') and hasattr(image_model.heads, 'head'):  # ViT系列
            image_model.heads.head = nn.Linear(image_model.heads.head.in_features, num_classes)
        model = image_model
        print(f"Created single-modal model: {model_name}")
    
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
        use_amp: 是否使用混合精度训练
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
        
        for i, batch in enumerate(tqdm(train_loader)):
            # 处理多模态数据
            if len(batch) == 3:  # 多模态数据：(images, metadata_features, labels)
                images, metadata_features, labels = batch
                metadata_features = metadata_features.to(DEVICE)
            else:  # 单模态数据：(images, labels)
                images, labels = batch
                metadata_features = None
            
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            with torch.set_grad_enabled(True):
                if use_amp and DEVICE == 'cuda':
                    with autocast():
                        # 根据输入类型调用模型
                        if metadata_features is not None:
                            outputs = model(images, metadata_features)
                        else:
                            outputs = model(images)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                    
                    # 检查损失是否为NaN
                    if torch.isnan(loss):
                        print(f"NaN loss detected at epoch {epoch}, batch {i}")
                        continue
                    
                    # 反向传播和优化（混合精度）
                    scaler.scale(loss).backward()
                    # 梯度裁剪，防止梯度爆炸
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # 根据输入类型调用模型
                    if metadata_features is not None:
                        outputs = model(images, metadata_features)
                    else:
                        outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # 检查损失是否为NaN
                    if torch.isnan(loss):
                        print(f"NaN loss detected at epoch {epoch}, batch {i}")
                        continue
                    
                    # 反向传播和优化，添加梯度裁剪
                    loss.backward()
                    # 梯度裁剪，防止梯度爆炸
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
    
    Returns:
        损失值和准确率
    """
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # 处理多模态数据
            if len(batch) == 3:  # 多模态数据：(images, metadata_features, labels)
                images, metadata_features, labels = batch
                metadata_features = metadata_features.to(DEVICE)
            else:  # 单模态数据：(images, labels)
                images, labels = batch
                metadata_features = None
            
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # 前向传播
            if metadata_features is not None:
                outputs = model(images, metadata_features)
            else:
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
        class_names: 类别名称列表
    Returns:
        评估报告和混淆矩阵
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            # 处理多模态数据
            if len(batch) == 3:  # 多模态数据：(images, metadata_features, labels)
                images, metadata_features, labels = batch
                metadata_features = metadata_features.to(DEVICE)
            else:  # 单模态数据：(images, labels)
                images, labels = batch
                metadata_features = None
            
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # 前向传播
            if metadata_features is not None:
                outputs = model(images, metadata_features)
            else:
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
    base_dir = '/data16t/'
    metadata_path = os.path.join(base_dir, 'Skin Cancer', 'HAM10000_metadata.csv')
    image_dir = os.path.join(base_dir, 'Skin Cancer', 'Skin Cancer')
    
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
    
    # 创建模型 - 可以选择'resnet50', 'densenet121', 'efficientnet_b0'或'mamba'
    print('Creating model...')
    # 从训练数据集中获取metadata_feature_dim
    # 获取一个批次的数据来确定特征维度
    for batch in train_loader:
        if len(batch) == 3:  # 多模态数据
            _, metadata_features, _ = batch
            metadata_feature_dim = metadata_features.shape[1]
            break
    else:
        # 如果没有找到多模态数据，使用默认值
        metadata_feature_dim = 10  # 默认值，实际应根据数据调整
    
    print(f'Metadata feature dimension: {metadata_feature_dim}')
    
    # 创建多模态融合模型
    # fusion_type可以是 'concat', 'attention', 'gated'
    model = create_model(
        model_name='resnet50',  # 切换为ViT模型
        num_classes=7,
        metadata_feature_dim=metadata_feature_dim,
        fusion_type='attention',
        source='local'  # 使用本地预训练权重
    )
    
    # 损失函数（考虑类别不平衡）
    criterion = nn.CrossEntropyLoss()
    
    # 优化器 - 使用更小的学习率和权重衰减以提高Mamba模型训练稳定性
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
        use_amp=False  # 启用混合精度训练
    )
    
    # 确保checkpoint目录存在
    os.makedirs('checkpoint', exist_ok=True)
    # 保存模型
    torch.save(model.state_dict(), 'checkpoint/skin_cancer_model.pth')
    print('Model saved to checkpoint/skin_cancer_model.pth')
    
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