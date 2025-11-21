# 皮肤癌分类项目

## 项目概述
这是一个三阶段皮肤癌分类系统，使用深度学习模型对皮肤病变进行分类。系统分为三个阶段：
1. 第一阶段：良性/恶性二分类
2. 第二阶段：恶性子类分类（akiec, bcc, mel）
3. 第三阶段：良性子类分类（df, vasc, bkl）

## 环境要求

### 安装依赖
```bash
pip install -r requirements.txt
```

### 主要依赖
- PyTorch 1.13.0+
- torchvision 0.14.0+
- scikit-learn 1.2.0+
- pandas 1.5.0+
- NumPy 1.22.0+
- Pillow 9.3.0+
- tqdm 4.64.0+

## 数据准备

### 数据集结构
请确保数据集按照以下结构组织：
```
Skin cancer/
├── dataset/
│   └── Skin Cancer/
│       ├── akiec/
│       ├── bcc/
│       ├── bkl/
│       ├── df/
│       ├── mel/
│       └── vasc/
├── main.py
├── requirements.txt
└── README.md
```

## 训练模型

### 开始训练
在项目根目录下运行以下命令开始三阶段训练：

```bash
python main.py
```

### 训练参数说明
- 训练完成后，模型权重将保存在当前目录：
  - `best_stage1.pth`: 第一阶段模型（良性/恶性分类）
  - `best_stage2_malignant.pth`: 第二阶段模型（恶性子类）
  - `best_stage3_benign.pth`: 第三阶段模型（良性子类）
- 错误分类的图片将保存在 `error_images` 目录中

## 预测单张图片

要使用训练好的模型预测单张图片，可以修改 `main.py` 中的代码，调用 `predict_image` 函数：

```python
# 示例代码
from main import load_models, predict_image

# 加载模型
model_stage1, model_stage2, model_stage3, device = load_models()

# 预测图片
result = predict_image("path/to/image.jpg", model_stage1, model_stage2, model_stage3, device)
print(result)
```

## 常见问题

### GPU 支持
- 代码会自动检测并使用可用的GPU
- 如果没有GPU，会默认使用CPU

### 内存问题
- 如需减少内存使用，可以修改 `main.py` 中的 `BATCH_SIZE` 参数
- 对于CPU训练，建议将 `BATCH_SIZE` 设置为较小值（如8或16）

### 模型权重下载
- 代码使用 torchvision 的预训练模型，会在首次运行时自动下载权重

## 注意事项
- 确保数据集包含所有必要的类别文件夹
- 训练时间取决于硬件配置，完整训练可能需要数小时
- 训练过程中会输出详细的性能指标