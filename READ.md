# 🛠️ 基于 PyTorch 的 CWRU 轴承故障诊断基准框架
**(CWRU Bearing Fault Diagnosis Benchmark with PyTorch)**

本项目是一个基于 PyTorch 的深度学习框架，专为 CWRU (凯斯西储大学) 轴承数据集设计。它不仅仅是一个简单的分类项目，而是构建了一个**多任务学习 (Multi-Task Learning)** 系统，能够同时进行故障检测、故障类型分类以及故障尺寸的量化预测（回归）。

项目中包含 **16 种主流深度学习模型**的 1D 改版实现，并集成了 Optuna 自动化调参和 Bokeh 交互式可视化，适合作为工业故障诊断领域的基准测试 (Benchmark) 工具。

## ✨ 核心特性 (Key Features)

### 🧠 多任务学习架构 (Multi-Task Learning Head)
每个模型都通过共享的主干网络 (Backbone) 提取特征，并同时输出 4 个预测结果：
* **Fault Check (Binary):** 二分类，判断正常/故障。
* **Fault Type (Multi-class):** 四分类，判断故障位置 (Normal, Inner Race, Ball, Outer Race)。
* **Fault Size Class (Multi-class):** 四分类，判断损伤等级 (0, 0.007, 0.014, 0.021)。
* **Fault Size Regression (Regression):** 回归预测，直接输出损伤直径数值 (mm)。

### 🏗️ 庞大的 1D 模型库 (Model Zoo)
本项目复现并调整了以下模型以适应一维振动信号处理：
* **基础模型:** MLP, 1D-CNN
* **RNN系列:** LSTM, GRU, BiLSTM
* **Transformer系列:** 1D-Transformer (Encoder-only)
* **现代CNN架构:** ResNet-1D, DenseNet-1D, MobileNet-1D, EfficientNet-1D, Xception-1D, Inception-1D, TinyVGG-1D
* **高级/创新模型:** DeepCNN, MEAT (Multi-scale Hierarchical Attention), **DenseResNet1D** (结合 Dense, Residual 和 SE-Block 的新架构)。

### ⚙️ 自动化与工程化
* **环境自适应:** 自动检测运行环境 (Local/Kaggle/Colab) 并调整数据路径。
* **Optuna 调参:** 集成 Optuna 框架，自动搜索最佳的学习率 (LR) 和 Batch Size。
* **数据增强与平衡:** 滑动窗口切片 (Sliding Window Segmentation) 与数据重采样 (Resampling)，支持类别平衡 (Balancing)。

### 📊 全面的评估与可视化
* **指标:** Accuracy, MSE, MAE, R2 Score, MAPE, Confusion Matrix。
* **可视化:** 使用 Bokeh 生成交互式的训练曲线（Loss, Accuracy, Time）对比图，以及 Matplotlib 绘制混淆矩阵和回归散点图。

---

## 📂 数据集准备 (Dataset)

本项目使用标准的 **CWRU Bearing Dataset**。代码支持从 Kaggle Hub 自动下载或读取本地目录。

**数据预处理流程：**
1.  **加载:** 读取 `.mat` 文件 (Drive End & Fan End)。
2.  **重采样:** 统一采样率至 12kHz。
3.  **切片:** 使用重叠滑动窗口将信号切分为固定长度 (默认 1024 点)。
4.  **归一化:** 使用 StandardScaler 标准化输入信号，使用 MinMaxScaler 归一化回归目标值。

---

## 🚀 快速开始 (Quick Start)

### 1. 安装依赖
请确保安装了以下 Python 库：
```
pip install -r requirements.txt
```

### 2. 运行项目
在代码的 Setup 部分，修改 data_dir 指向你存放 CWRU .mat 文件的文件夹路径：
# 示例
data_dir = './data/cwru' 
save_dir = './output'

### 3. 切换模型
在训练部分，你可以通过修改 model_name 来切换不同的模型进行训练：
# 例如切换为 Transformer
model_name = 'Transformer1D_pt'
train_loader, val_loader, test_loader = generate_dataloaders_pt(
    model_name, X_train, y_train, ...
)

---

## 📊 实验结果示例 (Results)
模型训练完成后，会自动在 ./output/PyTorch/metrics/ 目录下生成详细的 CSV 报告。

---

## 📁 目录结构 (Directory Structure)
.
├── dl-pytorch-regression.ipynb    # 核心代码
├── requirements.txt               # 依赖列表
├── README.md                      # 说明文档
├── data/                          # 数据集目录 (需自行放置或自动下载)
└── output/                        # 输出目录 (自动生成)
    └── PyTorch/
        ├── dict_models/           # 保存的模型权重 (.pth)
        ├── full_models/           # 保存的完整模型 (.pt)
        └── metrics/               # 训练日志与测试结果 (.csv)
