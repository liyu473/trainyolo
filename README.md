# YOLO11 训练环境

这是一个用于训练YOLO11目标检测模型的简易程序，支持从Label Studio标注工具导出数据并进行训练。

## 安装环境

### 1. 安装Python依赖

如果您的电脑是集显，请安装cpu依赖

```bash
pip install -r requirementsCpu.txt
```

或者您的电脑有高性能显卡，请安装GPU依赖

```bash
pip install -r requirementsGpu.txt
```

**注意**: 
- 需要Python 3.8或更高版本
- 如果有NVIDIA GPU，请确保安装了CUDA和cuDNN
- PyTorch会自动安装对应的CUDA版本

### 2. 下载预训练模型（可选）

YOLO11提供多个规模的预训练模型（程序集内置一个yolo11n.pt）：

- `yolo11n.pt` - Nano (最小最快)
- `yolo11s.pt` - Small
- `yolo11m.pt` - Medium
- `yolo11l.pt` - Large
- `yolo11x.pt` - XLarge (最大最准)

首次运行训练脚本时，Ultralytics会自动下载所需的预训练模型。

## 使用流程

### 参考starttrain_readme.md





