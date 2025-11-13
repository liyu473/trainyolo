# YOLO11 训练环境

这是一个用于训练YOLO11目标检测模型的完整环境，支持从Label Studio标注工具导出数据并进行训练。

## 项目结构

```
yoloForXz/
├── datasets/              # 数据集目录
│   ├── images/           # 图像文件
│   │   ├── train/       # 训练集图像
│   │   └── val/         # 验证集图像
│   ├── labels/          # 标注文件
│   │   ├── train/       # 训练集标注
│   │   └── val/         # 验证集标注
│   └── data.yaml        # 数据集配置文件
├── models/               # 自定义模型配置
├── weights/              # 预训练权重和保存的模型
├── runs/                 # 训练/验证/预测结果
├── utils/                # 工具脚本
│   └── convert_labelstudio.py  # Label Studio数据转换脚本
├── train.py              # 训练脚本
├── validate.py           # 验证脚本
├── predict.py            # 预测脚本
├── requirements.txt      # Python依赖
└── README.md             # 说明文档
```

## 安装环境

### 1. 安装Python依赖

```bash
pip install -r requirementsCpu.txt
```

或者

```bash
pip install -r requirementsGpu.txt
```

**注意**: 
- 需要Python 3.8或更高版本
- 如果有NVIDIA GPU，请确保安装了CUDA和cuDNN
- PyTorch会自动安装对应的CUDA版本

### 2. 下载预训练模型（可选）

YOLO11提供多个规模的预训练模型：

- `yolo11n.pt` - Nano (最小最快)
- `yolo11s.pt` - Small
- `yolo11m.pt` - Medium
- `yolo11l.pt` - Large
- `yolo11x.pt` - XLarge (最大最准)

首次运行训练脚本时，Ultralytics会自动下载所需的预训练模型。

## 使用流程

### 第一步: 在Label Studio中标注数据

1. 在Label Studio中创建项目并标注图像
2. 使用矩形框标注工具(Rectangle Labels)进行目标检测标注
3. 完成标注后，导出数据为JSON格式 (Export -> JSON)

### 第二步: 转换Label Studio数据为YOLO格式

使用提供的转换脚本:

```bash
python utils/convert_labelstudio.py --json path/to/labelstudio_export.json --output datasets --train-split 0.8
```

参数说明:
- `--json`: Label Studio导出的JSON文件路径
- `--output`: 输出目录（默认: datasets）
- `--train-split`: 训练集比例（默认: 0.8，即80%训练集，20%验证集）
- `--no-copy-images`: 如果图像已经在正确位置，可以不复制图像

转换完成后，会自动生成:
- 图像文件到 `datasets/images/train` 和 `datasets/images/val`
- 标注文件到 `datasets/labels/train` 和 `datasets/labels/val`
- 数据集配置文件 `datasets/data.yaml`

### 第三步: 修改数据集配置（如需要）

编辑 `datasets/data.yaml` 文件，确认类别名称和路径正确:

```yaml
path: D:/company/py/yoloForXz/datasets  # 数据集根目录
train: images/train
val: images/val

nc: 2  # 类别数量
names: ['cat', 'dog']  # 类别名称
```

### 第四步: 开始训练

使用基础参数训练:

```bash
python train.py --data datasets/data.yaml --epochs 100 --batch 16 --imgsz 640
```

完整参数示例:

```bash
python train.py \
    --data datasets/data.yaml \
    --epochs 100 \
    --batch 16 \
    --imgsz 640 \
    --device 0 \
    --pretrained yolo11n.pt \
    --name my_experiment \
    --project runs/train \
    --patience 50 \
    --optimizer auto \
    --lr0 0.01
```

主要参数说明:
- `--data`: 数据集配置文件路径
- `--epochs`: 训练轮数
- `--batch`: 批次大小（根据GPU内存调整）
- `--imgsz`: 输入图像大小（640, 1280等）
- `--device`: GPU设备号（0, 1, 2等）或 'cpu'
- `--pretrained`: 预训练模型（yolo11n.pt, yolo11s.pt等）
- `--name`: 实验名称
- `--workers`: 数据加载线程数（默认8）
- `--patience`: 早停耐心值
- `--optimizer`: 优化器（SGD, Adam, AdamW, auto）

### 第五步: 验证模型

训练完成后验证模型性能:

```bash
python validate.py --model runs/train/my_experiment/weights/best.pt --data datasets/data.yaml
```

参数说明:
- `--model`: 模型权重文件路径
- `--data`: 数据集配置文件
- `--imgsz`: 输入图像大小
- `--batch`: 批次大小
- `--conf`: 置信度阈值（默认0.001）
- `--iou`: IOU阈值（默认0.6）

### 第六步: 使用模型进行预测

对图像进行预测:

```bash
python predict.py --model runs/train/my_experiment/weights/best.pt --source path/to/images
```

对视频进行预测:

```bash
python predict.py --model runs/train/my_experiment/weights/best.pt --source path/to/video.mp4
```

使用摄像头实时检测:

```bash
python predict.py --model runs/train/my_experiment/weights/best.pt --source 0 --show
```

参数说明:
- `--model`: 模型权重文件路径
- `--source`: 输入源（图像/文件夹/视频/摄像头）
- `--imgsz`: 输入图像大小
- `--conf`: 置信度阈值（默认0.25）
- `--iou`: IOU阈值（默认0.45）
- `--save`: 保存结果图像
- `--save-txt`: 保存结果为txt
- `--show`: 显示结果
- `--classes`: 过滤特定类别

## 训练技巧

### 1. 选择合适的模型规模

- **yolo11n** (Nano): 适合边缘设备，速度最快，精度较低
- **yolo11s** (Small): 平衡速度和精度
- **yolo11m** (Medium): 推荐用于大多数应用
- **yolo11l** (Large): 高精度应用
- **yolo11x** (XLarge): 最高精度，需要强大GPU

### 2. 调整批次大小

根据GPU内存调整批次大小:
- RTX 3060 (12GB): batch=16-32
- RTX 3080 (10GB): batch=16-24
- RTX 3090 (24GB): batch=32-64
- GTX 1660 (6GB): batch=8-16

如果遇到内存不足，减小批次大小或图像大小。

### 3. 数据增强

YOLO11自动使用多种数据增强技术:
- Mosaic拼接
- MixUp混合
- 随机翻转、缩放、旋转
- HSV色彩增强

### 4. 恢复训练

如果训练中断，可以恢复:

```bash
python train.py --resume runs/train/my_experiment/weights/last.pt
```

### 5. 超参数调优

可以调整的关键超参数:
- `--lr0`: 初始学习率（默认0.01）
- `--lrf`: 最终学习率比率（默认0.01）
- `--momentum`: SGD动量（默认0.937）
- `--weight-decay`: 权重衰减（默认0.0005）
- `--warmup-epochs`: 预热轮数（默认3）
- `--cos-lr`: 使用余弦学习率调度

## 常见问题

### 1. CUDA内存不足

解决方法:
- 减小批次大小: `--batch 8` 或 `--batch 4`
- 减小图像大小: `--imgsz 416` 或 `--imgsz 320`
- 使用更小的模型: `yolo11n.pt` 代替 `yolo11l.pt`

### 2. 训练速度慢

优化方法:
- 增加workers数量: `--workers 16`
- 启用图像缓存: `--cache ram` (如果内存足够)
- 使用更小的图像尺寸

### 3. 模型过拟合

改进方法:
- 增加训练数据
- 启用早停: `--patience 50`
- 增加数据增强
- 使用正则化: 调整 `--weight-decay`

### 4. 检测效果不好

改进方法:
- 检查标注质量
- 增加训练轮数
- 使用更大的模型
- 调整置信度阈值: `--conf 0.3`
- 增加训练数据

## 模型导出

### 导出为ONNX格式

```python
from ultralytics import YOLO

model = YOLO('runs/train/my_experiment/weights/best.pt')
model.export(format='onnx')
```

### 支持的导出格式

- ONNX
- TensorRT
- CoreML
- TFLite
- OpenVINO

## 参考资料

- [Ultralytics文档](https://docs.ultralytics.com/)
- [YOLO11官方仓库](https://github.com/ultralytics/ultralytics)
- [Label Studio文档](https://labelstud.io/guide/)

## 许可证

本项目遵循MIT许可证。

## 联系方式

如有问题，请提交Issue或联系项目维护者。

如何一次性把「GPU 版 torch」写进依赖文件
方法 A：在 requirements.txt 里指定 CUDA 源
Text
复制
--extra-index-url https://download.pytorch.org/whl/cu121
torch==2.5.1+cu121
torchvision==0.20.1+cu121
其余包照常列即可。
方法 B：分两条命令（推荐）
bash
复制
# 先装 GPU 版 torch 三件套
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# 再装剩余依赖
pip install -r requirements.txt
这样不会重复卸载/重装，速度最快。








