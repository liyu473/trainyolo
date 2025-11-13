"""
将YOLO11模型导出为ONNX格式
"""
from ultralytics import YOLO
import os

# 加载训练好的模型
model_path = 'runs/train/checkpoint_gpu_train/weights/best.pt'
output_dir = '../xzTechnology (2)/xz_python/xiangzhongTechnology/OnnxModel'

print(f"加载模型: {model_path}")
model = YOLO(model_path)

# 导出为ONNX格式
print("导出为ONNX格式...")
onnx_path = model.export(
    format='onnx',
    imgsz=640,
    simplify=True,
    dynamic=False,
    opset=12
)

print(f"\nONNX模型已导出: {onnx_path}")

# 复制到目标目录
import shutil
os.makedirs(output_dir, exist_ok=True)
target_path = os.path.join(output_dir, 'checkpoint_detection.onnx')
shutil.copy2(onnx_path, target_path)

print(f"模型已复制到: {target_path}")
print("\n模型信息:")
print(f"- 输入尺寸: 640x640")
print(f"- 类别数量: 1 (检查点)")
print(f"- 模型性能: mAP50=99.5%, mAP50-95=75.1%")


