"""
YOLO11 模型验证脚本
用于验证训练好的模型性能
"""
from ultralytics import YOLO
import argparse
import os


def validate_model(args):
    """
    验证YOLO模型
    
    Args:
        args: 命令行参数
    """
    # 加载模型
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"模型文件不存在: {args.model}")
    
    model = YOLO(args.model)
    print(f"加载模型: {args.model}")
    
    # 验证
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        workers=args.workers,
        plots=True,
        save_json=args.save_json,
        save_hybrid=False,
        project=args.project,
        name=args.name,
    )
    
    # 打印结果
    print("\n验证结果:")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='YOLO11模型验证脚本')
    
    parser.add_argument('--model', type=str, required=True,
                        help='模型权重文件路径')
    parser.add_argument('--data', type=str, default='datasets/data.yaml',
                        help='数据集配置文件路径')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='输入图像大小')
    parser.add_argument('--batch', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--conf', type=float, default=0.001,
                        help='目标置信度阈值')
    parser.add_argument('--iou', type=float, default=0.6,
                        help='NMS的IOU阈值')
    parser.add_argument('--device', type=str, default='0',
                        help='CUDA设备')
    parser.add_argument('--workers', type=int, default=8,
                        help='数据加载工作线程数')
    parser.add_argument('--save-json', action='store_true',
                        help='保存结果为JSON格式')
    parser.add_argument('--project', type=str, default='runs/val',
                        help='项目保存目录')
    parser.add_argument('--name', type=str, default='exp',
                        help='实验名称')
    
    args = parser.parse_args()
    
    validate_model(args)


if __name__ == '__main__':
    main()



