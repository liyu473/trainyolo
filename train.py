"""
YOLO11 训练脚本
使用Ultralytics YOLO进行目标检测模型训练
"""
from ultralytics import YOLO
import os
import yaml
import argparse


def train_yolo(args):
    """
    训练YOLO模型
    
    Args:
        args: 命令行参数
    """
    # 加载模型
    if args.resume:
        # 从上次中断的训练恢复
        model = YOLO(args.resume)
        print(f"从检查点恢复训练: {args.resume}")
    elif args.pretrained:
        # 使用预训练模型
        model = YOLO(args.pretrained)
        print(f"使用预训练模型: {args.pretrained}")
    else:
        # 使用默认的YOLO11n模型
        model = YOLO('yolo11n.pt')
        print("使用默认YOLO11n模型")
    
    # 训练参数
    results = model.train(
        data=args.data,              # 数据集配置文件
        epochs=args.epochs,          # 训练轮数
        imgsz=args.imgsz,            # 图像大小
        batch=args.batch,            # 批次大小
        name=args.name,              # 实验名称
        project=args.project,        # 项目目录
        device=args.device,          # 设备 (0, 1, 2, 3 或 cpu)
        workers=args.workers,        # 数据加载工作线程数
        patience=args.patience,      # 早停耐心值
        save=True,                   # 保存检查点
        save_period=args.save_period,# 每N个epoch保存一次
        cache=args.cache,            # 缓存图像到内存
        optimizer=args.optimizer,    # 优化器
        verbose=True,                # 详细输出
        seed=args.seed,              # 随机种子
        deterministic=False,         # 确定性训练
        single_cls=args.single_cls,  # 单类训练
        rect=False,                  # 矩形训练
        cos_lr=args.cos_lr,          # 余弦学习率调度
        close_mosaic=10,             # 最后N个epoch关闭mosaic增强
        resume=args.resume is not None,
        amp=True,                    # 自动混合精度
        fraction=1.0,                # 使用数据集的比例
        profile=False,               # 性能分析
        lr0=args.lr0,                # 初始学习率
        lrf=args.lrf,                # 最终学习率 (lr0 * lrf)
        momentum=0.937,              # SGD动量/Adam beta1
        weight_decay=0.0005,         # 权重衰减
        warmup_epochs=3.0,           # 预热epoch数
        warmup_momentum=0.8,         # 预热初始动量
        warmup_bias_lr=0.1,          # 预热初始偏置学习率
        box=7.5,                     # box损失权重
        cls=0.5,                     # cls损失权重
        dfl=1.5,                     # dfl损失权重
        pose=12.0,                   # pose损失权重(仅姿态)
        kobj=2.0,                    # 关键点obj损失权重(仅姿态)
        label_smoothing=0.0,         # 标签平滑
        nbs=64,                      # 名义批次大小
        overlap_mask=True,           # 训练期间掩码重叠
        mask_ratio=4,                # 掩码下采样比率
        dropout=0.0,                 # 分类器dropout
        val=True,                    # 训练期间验证
    )
    
    print("\n训练完成!")
    print(f"最佳模型保存在: {model.trainer.best}")
    print(f"最后模型保存在: {model.trainer.last}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='YOLO11训练脚本')
    
    # 基础参数
    parser.add_argument('--data', type=str, default='datasets/data.yaml',
                        help='数据集配置文件路径')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='输入图像大小')
    parser.add_argument('--device', type=str, default='0',
                        help='CUDA设备, 例如 0 或 0,1,2,3 或 cpu')
    parser.add_argument('--workers', type=int, default=8,
                        help='数据加载工作线程数')
    
    # 模型参数
    parser.add_argument('--pretrained', type=str, default='yolo11n.pt',
                        help='预训练模型路径 (yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt)')
    parser.add_argument('--resume', type=str, default=None,
                        help='从检查点恢复训练的路径')
    
    # 项目参数
    parser.add_argument('--project', type=str, default='runs/train',
                        help='项目保存目录')
    parser.add_argument('--name', type=str, default='exp',
                        help='实验名称')
    
    # 训练策略
    parser.add_argument('--patience', type=int, default=50,
                        help='早停耐心值(epochs)')
    parser.add_argument('--save-period', type=int, default=-1,
                        help='每N个epoch保存检查点 (-1表示仅保存最后)')
    parser.add_argument('--cache', type=str, default='',
                        help='缓存图像 (ram/disk/"")')
    parser.add_argument('--optimizer', type=str, default='auto',
                        help='优化器选择 (SGD, Adam, AdamW, auto)')
    parser.add_argument('--seed', type=int, default=0,
                        help='全局训练种子')
    parser.add_argument('--single-cls', action='store_true',
                        help='将所有类别作为单一类别训练')
    parser.add_argument('--cos-lr', action='store_true',
                        help='使用余弦学习率调度器')
    
    # 学习率参数
    parser.add_argument('--lr0', type=float, default=0.01,
                        help='初始学习率')
    parser.add_argument('--lrf', type=float, default=0.01,
                        help='最终学习率比率 (lr0 * lrf)')
    
    args = parser.parse_args()
    
    # 检查数据配置文件是否存在
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"数据集配置文件不存在: {args.data}")
    
    # 开始训练
    train_yolo(args)


if __name__ == '__main__':
    main()


