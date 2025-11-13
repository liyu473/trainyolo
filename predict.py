"""
YOLO11 预测/推理脚本
用于对图像、视频或摄像头进行目标检测
"""
from ultralytics import YOLO
import argparse
import os


def predict(args):
    """
    使用YOLO模型进行预测
    
    Args:
        args: 命令行参数
    """
    # 加载模型
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"模型文件不存在: {args.model}")
    
    model = YOLO(args.model)
    print(f"加载模型: {args.model}")
    
    # 预测
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=args.save,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        save_crop=args.save_crop,
        show=args.show,
        project=args.project,
        name=args.name,
        visualize=args.visualize,
        augment=args.augment,
        agnostic_nms=args.agnostic_nms,
        classes=args.classes,
        retina_masks=True,
        boxes=True,
    )
    
    print(f"\n预测完成! 结果保存在: {args.project}/{args.name}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='YOLO11预测脚本')
    
    # 必需参数
    parser.add_argument('--model', type=str, required=True,
                        help='模型权重文件路径')
    parser.add_argument('--source', type=str, required=True,
                        help='输入源 (图像文件/文件夹/视频/0表示摄像头)')
    
    # 可选参数
    parser.add_argument('--imgsz', type=int, default=640,
                        help='输入图像大小')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='目标置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='NMS的IOU阈值')
    parser.add_argument('--device', type=str, default='0',
                        help='CUDA设备 (0, 1, 2, 3 或 cpu)')
    
    # 输出选项
    parser.add_argument('--save', action='store_true', default=True,
                        help='保存预测结果图像')
    parser.add_argument('--save-txt', action='store_true',
                        help='保存结果为txt文件')
    parser.add_argument('--save-conf', action='store_true',
                        help='在txt文件中保存置信度')
    parser.add_argument('--save-crop', action='store_true',
                        help='保存裁剪的预测框')
    parser.add_argument('--show', action='store_true',
                        help='显示结果')
    parser.add_argument('--project', type=str, default='runs/predict',
                        help='项目保存目录')
    parser.add_argument('--name', type=str, default='exp',
                        help='实验名称')
    
    # 高级选项
    parser.add_argument('--visualize', action='store_true',
                        help='可视化特征图')
    parser.add_argument('--augment', action='store_true',
                        help='使用增强推理')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='类别无关的NMS')
    parser.add_argument('--classes', type=int, nargs='+',
                        help='按类别过滤: --classes 0 2 3')
    
    args = parser.parse_args()
    
    predict(args)


if __name__ == '__main__':
    main()



