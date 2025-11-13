"""
Label Studio导出数据转换为YOLO格式
支持从Label Studio导出的JSON格式转换为YOLO训练格式
"""
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


def convert_labelstudio_to_yolo(json_file: str, output_dir: str, 
                                 train_split: float = 0.8,
                                 copy_images: bool = True):
    """
    将Label Studio导出的JSON转换为YOLO格式
    
    Args:
        json_file: Label Studio导出的JSON文件路径
        output_dir: 输出目录
        train_split: 训练集比例 (0-1之间)
        copy_images: 是否复制图像文件
    """
    # 创建输出目录
    output_path = Path(output_dir)
    train_img_dir = output_path / 'images' / 'train'
    val_img_dir = output_path / 'images' / 'val'
    train_label_dir = output_path / 'labels' / 'train'
    val_label_dir = output_path / 'labels' / 'val'
    
    for dir_path in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # 读取Label Studio导出的JSON
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 收集所有类别
    classes = set()
    annotations_data = []
    
    # 解析每个标注任务
    for task in data:
        if 'annotations' not in task or len(task['annotations']) == 0:
            continue
        
        # 获取图像信息
        image_url = task['data'].get('image', '')
        if not image_url:
            continue
        
        # 获取图像文件名
        image_filename = os.path.basename(image_url)
        
        # 获取图像尺寸
        annotation = task['annotations'][0]  # 使用第一个标注
        results = annotation.get('result', [])
        
        # 原始图像尺寸
        original_width = task.get('data', {}).get('width', 0)
        original_height = task.get('data', {}).get('height', 0)
        
        # 解析标注框
        boxes = []
        for result in results:
            if result['type'] == 'rectanglelabels':
                # 获取标签
                labels = result['value'].get('rectanglelabels', [])
                if not labels:
                    continue
                
                label = labels[0]
                classes.add(label)
                
                # 获取边界框坐标 (百分比格式)
                x = result['value']['x']  # 左上角x (%)
                y = result['value']['y']  # 左上角y (%)
                width = result['value']['width']  # 宽度 (%)
                height = result['value']['height']  # 高度 (%)
                
                # Label Studio使用的是图像的百分比坐标
                # 转换为YOLO格式 (中心点x, 中心点y, 宽度, 高度, 都是归一化的0-1)
                x_center = (x + width / 2) / 100.0
                y_center = (y + height / 2) / 100.0
                w_norm = width / 100.0
                h_norm = height / 100.0
                
                boxes.append({
                    'label': label,
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': w_norm,
                    'height': h_norm
                })
        
        if boxes:
            annotations_data.append({
                'image_filename': image_filename,
                'image_url': image_url,
                'boxes': boxes
            })
    
    # 创建类别映射
    classes_list = sorted(list(classes))
    class_to_id = {cls: idx for idx, cls in enumerate(classes_list)}
    
    print(f"发现 {len(classes_list)} 个类别: {classes_list}")
    print(f"共 {len(annotations_data)} 个标注任务")
    
    # 划分训练集和验证集
    import random
    random.seed(42)
    random.shuffle(annotations_data)
    
    split_idx = int(len(annotations_data) * train_split)
    train_data = annotations_data[:split_idx]
    val_data = annotations_data[split_idx:]
    
    print(f"训练集: {len(train_data)} 张图像")
    print(f"验证集: {len(val_data)} 张图像")
    
    # 处理训练集
    process_split(train_data, train_img_dir, train_label_dir, 
                  class_to_id, copy_images)
    
    # 处理验证集
    process_split(val_data, val_img_dir, val_label_dir, 
                  class_to_id, copy_images)
    
    # 生成data.yaml配置文件
    generate_yaml(output_path, classes_list)
    
    print("\n转换完成!")
    print(f"输出目录: {output_dir}")
    print(f"配置文件: {output_path / 'data.yaml'}")


def process_split(data: List[Dict], img_dir: Path, label_dir: Path,
                  class_to_id: Dict[str, int], copy_images: bool):
    """处理单个数据集分割"""
    for item in data:
        image_filename = item['image_filename']
        image_url = item['image_url']
        boxes = item['boxes']
        
        # 生成标注文件
        label_filename = Path(image_filename).stem + '.txt'
        label_path = label_dir / label_filename
        
        with open(label_path, 'w') as f:
            for box in boxes:
                class_id = class_to_id[box['label']]
                line = f"{class_id} {box['x_center']:.6f} {box['y_center']:.6f} {box['width']:.6f} {box['height']:.6f}\n"
                f.write(line)
        
        # 如果需要，复制图像文件
        if copy_images:
            # 尝试从本地路径复制
            # Label Studio的image_url可能是: /data/upload/xxx.jpg
            # 需要根据实际情况调整
            if os.path.exists(image_url):
                src_path = image_url
            else:
                # 如果是相对路径，尝试在当前目录查找
                src_path = Path(image_filename)
                if not src_path.exists():
                    print(f"警告: 找不到图像文件 {image_filename}, 跳过复制")
                    continue
            
            dst_path = img_dir / image_filename
            try:
                shutil.copy2(src_path, dst_path)
            except Exception as e:
                print(f"警告: 复制图像失败 {image_filename}: {e}")


def generate_yaml(output_path: Path, classes_list: List[str]):
    """生成YOLO数据集配置文件"""
    yaml_content = f"""# YOLO数据集配置文件
# 由Label Studio转换脚本自动生成

path: {output_path.absolute().as_posix()}  # 数据集根目录
train: images/train  # 训练图像
val: images/val      # 验证图像

# 类别
nc: {len(classes_list)}  # 类别数量
names: {classes_list}  # 类别名称
"""
    
    yaml_path = output_path / 'data.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)


def main():
    parser = argparse.ArgumentParser(
        description='将Label Studio导出的JSON转换为YOLO格式'
    )
    parser.add_argument('--json', type=str, required=True,
                        help='Label Studio导出的JSON文件路径')
    parser.add_argument('--output', type=str, default='datasets',
                        help='输出目录 (默认: datasets)')
    parser.add_argument('--train-split', type=float, default=0.8,
                        help='训练集比例 (默认: 0.8)')
    parser.add_argument('--no-copy-images', action='store_true',
                        help='不复制图像文件 (如果图像已在正确位置)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.json):
        raise FileNotFoundError(f"JSON文件不存在: {args.json}")
    
    convert_labelstudio_to_yolo(
        json_file=args.json,
        output_dir=args.output,
        train_split=args.train_split,
        copy_images=not args.no_copy_images
    )


if __name__ == '__main__':
    main()



