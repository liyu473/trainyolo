"""
准备训练数据：划分训练集和验证集
"""
import os
import shutil
from pathlib import Path
import random


def prepare_dataset(source_dir, output_dir, train_split=0.8, seed=42):
    """
    将数据集划分为训练集和验证集
    
    Args:
        source_dir: 源数据目录
        output_dir: 输出目录
        train_split: 训练集比例
        seed: 随机种子
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # 创建输出目录
    train_img_dir = output_path / 'images' / 'train'
    val_img_dir = output_path / 'images' / 'val'
    train_label_dir = output_path / 'labels' / 'train'
    val_label_dir = output_path / 'labels' / 'val'
    
    for dir_path in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图像文件
    image_dir = source_path / 'images'
    label_dir = source_path / 'labels'
    
    image_files = sorted(list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg')))
    
    print(f"找到 {len(image_files)} 张图像")
    
    # 随机打乱
    random.seed(seed)
    random.shuffle(image_files)
    
    # 划分数据集
    split_idx = int(len(image_files) * train_split)
    train_images = image_files[:split_idx]
    val_images = image_files[split_idx:]
    
    print(f"训练集: {len(train_images)} 张图像")
    print(f"验证集: {len(val_images)} 张图像")
    
    # 复制训练集
    print("\n复制训练集...")
    for img_path in train_images:
        # 复制图像
        shutil.copy2(img_path, train_img_dir / img_path.name)
        
        # 复制标注
        label_name = img_path.stem + '.txt'
        label_path = label_dir / label_name
        if label_path.exists():
            shutil.copy2(label_path, train_label_dir / label_name)
    
    # 复制验证集
    print("复制验证集...")
    for img_path in val_images:
        # 复制图像
        shutil.copy2(img_path, val_img_dir / img_path.name)
        
        # 复制标注
        label_name = img_path.stem + '.txt'
        label_path = label_dir / label_name
        if label_path.exists():
            shutil.copy2(label_path, val_label_dir / label_name)
    
    # 读取类别
    classes_file = source_path / 'classes.txt'
    with open(classes_file, 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f if line.strip()]
    
    # 创建data.yaml
    yaml_content = f"""# YOLO数据集配置文件
# 检查点目标检测数据集

path: {output_path.absolute().as_posix()}  # 数据集根目录
train: images/train  # 训练图像
val: images/val      # 验证图像

# 类别
nc: {len(classes)}  # 类别数量
names: {classes}  # 类别名称
"""
    
    yaml_path = output_path / 'data.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"\n数据准备完成!")
    print(f"输出目录: {output_dir}")
    print(f"配置文件: {yaml_path}")
    print(f"类别: {classes}")


if __name__ == '__main__':
    # 配置
    source_dir = 'project-6-at-2025-10-29-15-54-bac1d4f3'
    output_dir = 'datasets'
    train_split = 0.8
    
    prepare_dataset(source_dir, output_dir, train_split)



