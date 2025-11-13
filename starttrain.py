#!/usr/bin/env python3
"""
YOLO训练一体化脚本 - StartTrain.py
功能集成:
1. GPU/CPU检测和选择
2. 数据集准备和格式转换
3. YOLO模型训练
4. 模型导出到Model文件夹
5. 可选ONNX格式转换

作者: AI Assistant
"""

import os
import sys
import argparse
import shutil
import random
import platform
from datetime import datetime
from pathlib import Path

# 检查依赖
try:
    import torch
    from ultralytics import YOLO
    import yaml
except ImportError as e:
    print(f"错误: 缺少必要的依赖包: {e}")
    print("请运行: pip install torch ultralytics pyyaml")
    sys.exit(1)


class YOLOTrainer:
    """YOLO训练器类，整合所有训练功能"""
    
    def __init__(self, args):
        self.args = args
        self.device = None
        self.model_output_dir = None
        
    def check_gpu(self):
        """检测GPU可用性并设置设备"""
        print("=" * 50)
        print("🔍 检测计算设备...")
        print("=" * 50)
        
        print(f"PyTorch版本: {torch.__version__}")
        print(f"系统: {platform.system()} {platform.release()}")
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"✅ 检测到 {gpu_count} 个GPU:")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"   GPU {i}: {gpu_name}")
            
            if self.args.force_cpu:
                print("⚠️  强制使用CPU模式")
                self.device = 'cpu'
            else:
                self.device = '0' if gpu_count == 1 else ','.join(map(str, range(gpu_count)))
                print(f"🚀 将使用GPU进行训练: {self.device}")
        else:
            print("❌ 未检测到可用GPU")
            self.device = 'cpu'
            print("🐌 将使用CPU进行训练")
        
        print()
        return self.device
    
    def prepare_dataset(self):
        """准备训练数据集"""
        if not self.args.prepare_data:
            print("⏩ 跳过数据准备步骤")
            return True
            
        print("=" * 50)
        print("📊 准备训练数据...")
        print("=" * 50)
        
        source_path = Path(self.args.source_dir)
        output_path = Path(self.args.data_dir)
        
        if not source_path.exists():
            print(f"❌ 源数据目录不存在: {source_path}")
            return False
        
        # 创建输出目录结构
        train_img_dir = output_path / 'images' / 'train'
        val_img_dir = output_path / 'images' / 'val'
        train_label_dir = output_path / 'labels' / 'train'
        val_label_dir = output_path / 'labels' / 'val'
        
        for dir_path in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 获取所有图像文件
        image_dir = source_path / 'images'
        label_dir = source_path / 'labels'
        
        if not image_dir.exists() or not label_dir.exists():
            print(f"❌ 源目录结构不正确，需要包含 images/ 和 labels/ 文件夹")
            return False
        
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(image_dir.glob(ext)))
        
        image_files = sorted(image_files)
        print(f"📷 找到 {len(image_files)} 张图像")
        
        if len(image_files) == 0:
            print("❌ 未找到任何图像文件")
            return False
        
        # 随机打乱
        random.seed(self.args.seed)
        random.shuffle(image_files)
        
        # 划分数据集
        split_idx = int(len(image_files) * self.args.train_split)
        train_images = image_files[:split_idx]
        val_images = image_files[split_idx:]
        
        print(f"📈 训练集: {len(train_images)} 张图像")
        print(f"📉 验证集: {len(val_images)} 张图像")
        
        # 复制文件
        def copy_files(file_list, img_target, label_target, dataset_type):
            print(f"📋 复制{dataset_type}...")
            for img_path in file_list:
                # 复制图像
                shutil.copy2(img_path, img_target / img_path.name)
                
                # 复制标注
                label_name = img_path.stem + '.txt'
                label_path = label_dir / label_name
                if label_path.exists():
                    shutil.copy2(label_path, label_target / label_name)
                else:
                    print(f"⚠️  缺少标注文件: {label_name}")
        
        copy_files(train_images, train_img_dir, train_label_dir, "训练集")
        copy_files(val_images, val_img_dir, val_label_dir, "验证集")
        
        # 读取类别信息
        classes_file = source_path / 'classes.txt'
        if classes_file.exists():
            with open(classes_file, 'r', encoding='utf-8') as f:
                classes = [line.strip() for line in f if line.strip()]
        else:
            print("⚠️  未找到classes.txt，使用默认类别")
            classes = ['object']
        
        # 创建data.yaml配置文件
        yaml_content = f"""# YOLO数据集配置文件
# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

path: {output_path.absolute().as_posix()}  # 数据集根目录
train: images/train  # 训练图像相对路径
val: images/val      # 验证图像相对路径

# 类别配置
nc: {len(classes)}  # 类别数量
names: {classes}  # 类别名称列表
"""
        
        yaml_path = output_path / 'data.yaml'
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        
        print(f"✅ 数据准备完成!")
        print(f"📁 输出目录: {output_path}")
        print(f"📄 配置文件: {yaml_path}")
        print(f"🏷️  类别数量: {len(classes)}")
        print(f"🏷️  类别名称: {classes}")
        print()
        
        return True
    
    def setup_model_output_dir(self):
        """设置模型输出目录，支持自定义路径"""
        base_dir = Path(self.args.model_output_dir)
        
        if self.args.use_timestamp:
            # 使用时间戳避免覆盖
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            exp_name = f"{self.args.experiment_name}_{timestamp}"
            self.model_output_dir = base_dir / exp_name
        else:
            # 直接使用实验名称或指定的完整路径
            if base_dir.is_absolute() and self.args.experiment_name in str(base_dir):
                # 如果model_output_dir已经包含了完整路径，直接使用
                self.model_output_dir = base_dir
            else:
                # 否则在base_dir下创建experiment_name目录
                self.model_output_dir = base_dir / self.args.experiment_name
        
        self.model_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 模型将保存到: {self.model_output_dir}")
        return str(self.model_output_dir)
    
    def train_model(self):
        """训练YOLO模型"""
        print("=" * 50)
        print("🚀 开始训练YOLO模型...")
        print("=" * 50)
        
        # 检查数据配置文件
        data_yaml = Path(self.args.data_yaml)
        if not data_yaml.exists():
            print(f"❌ 数据配置文件不存在: {data_yaml}")
            return None
        
        # 设置模型输出目录
        project_dir = self.setup_model_output_dir()
        
        # 加载模型
        if self.args.resume_from:
            model = YOLO(self.args.resume_from)
            print(f"📂 从检查点恢复训练: {self.args.resume_from}")
        else:
            model = YOLO(self.args.model_size)
            print(f"🤖 使用预训练模型: {self.args.model_size}")
        
        print(f"⚙️  训练参数:")
        print(f"   - 训练轮数: {self.args.epochs}")
        print(f"   - 批次大小: {self.args.batch_size}")
        print(f"   - 图像大小: {self.args.image_size}")
        print(f"   - 设备: {self.device}")
        print(f"   - 工作线程: {self.args.workers}")
        print()
        
        # 开始训练
        results = model.train(
            data=str(data_yaml),
            epochs=self.args.epochs,
            imgsz=self.args.image_size,
            batch=self.args.batch_size,
            name='train',
            project=project_dir,
            device=self.device,
            workers=self.args.workers,
            patience=self.args.patience,
            save=True,
            save_period=self.args.save_period if self.args.save_period > 0 else -1,
            cache=self.args.cache,
            optimizer=self.args.optimizer,
            verbose=True,
            seed=self.args.seed,
            lr0=self.args.learning_rate,
            lrf=self.args.lr_final_ratio,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            cos_lr=self.args.cosine_lr,
            val=True,
        )
        
        print("\n" + "=" * 50)
        print("✅ 训练完成!")
        print("=" * 50)
        
        # 训练结果路径
        train_dir = Path(project_dir) / 'train'
        weights_dir = train_dir / 'weights'
        
        best_model = weights_dir / 'best.pt'
        last_model = weights_dir / 'last.pt'
        
        if best_model.exists():
            print(f"🏆 最佳模型: {best_model}")
        if last_model.exists():
            print(f"📱 最终模型: {last_model}")
        
        return best_model if best_model.exists() else last_model
    
    def export_to_onnx(self, model_path):
        """导出模型为ONNX格式"""
        if not self.args.export_onnx:
            print("⏩ 跳过ONNX导出")
            return
        
        if not model_path or not Path(model_path).exists():
            print("❌ 模型文件不存在，无法导出ONNX")
            return
        
        print("=" * 50)
        print("📦 导出ONNX模型...")
        print("=" * 50)
        
        try:
            model = YOLO(str(model_path))
            
            print(f"📄 加载模型: {model_path}")
            print(f"🔧 导出配置: 图像大小={self.args.image_size}, 简化=True")
            
            # 导出ONNX
            onnx_path = model.export(
                format='onnx',
                imgsz=self.args.image_size,
                simplify=True,
                dynamic=False,
                opset=12
            )
            
            # 复制到Model目录
            onnx_filename = f"{self.args.experiment_name}_model.onnx"
            target_onnx = self.model_output_dir / onnx_filename
            shutil.copy2(onnx_path, target_onnx)
            
            print(f"✅ ONNX模型已导出: {target_onnx}")
            print(f"📊 模型信息:")
            print(f"   - 输入尺寸: {self.args.image_size}x{self.args.image_size}")
            print(f"   - 格式: ONNX (opset=12)")
            
        except Exception as e:
            print(f"❌ ONNX导出失败: {e}")
    
    def run(self):
        """执行完整的训练流程"""
        print("\n" + "🎯" * 20)
        print("YOLO训练一体化脚本启动")
        print("🎯" * 20)
        
        # 1. 检测GPU/CPU
        self.check_gpu()
        
        # 2. 准备数据集
        if not self.prepare_dataset():
            print("❌ 数据准备失败，终止训练")
            return False
        
        # 3. 训练模型
        model_path = self.train_model()
        if not model_path:
            print("❌ 模型训练失败")
            return False
        
        # 4. 导出ONNX
        self.export_to_onnx(model_path)
        
        print("\n" + "🎉" * 20)
        print("🎉 训练流程全部完成! 🎉")
        print("🎉" * 20)
        print(f"📁 输出目录: {self.model_output_dir}")
        
        return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='YOLO训练一体化脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基础训练
  python StartTrain.py --source-dir project-6-at-2025-10-29-15-54-bac1d4f3 --epochs 100
  
  # 完整配置训练
  python StartTrain.py \\
    --source-dir project-6-at-2025-10-29-15-54-bac1d4f3 \\
    --data-dir datasets \\
    --epochs 200 \\
    --batch-size 32 \\
    --model-size yolo11s.pt \\
    --export-onnx \\
    --experiment-name checkpoint_detection
        """
    )
    
    # 数据相关参数
    data_group = parser.add_argument_group('数据配置')
    data_group.add_argument('--source-dir', type=str, 
                           default='project-6-at-2025-10-29-15-54-bac1d4f3',
                           help='原始数据集目录')
    data_group.add_argument('--data-dir', type=str, default='datasets',
                           help='处理后的数据集输出目录')
    data_group.add_argument('--data-yaml', type=str, default='datasets/data.yaml',
                           help='YOLO数据配置文件路径')
    data_group.add_argument('--prepare-data', action='store_true', default=True,
                           help='是否执行数据准备步骤')
    data_group.add_argument('--no-prepare-data', dest='prepare_data', action='store_false',
                           help='跳过数据准备步骤')
    data_group.add_argument('--train-split', type=float, default=0.8,
                           help='训练集比例 (0.0-1.0)')
    
    # 训练相关参数
    train_group = parser.add_argument_group('训练配置')
    train_group.add_argument('--epochs', type=int, default=100,
                            help='训练轮数')
    train_group.add_argument('--batch-size', type=int, default=16,
                            help='批次大小')
    train_group.add_argument('--image-size', type=int, default=640,
                            help='输入图像大小')
    train_group.add_argument('--model-size', type=str, default='yolo11n.pt',
                            choices=['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt'],
                            help='预训练模型大小')
    train_group.add_argument('--resume-from', type=str, default=None,
                            help='从检查点恢复训练的模型路径')
    train_group.add_argument('--workers', type=int, default=8,
                            help='数据加载工作线程数')
    train_group.add_argument('--patience', type=int, default=50,
                            help='早停耐心值(epochs)')
    train_group.add_argument('--save-period', type=int, default=-1,
                            help='每N个epoch保存检查点 (-1仅保存最后)')
    
    # 优化器相关参数
    optim_group = parser.add_argument_group('优化器配置')
    optim_group.add_argument('--optimizer', type=str, default='auto',
                            choices=['SGD', 'Adam', 'AdamW', 'auto'],
                            help='优化器选择')
    optim_group.add_argument('--learning-rate', type=float, default=0.01,
                            help='初始学习率')
    optim_group.add_argument('--lr-final-ratio', type=float, default=0.01,
                            help='最终学习率比率')
    optim_group.add_argument('--cosine-lr', action='store_true',
                            help='使用余弦学习率调度器')
    optim_group.add_argument('--cache', type=str, default='',
                            choices=['', 'ram', 'disk'],
                            help='图像缓存方式')
    
    # 设备相关参数
    device_group = parser.add_argument_group('设备配置')
    device_group.add_argument('--force-cpu', action='store_true',
                             help='强制使用CPU训练')
    
    # 输出相关参数
    output_group = parser.add_argument_group('输出配置')
    output_group.add_argument('--model-output-dir', type=str, default='Model',
                             help='模型输出根目录')
    output_group.add_argument('--experiment-name', type=str, default='yolo_train',
                             help='实验名称')
    output_group.add_argument('--export-onnx', action='store_true',
                             help='训练完成后导出ONNX格式')
    
    # 其他参数
    misc_group = parser.add_argument_group('其他配置')
    misc_group.add_argument('--seed', type=int, default=42,
                           help='随机种子')
    misc_group.add_argument('--verbose', action='store_true', default=True,
                           help='详细输出')
    
    args = parser.parse_args()
    
    # 创建训练器并运行
    trainer = YOLOTrainer(args)
    
    try:
        success = trainer.run()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ 训练被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def start_train(
    source_dir='project-6-at-2025-10-29-15-54-bac1d4f3',
    epochs=100,
    batch_size=16,
    model_size='yolo11n.pt',
    experiment_name='yolo_train',
    export_onnx=False,
    force_cpu=False,
    data_dir='datasets',
    image_size=640,
    learning_rate=0.01,
    workers=8,
    resume_from=None,
    model_output_dir='Model',
    use_timestamp=True,
    prepare_data=True,
    **kwargs
):
    """
    直接调用训练函数，像C#那样传参
    
    Args:
        source_dir (str): 原始数据集目录
        epochs (int): 训练轮数
        batch_size (int): 批次大小
        model_size (str): 模型大小 ('yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt')
        experiment_name (str): 实验名称
        export_onnx (bool): 是否导出ONNX格式
        force_cpu (bool): 是否强制使用CPU
        data_dir (str): 处理后数据集输出目录
        image_size (int): 输入图像大小
        learning_rate (float): 学习率
        workers (int): 工作线程数
        resume_from (str): 从指定模型/检查点恢复训练的路径，如果为None则从头开始训练
        model_output_dir (str): 模型输出根目录，可以指定完整路径
        use_timestamp (bool): 是否在输出目录名中添加时间戳，False则直接使用experiment_name
        prepare_data (bool): 是否执行数据准备步骤，False则跳过数据转换
        **kwargs: 其他参数
    
    Returns:
        dict: 包含训练结果和模型路径的字典
            {
                'success': bool,  # 训练是否成功
                'model_dir': str,  # 模型输出目录
                'best_model': str,  # 最佳模型路径
                'last_model': str,  # 最终模型路径
                'onnx_model': str  # ONNX模型路径(如果导出)
            }
        
    Example:
        # 基础调用
        success = start_train(
            source_dir='project-6-at-2025-10-29-15-54-bac1d4f3',
            epochs=50,
            batch_size=16
        )
        
        # 完整调用
        success = start_train(
            source_dir='my_dataset',
            epochs=200,
            batch_size=32,
            model_size='yolo11s.pt',
            experiment_name='my_model',
            export_onnx=True,
            learning_rate=0.01
        )
        
        # 从已有模型继续训练
        success = start_train(
            source_dir='my_dataset',
            epochs=100,
            batch_size=16,
            experiment_name='fine_tuned_model',
            resume_from='Model/previous_model_20241113_120000/train/weights/best.pt'
        )
        
        # 指定保存地址
        success = start_train(
            source_dir='my_dataset',
            epochs=100,
            experiment_name='custom_model',
            model_output_dir='D:/MyModels/CustomPath',  # 自定义保存路径
            use_timestamp=False  # 不使用时间戳
        )
    """
    
    # 创建参数对象
    class Args:
        def __init__(self):
            # 数据配置
            self.source_dir = source_dir
            self.data_dir = data_dir
            self.data_yaml = f'{data_dir}/data.yaml'
            self.prepare_data = prepare_data
            self.train_split = kwargs.get('train_split', 0.8)
            
            # 训练配置
            self.epochs = epochs
            self.batch_size = batch_size
            self.image_size = image_size
            self.model_size = model_size
            self.resume_from = resume_from
            self.workers = workers
            self.patience = kwargs.get('patience', 50)
            self.save_period = kwargs.get('save_period', -1)
            
            # 优化器配置
            self.optimizer = kwargs.get('optimizer', 'auto')
            self.learning_rate = learning_rate
            self.lr_final_ratio = kwargs.get('lr_final_ratio', 0.01)
            self.cosine_lr = kwargs.get('cosine_lr', False)
            self.cache = kwargs.get('cache', '')
            
            # 设备配置
            self.force_cpu = force_cpu
            
            # 输出配置
            self.model_output_dir = model_output_dir
            self.experiment_name = experiment_name
            self.export_onnx = export_onnx
            self.use_timestamp = use_timestamp
            
            # 其他配置
            self.seed = kwargs.get('seed', 42)
            self.verbose = kwargs.get('verbose', True)
    
    # 创建训练器并运行
    args = Args()
    trainer = YOLOTrainer(args)
    
    try:
        print(f"\n🚀 开始训练 - 实验名称: {experiment_name}")
        print(f"📂 数据源: {source_dir}")
        print(f"⚙️  配置: {epochs}轮次, 批次大小{batch_size}")
        
        if resume_from:
            print(f"📂 从模型恢复: {resume_from}")
        else:
            print(f"🤖 使用预训练模型: {model_size}")
            
        print(f"💾 输出: Model/{experiment_name}_[时间戳]/")
        if export_onnx:
            print(f"📦 将导出ONNX格式")
        print()
        
        success = trainer.run()
        
        # 收集模型路径信息
        result = {
            'success': success,
            'model_dir': str(trainer.model_output_dir) if trainer.model_output_dir else None,
            'best_model': None,
            'last_model': None,
            'onnx_model': None
        }
        
        if success and trainer.model_output_dir:
            # 构建模型路径
            train_dir = Path(trainer.model_output_dir) / 'train'
            weights_dir = train_dir / 'weights'
            
            best_model = weights_dir / 'best.pt'
            last_model = weights_dir / 'last.pt'
            
            if best_model.exists():
                result['best_model'] = str(best_model)
            if last_model.exists():
                result['last_model'] = str(last_model)
            
            # ONNX模型路径
            if export_onnx:
                onnx_filename = f"{experiment_name}_model.onnx"
                onnx_path = Path(trainer.model_output_dir) / onnx_filename
                if onnx_path.exists():
                    result['onnx_model'] = str(onnx_path)
            
            print(f"\n✅ 训练成功完成!")
            print(f"📁 模型目录: {result['model_dir']}")
            if result['best_model']:
                print(f"🏆 最佳模型: {result['best_model']}")
            if result['last_model']:
                print(f"📱 最终模型: {result['last_model']}")
            if result['onnx_model']:
                print(f"📦 ONNX模型: {result['onnx_model']}")
        else:
            print(f"\n❌ 训练失败")
            
        return result
        
    except KeyboardInterrupt:
        print("\n❌ 训练被用户中断")
        return {'success': False, 'model_dir': None, 'best_model': None, 'last_model': None, 'onnx_model': None}
    except Exception as e:
        print(f"\n❌ 训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'model_dir': None, 'best_model': None, 'last_model': None, 'onnx_model': None}


if __name__ == '__main__':
    main()
