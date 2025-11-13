#!/usr/bin/env python3
"""
StartTrain.py 函数调用示例
展示如何像C#那样直接在代码中传参调用训练函数
"""

from starttrain import start_train

def example_basic_training():
    """基础训练示例"""
    print("=" * 50)
    print("🎯 基础训练示例")
    print("=" * 50)
    
    success = start_train(
        source_dir='project-6-at-2025-10-29-15-54-bac1d4f3',
        epochs=10,  # 测试用少量轮次
        batch_size=8,
        experiment_name='basic_test'
    )
    
    if success['success']:
        print("✅ 基础训练完成!")
        print(f"📁 模型保存在: {success['model_dir']}")
        print(f"🏆 最佳模型: {success['best_model']}")
    else:
        print("❌ 基础训练失败!")
    
    return success


def example_advanced_training():
    """高级训练示例"""
    print("=" * 50)
    print("🚀 高级训练示例")
    print("=" * 50)
    
    success = start_train(
        source_dir='project-6-at-2025-10-29-15-54-bac1d4f3',
        epochs=200,
        batch_size=32,
        model_size='yolo11s.pt',  # 更大的模型
        experiment_name='advanced_model',
        export_onnx=True,  # 导出ONNX
        learning_rate=0.01,
        image_size=640,
        workers=8,
        # 额外参数通过kwargs传递
        cosine_lr=True,
        patience=30,
        cache='ram'
    )
    
    if success['success']:
        print("✅ 高级训练完成!")
        print(f"📁 模型保存在: {success['model_dir']}")
        print(f"🏆 最佳模型: {success['best_model']}")
    else:
        print("❌ 高级训练失败!")
    
    return success


def example_cpu_training():
    """CPU训练示例"""
    print("=" * 50)
    print("🖥️  CPU训练示例")
    print("=" * 50)
    
    success = start_train(
        source_dir='project-6-at-2025-10-29-15-54-bac1d4f3',
        epochs=20,
        batch_size=4,  # CPU用小批次
        force_cpu=True,  # 强制CPU
        experiment_name='cpu_test',
        workers=2  # CPU用少线程
    )
    
    if success['success']:
        print("✅ CPU训练完成!")
        print(f"📁 模型保存在: {success['model_dir']}")
        print(f"🏆 最佳模型: {success['best_model']}")
    else:
        print("❌ CPU训练失败!")
    
    return success


def example_resume_training():
    """恢复训练示例"""
    print("=" * 50)
    print("🔄 恢复训练示例")
    print("=" * 50)
    
    # 假设有之前的检查点
    checkpoint_path = "Model/yolo_train_20241113_143000/train/weights/last.pt"
    
    success = start_train(
        source_dir='project-6-at-2025-10-29-15-54-bac1d4f3',
        epochs=50,
        batch_size=16,
        experiment_name='resumed_training',
        resume_from=checkpoint_path  # 从检查点恢复
    )
    
    if success['success']:
        print("✅ 恢复训练完成!")
        print(f"📁 模型保存在: {success['model_dir']}")
        print(f"🏆 最佳模型: {success['best_model']}")
    else:
        print("❌ 恢复训练失败!")
    
    return success


def main():
    """主函数 - 演示不同的调用方式"""
    print("\n🎯 StartTrain.py 函数调用示例")
    print("=" * 60)
    
    # 选择要运行的示例
    examples = {
        '1': ('基础训练', example_basic_training),
        '2': ('高级训练', example_advanced_training),
        '3': ('CPU训练', example_cpu_training),
        '4': ('恢复训练', example_resume_training)
    }
    
    print("请选择要运行的示例:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    
    choice = input("\n请输入选择 (1-4): ").strip()
    
    if choice in examples:
        name, func = examples[choice]
        print(f"\n开始运行: {name}")
        func()
    else:
        print("❌ 无效选择，运行默认基础训练示例")
        example_basic_training()


if __name__ == '__main__':
    # 直接调用示例 - 像C#那样
    
    # 方式1: 最简单的调用
    print("方式1: 最简单调用")
    success1 = start_train()  # 使用所有默认参数
    
    # 方式2: 指定部分参数
    print("\n方式2: 指定部分参数")
    success2 = start_train(
        epochs=50,
        batch_size=16,
        experiment_name='my_test'        
    )

    #通用示例
    success3 = start_train(
        source_dir='project-6-at-2025-10-29-15-54-bac1d4f3', # label studio 解压之后的文件夹地址
        data_dir='datasets', # 将source_dir转换成数据集目录的保存目录
        prepare_data=True,# 是否需要准备数据集(如果source_dir已经是数据集目录，则不需要，如果是labelstudio解压之后的数据，则需要)
        epochs=100, # 训练轮次
        batch_size=16, # 批次大小
        experiment_name='my_test', # 实验名称
        resume_from=None, # 从哪个模型的基础上进行训练，如果没有就按照默认model_dize(默认是yolo11n.pt)进行训练
        experiment_name='my_test', # 实验名称
        model_output_dir='Model', # 模型输出目录
        use_timestamp=True, # 是否使用时间戳作为实验名称
        export_onnx=False, # 是否导出onnx
        force_cpu=False, # 是否强制使用CPU(默认使用GPU)
        image_size=640, # 图片尺寸
        learning_rate=0.01, # 学习率
        workers=8, # 线程数
    )
    
    print(f"\n📊 训练结果:")
    print(f"  默认训练: {'✅成功' if success1['success'] else '❌失败'}")
    print(f"  测试训练: {'✅成功' if success2['success'] else '❌失败'}")
    print(f"  通用示例: {'✅成功' if success3['success'] else '❌失败'}")
    
    # 打印模型保存路径
    print(f"\n📁 模型保存位置:")
    if success1['success']:
        print(f"  默认训练模型: {success1['model_dir']}")
        print(f"  默认最佳模型: {success1['best_model']}")
    
    if success2['success']:
        print(f"  测试训练模型: {success2['model_dir']}")
        print(f"  测试最佳模型: {success2['best_model']}")
    
    if success3['success']:
        print(f"  通用示例模型: {success3['model_dir']}")
        print(f"  通用最佳模型: {success3['best_model']}")
        if success3['onnx_model']:
            print(f"  通用ONNX模型: {success3['onnx_model']}")
