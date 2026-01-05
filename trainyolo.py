from starttrain import start_train

if __name__ == '__main__':
    success3 = start_train(
        source_dir='project-6-at-2025-10-29-15-54-bac1d4f3', # label studio 解压之后的文件夹地址
        data_dir='datasets', # 将source_dir转换成数据集目录的保存目录
        prepare_data=True,# 是否需要准备数据集(如果source_dir已经是数据集目录，则不需要，如果是labelstudio解压之后的数据，则需要)
        epochs=100, # 训练轮次
        batch_size=16, # 批次大小
        resume_from=None, # 从哪个模型的基础上进行训练，如果没有就按照默认model_dize(默认是yolo11n.pt)进行训练
        experiment_name='my_test', # 实验名称
        model_output_dir='models', # 模型输出目录
        use_timestamp=True, # 是否使用时间戳作为实验名称
        export_onnx=True, # 是否导出onnx
        force_cpu=False, # 是否强制使用CPU(默认使用GPU)
        image_size=640, # 图片尺寸
        learning_rate=0.01, # 学习率
        workers=8, # 线程数
    )

    print(f"\n📊 训练结果:")
    print(f"  结果: {'✅成功' if success3['success'] else '❌失败'}")
    
    # 打印模型保存路径
    print(f"\n📁 模型保存位置:")  
    print(f"  模型: {success3['model_dir']}")
    print(f"  最佳模型: {success3['best_model']}")
    if success3['onnx_model']:
        print(f"  ONNX模型: {success3['onnx_model']}")
