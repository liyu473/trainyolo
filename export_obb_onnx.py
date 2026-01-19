"""
正确导出 OBB YOLO11 模型为 ONNX 格式
专门用于旋转边界框(OBB)模型
"""
from ultralytics import YOLO
import os

# 配置
PT_MODEL_PATH = 'models/icon.pt'  # 修改为你的PT模型路径
OUTPUT_DIR = 'models'  # 输出目录
OUTPUT_NAME = 'icon_obb.onnx'  # 输出文件名

def export_obb_to_onnx():
    print("=" * 60)
    print("OBB 模型导出为 ONNX")
    print("=" * 60)
    
    # 检查模型文件
    if not os.path.exists(PT_MODEL_PATH):
        print(f"❌ 错误: 模型文件不存在: {PT_MODEL_PATH}")
        return None
    
    # 加载模型
    print(f"📦 加载模型: {PT_MODEL_PATH}")
    model = YOLO(PT_MODEL_PATH)
    
    # 打印模型信息
    print(f"📊 模型信息:")
    print(f"   任务类型: {model.task}")
    print(f"   类别数量: {len(model.names)}")
    print(f"   类别名称: {model.names}")
    
    # 确认是OBB模型
    if model.task != 'obb':
        print(f"⚠️  警告: 该模型不是OBB模型 (任务类型: {model.task})")
        print(f"   继续导出可能会出现问题")
    
    # 导出为ONNX
    print(f"\n🚀 开始导出ONNX...")
    print(f"⚙️  导出配置:")
    print(f"   - 格式: ONNX")
    print(f"   - 图像尺寸: 640x640")
    print(f"   - 简化: True")
    print(f"   - 动态batch: False")
    print(f"   - Opset版本: 12")
    
    try:
        # 导出ONNX (OBB模型会自动使用正确的格式)
        onnx_path = model.export(
            format='onnx',
            imgsz=640,
            simplify=True,
            dynamic=False,
            opset=12,
            # OBB模型导出时会自动处理旋转框格式
        )
        
        print(f"\n✅ ONNX模型已导出: {onnx_path}")
        
        # 复制到指定目录
        import shutil
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        target_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
        shutil.copy2(onnx_path, target_path)
        
        print(f"📁 模型已复制到: {target_path}")
        
        # 导出类别文件
        classes_file = os.path.join(OUTPUT_DIR, 'classes.txt')
        with open(classes_file, 'w', encoding='utf-8') as f:
            for class_name in model.names.values():
                f.write(f"{class_name}\n")
        print(f"📄 类别文件已保存: {classes_file}")
        
        print(f"\n📊 导出信息:")
        print(f"   - 输入尺寸: 640x640")
        print(f"   - 类别数量: {len(model.names)}")
        print(f"   - 输出格式: OBB [cx, cy, w, h, angle, class_scores...]")
        print(f"   - 输出形状: [1, {5+len(model.names)}, 8400]")
        
        return target_path
        
    except Exception as e:
        print(f"\n❌ 导出失败: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    result = export_obb_to_onnx()
    
    if result:
        print("\n" + "=" * 60)
        print("✅ 导出完成!")
        print("=" * 60)
        print(f"\n💡 测试导出的模型:")
        print(f"   python test_obb_onnx.py")
    else:
        print("\n" + "=" * 60)
        print("❌ 导出失败")
        print("=" * 60)
