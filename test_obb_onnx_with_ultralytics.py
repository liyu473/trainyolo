"""
使用 Ultralytics 直接测试 ONNX 模型
看看是否是我们后处理的问题
"""
from ultralytics import YOLO
import cv2

# 配置
ONNX_MODEL_PATH = 'models/icon_obb.onnx'
TEST_IMAGE_PATH = 'models/test_image.jpg'
CONF_THRESHOLD = 0.25

print("=" * 60)
print("使用 Ultralytics 测试 ONNX 模型")
print("=" * 60)

# 加载ONNX模型
print(f"📦 加载ONNX模型: {ONNX_MODEL_PATH}")
model = YOLO(ONNX_MODEL_PATH, task='obb')

print(f"📊 模型信息:")
print(f"   任务类型: {model.task}")
print(f"   类别名称: {model.names}")

# 进行预测
print(f"\n🚀 开始推理...")
results = model.predict(
    source=TEST_IMAGE_PATH,
    conf=CONF_THRESHOLD,
    imgsz=640,
    verbose=False
)

result = results[0]

# 检查结果
if hasattr(result, 'obb') and result.obb is not None:
    obb = result.obb
    print(f"\n✅ 检测到 {len(obb)} 个目标")
    
    if len(obb) > 0:
        print(f"\n📋 检测结果:")
        for i in range(len(obb)):
            class_id = int(obb.cls[i])
            class_name = model.names[class_id]
            conf = float(obb.conf[i])
            
            if hasattr(obb, 'xywhr'):
                cx, cy, w, h, angle = obb.xywhr[i]
                print(f"  [{i}] {class_name}: {conf:.4f}")
                print(f"       中心: ({cx:.1f}, {cy:.1f}), 尺寸: {w:.1f}x{h:.1f}, 角度: {angle*180/3.14159:.2f}°")
        
        # 保存结果
        annotated = result.plot()
        cv2.imwrite('test_obb_ultralytics_result.jpg', annotated)
        print(f"\n✅ 结果已保存: test_obb_ultralytics_result.jpg")
else:
    print("\n❌ 未检测到OBB结果")

print("\n" + "=" * 60)
