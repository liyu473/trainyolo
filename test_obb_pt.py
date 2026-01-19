"""
测试 OBB（旋转边界框）PT 模型推理
用于验证 PT 模型是否正常工作
"""
from ultralytics import YOLO
import cv2
import os

# 配置
PT_MODEL_PATH = 'models/icon.pt'  # 修改为你的PT模型路径
TEST_IMAGE_PATH = 'models/test_image.jpg'  # 修改为你的测试图片路径
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

def main():
    print("=" * 60)
    print("OBB PT 模型测试工具")
    print("=" * 60)
    
    # 检查模型文件
    if not os.path.exists(PT_MODEL_PATH):
        print(f"❌ 错误: 模型文件不存在: {PT_MODEL_PATH}")
        return
    
    # 加载模型
    print(f"📦 加载模型: {PT_MODEL_PATH}")
    model = YOLO(PT_MODEL_PATH)
    
    # 打印模型信息
    print(f"📊 模型信息:")
    print(f"   任务类型: {model.task}")
    print(f"   类别名称: {model.names}")
    print(f"   类别数量: {len(model.names)}")
    
    # 检查是否是OBB模型
    if model.task != 'obb':
        print(f"⚠️  警告: 模型任务类型是 '{model.task}'，不是 'obb'")
        print(f"   这可能不是一个OBB模型")
    else:
        print(f"✅ 确认这是一个OBB模型")
    
    # 检查测试图片
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"❌ 错误: 测试图片不存在: {TEST_IMAGE_PATH}")
        return
    
    print(f"\n🖼️  测试图片: {TEST_IMAGE_PATH}")
    
    # 读取图片
    img = cv2.imread(TEST_IMAGE_PATH)
    if img is None:
        print(f"❌ 错误: 无法读取图片: {TEST_IMAGE_PATH}")
        return
    
    print(f"📐 图片尺寸: {img.shape}")
    
    # 进行预测
    print(f"\n🚀 开始推理...")
    results = model.predict(
        source=TEST_IMAGE_PATH,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=640,
        verbose=False,
        save=False
    )
    
    print(f"✅ 推理完成!")
    
    # 分析结果
    result = results[0]
    
    # 检查是否有OBB属性
    if hasattr(result, 'obb') and result.obb is not None:
        print(f"\n✅ 检测到OBB结果")
        obb = result.obb
        
        print(f"\n📊 检测统计:")
        print(f"   检测数量: {len(obb)}")
        
        if len(obb) > 0:
            print(f"\n📋 检测结果详情:")
            
            # 获取所有检测框的信息
            boxes = obb.xyxyxyxy  # 旋转框的四个角点 [N, 4, 2]
            confs = obb.conf  # 置信度 [N]
            classes = obb.cls  # 类别 [N]
            
            # 如果有角度信息
            if hasattr(obb, 'xywhr'):
                xywhr = obb.xywhr  # [cx, cy, w, h, rotation] [N, 5]
                print(f"   (包含旋转角度信息)")
            
            for i in range(len(obb)):
                class_id = int(classes[i])
                class_name = model.names[class_id]
                conf = float(confs[i])
                
                print(f"\n  [{i}] {class_name}:")
                print(f"      置信度: {conf:.4f}")
                
                if hasattr(obb, 'xywhr'):
                    cx, cy, w, h, angle = xywhr[i]
                    print(f"      中心: ({cx:.1f}, {cy:.1f})")
                    print(f"      尺寸: {w:.1f} x {h:.1f}")
                    print(f"      角度: {angle:.4f} 弧度 ({angle * 180 / 3.14159:.2f}°)")
                
                # 打印四个角点
                corners = boxes[i]
                print(f"      角点:")
                for j, corner in enumerate(corners):
                    print(f"        点{j+1}: ({corner[0]:.1f}, {corner[1]:.1f})")
            
            # 绘制结果
            print(f"\n🎨 绘制检测结果...")
            annotated_frame = result.plot()
            
            # 保存结果
            output_path = 'test_obb_pt_result.jpg'
            cv2.imwrite(output_path, annotated_frame)
            print(f"✅ 结果已保存: {output_path}")
            
        else:
            print(f"\n⚠️  未检测到任何目标!")
            print(f"💡 可能的原因:")
            print(f"   1. 测试图片中没有目标")
            print(f"   2. 置信度阈值太高 (当前: {CONF_THRESHOLD})")
            print(f"   3. 模型训练不充分")
    
    elif hasattr(result, 'boxes') and result.boxes is not None:
        print(f"\n⚠️  检测到普通边界框结果，不是OBB")
        print(f"   这个模型可能是普通检测模型，不是OBB模型")
        
        boxes = result.boxes
        print(f"\n📊 检测统计:")
        print(f"   检测数量: {len(boxes)}")
        
        if len(boxes) > 0:
            print(f"\n📋 检测结果详情:")
            for i in range(len(boxes)):
                class_id = int(boxes.cls[i])
                class_name = model.names[class_id]
                conf = float(boxes.conf[i])
                x1, y1, x2, y2 = boxes.xyxy[i]
                
                print(f"  [{i}] {class_name}: {conf:.4f} @ ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
            
            # 绘制结果
            annotated_frame = result.plot()
            output_path = 'test_obb_pt_result.jpg'
            cv2.imwrite(output_path, annotated_frame)
            print(f"\n✅ 结果已保存: {output_path}")
    else:
        print(f"\n❌ 未找到任何检测结果")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
