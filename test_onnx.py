"""
测试 ONNX 模型推理
用于验证 ONNX 模型是否正常工作
"""
import onnxruntime as ort
import numpy as np
import cv2
import os

# 配置
ONNX_MODEL_PATH = 'models\my_test_continued_20260114_205058\my_test_continued_model.onnx'
TEST_IMAGE_PATH = 'datasets/images/val'  # 使用验证集中的图片
CONF_THRESHOLD = 0.01
IOU_THRESHOLD = 0.45
INPUT_SIZE = 640

# 类别名称
CLASS_NAMES = ['Ellipse', 'EllipseWithHole', 'Key', 'WaterDrop']


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Letterbox 图像预处理，保持宽高比"""
    shape = img.shape[:2]  # 当前形状 [height, width]
    
    # 计算缩放比例
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    # 计算新的未填充尺寸
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    
    # 计算填充
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    
    # 调整大小
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    # 添加边框
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return img, r, (dw, dh)


def preprocess(img):
    """预处理图像"""
    # Letterbox
    img_letterbox, ratio, pad = letterbox(img, (INPUT_SIZE, INPUT_SIZE))
    
    # BGR to RGB
    img_rgb = cv2.cvtColor(img_letterbox, cv2.COLOR_BGR2RGB)
    
    # 归一化到 0-1
    img_normalized = img_rgb.astype(np.float32) / 255.0
    
    # HWC to CHW
    img_transposed = np.transpose(img_normalized, (2, 0, 1))
    
    # 添加 batch 维度
    img_batch = np.expand_dims(img_transposed, axis=0)
    
    return img_batch, ratio, pad


def postprocess(output, ratio, pad, conf_threshold=0.25):
    """后处理 YOLO 输出"""
    # output shape: [1, 8, 8400] -> [1, 4+num_classes, num_predictions]
    output = output[0]  # 移除 batch 维度 -> [8, 8400]
    
    # 转置为 [8400, 8]
    output = np.transpose(output)
    
    detections = []
    
    for pred in output:
        # pred: [cx, cy, w, h, class0_conf, class1_conf, ...]
        cx, cy, w, h = pred[:4]
        class_scores = pred[4:]
        
        # 获取最大类别置信度 (YOLOv8/YOLO11 输出已经是 sigmoid 后的值，但需要确认)
        max_score = np.max(class_scores)
        class_id = np.argmax(class_scores)
        
        if max_score < conf_threshold:
            continue
        
        # 转换为 xyxy 格式
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        
        # 去除 padding 和缩放
        x1 = (x1 - pad[0]) / ratio
        y1 = (y1 - pad[1]) / ratio
        x2 = (x2 - pad[0]) / ratio
        y2 = (y2 - pad[1]) / ratio
        
        detections.append([x1, y1, x2, y2, max_score, class_id])
    
    return np.array(detections) if detections else np.array([])


def nms(detections, iou_threshold=0.45):
    """非极大值抑制"""
    if len(detections) == 0:
        return []
    
    # 按置信度排序
    indices = np.argsort(detections[:, 4])[::-1]
    detections = detections[indices]
    
    keep = []
    while len(detections) > 0:
        keep.append(detections[0])
        if len(detections) == 1:
            break
        
        # 计算 IoU
        ious = compute_iou(detections[0, :4], detections[1:, :4])
        
        # 保留 IoU 小于阈值的
        mask = ious < iou_threshold
        detections = detections[1:][mask]
    
    return np.array(keep)


def compute_iou(box, boxes):
    """计算 IoU"""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    union = box_area + boxes_area - intersection
    
    return intersection / (union + 1e-6)


def draw_detections(img, detections):
    """绘制检测结果"""
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
    
    for det in detections:
        x1, y1, x2, y2, conf, class_id = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        class_id = int(class_id)
        
        color = colors[class_id % len(colors)]
        label = f"{CLASS_NAMES[class_id]}: {conf:.2f}"
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return img


def main():
    # 检查模型文件
    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"错误: 模型文件不存在: {ONNX_MODEL_PATH}")
        return
    
    # 加载 ONNX 模型
    print(f"加载模型: {ONNX_MODEL_PATH}")
    session = ort.InferenceSession(ONNX_MODEL_PATH)
    
    # 打印模型信息
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    print(f"输入: {input_info.name}, shape: {input_info.shape}, dtype: {input_info.type}")
    print(f"输出: {output_info.name}, shape: {output_info.shape}, dtype: {output_info.type}")
    
    # 获取测试图片
    if os.path.isdir(TEST_IMAGE_PATH):
        image_files = [f for f in os.listdir(TEST_IMAGE_PATH) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not image_files:
            print(f"错误: 在 {TEST_IMAGE_PATH} 中没有找到图片")
            return
        test_image = os.path.join(TEST_IMAGE_PATH, image_files[0])
    else:
        test_image = TEST_IMAGE_PATH
    
    print(f"\n测试图片: {test_image}")
    
    # 读取图片
    img = cv2.imread(test_image)
    if img is None:
        print(f"错误: 无法读取图片: {test_image}")
        return
    
    print(f"图片尺寸: {img.shape}")
    
    # 预处理
    input_tensor, ratio, pad = preprocess(img)
    print(f"输入张量形状: {input_tensor.shape}")
    print(f"缩放比例: {ratio}, 填充: {pad}")
    
    # 推理
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_tensor})
    
    print(f"\n输出形状: {output[0].shape}")
    print(f"输出范围: min={output[0].min():.6f}, max={output[0].max():.6f}")
    
    # 检查原始输出中的最大置信度
    raw_output = output[0][0]  # [8, 8400]
    class_scores = raw_output[4:, :]  # [4, 8400]
    max_scores = np.max(class_scores, axis=0)  # [8400]
    print(f"所有预测中的最大置信度: {max_scores.max():.6f}")
    print(f"置信度 > 0.1 的预测数量: {np.sum(max_scores > 0.1)}")
    print(f"置信度 > 0.25 的预测数量: {np.sum(max_scores > 0.25)}")
    print(f"置信度 > 0.5 的预测数量: {np.sum(max_scores > 0.5)}")
    
    # 后处理
    detections = postprocess(output[0], ratio, pad, CONF_THRESHOLD)
    print(f"\n检测到 {len(detections)} 个目标 (NMS前)")
    
    if len(detections) > 0:
        # NMS
        detections = nms(detections, IOU_THRESHOLD)
        print(f"检测到 {len(detections)} 个目标 (NMS后)")
        
        # 打印检测结果
        for i, det in enumerate(detections):
            x1, y1, x2, y2, conf, class_id = det
            print(f"  [{i}] {CLASS_NAMES[int(class_id)]}: {conf:.4f} @ ({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})")
        
        # 绘制结果
        result_img = draw_detections(img.copy(), detections)
        
        # 保存结果
        output_path = 'test_onnx_result.jpg'
        cv2.imwrite(output_path, result_img)
        print(f"\n结果已保存: {output_path}")
    else:
        print("\n未检测到任何目标!")
        print("可能的原因:")
        print("1. 测试图片中没有目标")
        print("2. 置信度阈值太高")
        print("3. 模型训练不充分")


if __name__ == '__main__':
    main()
