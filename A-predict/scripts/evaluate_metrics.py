import os
import json
import numpy as np
from mmdet.apis import init_detector, inference_detector

# ================= 1. 配置参数 =================
config_file = '/group/chenjinming/SOTA_MMD/A-config/cascade_150e_test_unlimit.py'
checkpoint_file = '/group/chenjinming/SOTA_MMD/A-output/cascade_mask_rcnn_r50_150e_aug/epoch_150.pth'
data_dir = '/group/chenjinming/Datas/test-img-json'

# 置信度阈值：大于这个分数的预测框才会被保留
SCORE_THR = 0.15
# IoU 匹配阈值：预测框和真实框重合度大于多少，算作“成功匹配”
IOU_THR = 0.40

# ================= 2. 核心：IoU 计算函数 =================
def calculate_iou(boxes1, boxes2):
    """计算两组 Bounding Box 之间的 IoU 矩阵"""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)))
    
    tl = np.maximum(boxes1[:, None, :2], boxes2[:, :2])
    br = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    hw = np.maximum(br - tl, 0)
    inter_area = hw[:, :, 0] * hw[:, :, 1]
    
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union_area = area1[:, None] + area2 - inter_area
    
    return inter_area / union_area

# ================= 3. 初始化模型 =================
print("🚀 正在加载模型 (Cascade 150e 无封印版)...")
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# ================= 4. 开始遍历评估 =================
img_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

total_gt = 0
total_detected = 0
total_multiple_detected = 0
total_pred_valid = 0

print(f"\n🔍 开始评估 {len(img_files)} 张图片...\n")
print("-" * 75)
print(f"{'图片名称':<25} | {'人工标注':<5} | {'有效预测':<5} | {'检出率(Recall)':<10} | {'多次检出数':<5}")
print("-" * 75)

for img_name in img_files:
    img_path = os.path.join(data_dir, img_name)
    json_name = os.path.splitext(img_name)[0] + '.json'
    json_path = os.path.join(data_dir, json_name)
    
    if not os.path.exists(json_path):
        print(f"⚠️ 找不到 {img_name} 对应的 JSON 文件，跳过。")
        continue

    # --- 解析人工标注 (Ground Truth) ---
    with open(json_path, 'r', encoding='utf-8') as f:
        anno_data = json.load(f)
    
    gt_bboxes = []
    for shape in anno_data.get('shapes', []):
        pts = np.array(shape['points'])
        # 将多边形点转化为外接矩形框 [x_min, y_min, x_max, y_max]
        gt_bboxes.append([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()])
    gt_bboxes = np.array(gt_bboxes)
    num_gt = len(gt_bboxes)

    # --- 模型预测 ---
    result = inference_detector(model, img_path)
    pred_instances = result.pred_instances
    
    # 过滤低分预测
    valid_mask = pred_instances.scores > SCORE_THR
    pred_bboxes = pred_instances.bboxes[valid_mask].cpu().numpy()
    num_pred = len(pred_bboxes)

    # --- IoU 匹配与统计 ---
    detected_gt_count = 0        # 成功检出的真实石头个数
    multiple_detected_count = 0  # 被切碎（一个真实石头对应多个预测）的个数

    if num_gt > 0 and num_pred > 0:
        iou_matrix = calculate_iou(gt_bboxes, pred_bboxes)
        
        for i in range(num_gt):
            # 找到和当前这个真实石头 IoU 大于阈值的所有预测框
            matched_preds = np.where(iou_matrix[i] > IOU_THR)[0]
            
            if len(matched_preds) >= 1:
                detected_gt_count += 1
            if len(matched_preds) > 1:
                multiple_detected_count += 1

    # 单张图片输出
    recall_rate = (detected_gt_count / num_gt * 100) if num_gt > 0 else 0
    print(f"{img_name[:25]:<25} | {num_gt:<9} | {num_pred:<9} | {recall_rate:>6.1f}%     | {multiple_detected_count:<5}")

    # 全局累加
    total_gt += num_gt
    total_pred_valid += num_pred
    total_detected += detected_gt_count
    total_multiple_detected += multiple_detected_count

print("-" * 75)
print("\n📊 【全局汇总报告】")
print(f"总计人工标注石头 (GT):     {total_gt} 块")
print(f"模型总计有效预测数:        {total_pred_valid} 块")
print(f"成功匹配/检出石头数:       {total_detected} 块")
print(f"👉 整体真实检出率 (Recall): {(total_detected / total_gt * 100) if total_gt > 0 else 0:.2f}%")
print(f"👉 发生【多次检出】的石头:  {total_multiple_detected} 块 (占比: {(total_multiple_detected / total_gt * 100) if total_gt > 0 else 0:.2f}%)")
print(f"注：如果 [有效预测数] 远大于 [人工标注]，且 [多次检出数] 很高，说明模型存在严重的边缘断裂(过分割)。")
