import os
import json
import cv2
import numpy as np
from mmdet.apis import init_detector, inference_detector

# ================= 1. 配置路径 =================
config_file = '/group/chenjinming/SOTA_MMD/A-config/point_rend_r50_150e_aug.py'
checkpoint_file = '/group/chenjinming/SOTA_MMD/A-output/point_rend_r50_150e_aug/epoch_150.pth'
data_dir = '/group/chenjinming/Datas/test-img-json'

out_vis_dir = '/group/chenjinming/SOTA_MMD/A-predict/pointrend_smooth_vis'
out_txt_count = '/group/chenjinming/SOTA_MMD/A-predict/report_count_strict.txt'
out_txt_area = '/group/chenjinming/SOTA_MMD/A-predict/report_area_strict.txt'
os.makedirs(out_vis_dir, exist_ok=True)

# ================= 2. 参数调节 =================
SCORE_THR = 0.15
IOU_THR = 0.40
ALPHA = 0.45

# ================= 3. 辅助函数 =================
def calculate_iou(boxes1, boxes2):
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

def post_process_mask(mask_bool):
    mask_uint8 = (mask_bool * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask_bool, 0 
    largest_contour = max(contours, key=cv2.contourArea)
    clean_mask_uint8 = np.zeros_like(mask_uint8)
    cv2.drawContours(clean_mask_uint8, [largest_contour], -1, 255, thickness=cv2.FILLED)
    return clean_mask_uint8 > 0, cv2.contourArea(largest_contour)

# ================= 4. 初始化模型 =================
print("🚀 正在加载 PointRend 模型 (启用 1对1 严格匹配算法)...")
model = init_detector(config_file, checkpoint_file, device='cuda:0')
img_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

# ================= 5. 核心处理流程 =================
with open(out_txt_count, 'w', encoding='utf-8') as f_count, \
     open(out_txt_area, 'w', encoding='utf-8') as f_area:
    
    f_count.write(f"{'图片名称':<25} | {'实际个数(GT)':<12} | {'有效预测':<10} | {'成功匹配':<10} | {'真实检出率(Recall)'}\n")
    f_count.write("-" * 80 + "\n")
    
    f_area.write("📏 石头掩码面积误差报告 (严格 1对1 匹配计算)\n")
    f_area.write("=" * 100 + "\n")

    for img_name in img_files:
        img_path = os.path.join(data_dir, img_name)
        json_path = os.path.join(data_dir, os.path.splitext(img_name)[0] + '.json')
        img = cv2.imread(img_path)
        
        # --- 读取真实标注 ---
        gt_bboxes = []
        gt_areas = []
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as fj:
                anno_data = json.load(fj)
            for shape in anno_data.get('shapes', []):
                pts = np.array(shape['points'], dtype=np.float32)
                gt_areas.append(cv2.contourArea(pts))
                gt_bboxes.append([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()])
            gt_bboxes = np.array(gt_bboxes)
        num_gt = len(gt_bboxes)
        
        # --- 模型预测 ---
        result = inference_detector(model, img_path)
        scores = result.pred_instances.scores.cpu().numpy()
        masks = result.pred_instances.masks.cpu().numpy()
        bboxes = result.pred_instances.bboxes.cpu().numpy()
        
        valid_inds = np.where(scores > SCORE_THR)[0]
        pred_bboxes = bboxes[valid_inds]
        valid_scores = scores[valid_inds]
        num_pred = len(pred_bboxes)

        # 形态学清理，提取干净的像素掩码和面积
        cleaned_areas = []
        for i in valid_inds:
            _, clean_area = post_process_mask(masks[i])
            cleaned_areas.append(clean_area)

        # ---------------------------------------------------------
        # 👑 核心修复：1 对 1 严格二分图贪心匹配
        # ---------------------------------------------------------
        detected_gt_count = 0
        gt_matched = np.zeros(num_gt, dtype=bool)  # 记录真实石头是否已被占用
        gt_match_pred_idx = np.full(num_gt, -1)    # 记录真实石头被哪个预测框拿走了
        gt_match_iou = np.zeros(num_gt)

        if num_gt > 0 and num_pred > 0:
            iou_matrix = calculate_iou(gt_bboxes, pred_bboxes)
            # 让模型最自信的预测框优先挑选石头
            sort_inds = np.argsort(-valid_scores)
            
            for p_idx in sort_inds:
                best_gt_idx = -1
                best_iou = 0
                # 遍历所有真实石头，找一个没被占用且重合度最高的
                for g_idx in range(num_gt):
                    if not gt_matched[g_idx] and iou_matrix[g_idx, p_idx] > best_iou:
                        best_iou = iou_matrix[g_idx, p_idx]
                        best_gt_idx = g_idx
                
                # 如果最高重合度及格了，锁定这对外貌相同的对象
                if best_gt_idx >= 0 and best_iou > IOU_THR:
                    gt_matched[best_gt_idx] = True
                    gt_match_pred_idx[best_gt_idx] = p_idx
                    gt_match_iou[best_gt_idx] = best_iou
                    detected_gt_count += 1

        # 写入数量统计
        recall_pct = (detected_gt_count / num_gt * 100) if num_gt > 0 else 0
        f_count.write(f"{img_name[:25]:<25} | {num_gt:<14} | {num_pred:<12} | {detected_gt_count:<10} | {recall_pct:>6.1f}%\n")

        # --- 写入面积误差 ---
        f_area.write(f"\n▶ 图片: {img_name} (有效标注: {num_gt} | 有效预测: {num_pred})\n")
        f_area.write("-" * 80 + "\n")
        
        for i in range(num_gt):
            gt_area = gt_areas[i]
            p_idx = gt_match_pred_idx[i]
            if p_idx != -1: # 匹配成功
                pred_area = cleaned_areas[p_idx]
                iou_val = gt_match_iou[i]
                diff = pred_area - gt_area
                diff_pct = (diff / gt_area * 100) if gt_area > 0 else 0
                f_area.write(f"  [1对1精准匹配] 石头_{i:03d} | IoU: {iou_val:.2f} | 真实: {gt_area:>6.0f}px | 预测: {pred_area:>6.0f}px | 误差: {diff:>+6.0f}px ({diff_pct:>+6.1f}%)\n")
            else:
                f_area.write(f"  [彻底漏检]     石头_{i:03d} | 真实: {gt_area:>6.0f}px | 无匹配预测\n")

print(f"🎉 重新匹配完成！水分已全部挤干！")
print(f"👉 真实检出报告: {out_txt_count}")
print(f"👉 真实误差报告: {out_txt_area}")
