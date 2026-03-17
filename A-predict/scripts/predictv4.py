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
out_txt_count = '/group/chenjinming/SOTA_MMD/A-predict/report_count_diff.txt'
out_txt_area = '/group/chenjinming/SOTA_MMD/A-predict/report_area_error.txt'
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
    """【核心后处理】平滑边缘，消除散落点和内部空洞"""
    mask_uint8 = (mask_bool * 255).astype(np.uint8)
    
    # 1. 闭运算：填补内部小洞，平滑外边缘 (使用 5x5 的椭圆核)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    
    # 2. 最大连通域提取：消除外部散落的碎点
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask_bool, 0 
    
    # 找到面积最大的那个轮廓
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 重新画一个干净的掩码
    clean_mask_uint8 = np.zeros_like(mask_uint8)
    cv2.drawContours(clean_mask_uint8, [largest_contour], -1, 255, thickness=cv2.FILLED)
    
    return clean_mask_uint8 > 0, cv2.contourArea(largest_contour)

# ================= 4. 初始化模型 =================
print("🚀 正在加载 PointRend 模型 (附带形态学平滑滤镜)...")
model = init_detector(config_file, checkpoint_file, device='cuda:0')
img_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

# ================= 5. 核心处理流程 =================
with open(out_txt_count, 'w', encoding='utf-8') as f_count, \
     open(out_txt_area, 'w', encoding='utf-8') as f_area:
    
    # 写表头
    f_count.write(f"{'图片名称':<25} | {'实际个数(GT)':<12} | {'预测个数':<10} | {'个数差值':<8} | {'差值百分比'}\n")
    f_count.write("-" * 80 + "\n")
    
    f_area.write("📏 石头掩码面积误差报告 (已应用形态学去噪平滑)\n")
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
                gt_area = cv2.contourArea(pts)
                gt_areas.append(gt_area)
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
        num_pred = len(pred_bboxes)

        # 写入个数统计报告 (修复了格式化字符串报错)
        diff_count = num_pred - num_gt
        diff_count_pct = (diff_count / num_gt * 100) if num_gt > 0 else 0
        diff_count_str = f"{diff_count:+d}"  # 强转为带符号字符串
        f_count.write(f"{img_name[:25]:<25} | {num_gt:<14} | {num_pred:<12} | {diff_count_str:<8} | {diff_count_pct:>+7.1f}%\n")

        # --- 渲染与面积对比逻辑 ---
        f_area.write(f"\n▶ 图片: {img_name}\n")
        f_area.write("-" * 80 + "\n")
        
        color_mask_layer = np.zeros_like(img)
        stone_pixels_idx = np.zeros(img.shape[:2], dtype=bool)
        
        cleaned_masks = []
        cleaned_areas = []

        for i in valid_inds:
            raw_mask = masks[i]
            # 🧹 执行形态学清理：剔除散落点，平滑边缘
            clean_mask, clean_area = post_process_mask(raw_mask)
            
            cleaned_masks.append(clean_mask)
            cleaned_areas.append(clean_area)
            
            color = np.random.randint(50, 256, (3,)).tolist()
            color_mask_layer[clean_mask] = color
            stone_pixels_idx = stone_pixels_idx | clean_mask
            
            # 画平滑后的边缘
            contours, _ = cv2.findContours(clean_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, color, 1)

        img[stone_pixels_idx] = img[stone_pixels_idx] * (1 - ALPHA) + color_mask_layer[stone_pixels_idx] * ALPHA
        cv2.imwrite(os.path.join(out_vis_dir, img_name), img)

        # --- IoU 匹配与面积误差写入 ---
        if num_gt > 0 and num_pred > 0:
            iou_matrix = calculate_iou(gt_bboxes, pred_bboxes)
            for i in range(num_gt):
                best_pred_idx = np.argmax(iou_matrix[i])
                best_iou = iou_matrix[i, best_pred_idx]
                gt_area = gt_areas[i]

                if best_iou > IOU_THR:
                    pred_area = cleaned_areas[best_pred_idx]
                    diff = pred_area - gt_area
                    diff_pct = (diff / gt_area * 100) if gt_area > 0 else 0
                    f_area.write(f"  [匹配] 石头_{i:03d} | 真实: {gt_area:>6.0f}px | 预测: {pred_area:>6.0f}px | 误差: {diff:>+6.0f}px ({diff_pct:>+6.1f}%)\n")
                else:
                    f_area.write(f"  [漏检] 石头_{i:03d} | 真实: {gt_area:>6.0f}px | 预测未匹配\n")

print(f"🎉 全部处理完成！")
print(f"👉 数量报告: {out_txt_count}")
print(f"👉 面积报告: {out_txt_area}")
print(f"👉 平滑图片: {out_vis_dir}")
