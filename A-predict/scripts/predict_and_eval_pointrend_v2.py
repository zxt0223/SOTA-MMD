import os
import json
import cv2
import numpy as np
from mmdet.apis import init_detector, inference_detector

# ================= 1. 配置路径 =================
config_file = '/group/chenjinming/SOTA_MMD/A-config/point_rend_r50_150e_aug.py'
checkpoint_file = '/group/chenjinming/SOTA_MMD/A-output/point_rend_r50_150e_aug/epoch_150.pth'
data_dir = '/group/chenjinming/Datas/test-img-json'

# 新的输出文件夹和 TXT 报告文件
out_vis_dir = '/group/chenjinming/SOTA_MMD/A-predict/pointrend_final_vis'
out_txt_file = '/group/chenjinming/SOTA_MMD/A-predict/pointrend_evaluation_report.txt'
os.makedirs(out_vis_dir, exist_ok=True)

# ================= 2. 参数调节 =================
SCORE_THR = 0.15
IOU_THR = 0.40
ALPHA = 0.45  # 【关键修改】透明度设为 45%

# ================= 3. IoU 计算函数 =================
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

# ================= 4. 初始化模型 =================
print("🚀 正在加载 PointRend 终极版模型...")
model = init_detector(config_file, checkpoint_file, device='cuda:0')
img_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

total_gt = 0
total_detected = 0
total_multiple = 0
total_pred_valid = 0

# ================= 5. 开始批量处理并写入 TXT =================
with open(out_txt_file, 'w', encoding='utf-8') as f_out:
    header = f"🔍 PointRend 终极测试报告 (透明度: {ALPHA}, 阈值: {SCORE_THR})\n"
    header += "=" * 80 + "\n"
    header += f"{'图片名称':<25} | {'人工标注(GT)':<10} | {'有效预测':<10} | {'检出率(Recall)':<14} | {'多次检出数':<10}\n"
    header += "-" * 80 + "\n"
    f_out.write(header)
    print("📝 正在进行预测、对比并生成可视化图片，请稍候...")

    for img_name in img_files:
        img_path = os.path.join(data_dir, img_name)
        json_path = os.path.join(data_dir, os.path.splitext(img_name)[0] + '.json')
        img = cv2.imread(img_path)
        
        # --- 1. 读取 JSON 标注 ---
        num_gt = 0
        gt_bboxes = []
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as fj:
                anno_data = json.load(fj)
            for shape in anno_data.get('shapes', []):
                pts = np.array(shape['points'])
                gt_bboxes.append([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()])
            gt_bboxes = np.array(gt_bboxes)
            num_gt = len(gt_bboxes)

        # --- 2. 模型预测与可视化 ---
        result = inference_detector(model, img_path)
        scores = result.pred_instances.scores.cpu().numpy()
        masks = result.pred_instances.masks.cpu().numpy()
        bboxes = result.pred_instances.bboxes.cpu().numpy()
        
        valid_inds = np.where(scores > SCORE_THR)[0]
        pred_bboxes = bboxes[valid_inds]
        num_pred = len(pred_bboxes)

        color_mask_layer = np.zeros_like(img)
        stone_pixels_idx = np.zeros(img.shape[:2], dtype=bool)

        for i in valid_inds:
            mask = masks[i]
            color = np.random.randint(50, 256, (3,)).tolist()
            color_mask_layer[mask] = color
            stone_pixels_idx = stone_pixels_idx | mask
            
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, color, 1)

        # 45% 透明度混合
        img[stone_pixels_idx] = img[stone_pixels_idx] * (1 - ALPHA) + color_mask_layer[stone_pixels_idx] * ALPHA
        cv2.imwrite(os.path.join(out_vis_dir, img_name), img)

        # --- 3. IoU 匹配与统计 ---
        detected_gt_count = 0        
        multiple_detected_count = 0  

        if num_gt > 0 and num_pred > 0:
            iou_matrix = calculate_iou(gt_bboxes, pred_bboxes)
            for i in range(num_gt):
                matched_preds = np.where(iou_matrix[i] > IOU_THR)[0]
                if len(matched_preds) >= 1:
                    detected_gt_count += 1
                if len(matched_preds) > 1:
                    multiple_detected_count += 1

        recall_rate = (detected_gt_count / num_gt * 100) if num_gt > 0 else 0
        line_str = f"{img_name[:25]:<25} | {num_gt:<12} | {num_pred:<12} | {recall_rate:>6.1f}%         | {multiple_detected_count:<10}\n"
        f_out.write(line_str)

        total_gt += num_gt
        total_pred_valid += num_pred
        total_detected += detected_gt_count
        total_multiple += multiple_detected_count

    # --- 4. 写入全局汇总 ---
    summary = "=" * 80 + "\n"
    summary += "📊 【全局汇总报告】\n"
    summary += f"总计人工标注石头 (GT):     {total_gt} 块\n"
    summary += f"模型总计有效预测数:        {total_pred_valid} 块\n"
    summary += f"成功匹配/检出石头数:       {total_detected} 块\n"
    summary += f"👉 整体真实检出率 (Recall): {(total_detected / total_gt * 100) if total_gt > 0 else 0:.2f}%\n"
    summary += f"👉 发生【多次检出】的石头:  {total_multiple} 块 (占比: {(total_multiple / total_gt * 100) if total_gt > 0 else 0:.2f}%)\n"
    summary += "=" * 80 + "\n"
    f_out.write(summary)

print(f"🎉 全部处理完成！\n👉 图片保存在: {out_vis_dir}\n👉 详细报告保存在: {out_txt_file}")
