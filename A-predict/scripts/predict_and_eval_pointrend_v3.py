import os
import json
import cv2
import numpy as np
from mmdet.apis import init_detector, inference_detector

# ================= 1. 配置路径 =================
config_file = '/group/chenjinming/SOTA_MMD/A-config/point_rend_r50_150e_aug.py'
# 💡 强烈建议：如果你在输出文件夹里看到了 best_xxx.pth，请替换掉下面的 150.pth
checkpoint_file = '/group/chenjinming/SOTA_MMD/A-output/point_rend_r50_150e_aug/epoch_150.pth'
data_dir = '/group/chenjinming/Datas/test-img-json'

out_vis_dir = '/group/chenjinming/SOTA_MMD/A-predict/pointrend_final_vis'
out_txt_file = '/group/chenjinming/SOTA_MMD/A-predict/pointrend_detailed_report.txt'
os.makedirs(out_vis_dir, exist_ok=True)

# ================= 2. 参数调节 =================
SCORE_THR = 0.15
IOU_THR = 0.40
ALPHA = 0.45

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
total_pred_valid = 0

# ================= 5. 核心处理流程 =================
with open(out_txt_file, 'w', encoding='utf-8') as f_out:
    f_out.write(f"🔍 PointRend 极细粒度测试报告 (透明度: {ALPHA}, 阈值: {SCORE_THR})\n")
    f_out.write("=" * 100 + "\n\n")
    print("📝 正在进行像素级对比，请稍候...")

    for img_name in img_files:
        img_path = os.path.join(data_dir, img_name)
        json_path = os.path.join(data_dir, os.path.splitext(img_name)[0] + '.json')
        img = cv2.imread(img_path)
        
        # --- 1. 读取 JSON 标注并计算真实面积 ---
        gt_bboxes = []
        gt_areas = []
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as fj:
                anno_data = json.load(fj)
            for shape in anno_data.get('shapes', []):
                pts = np.array(shape['points'], dtype=np.float32)
                # 计算人工标注多边形的真实面积(像素)
                gt_area = cv2.contourArea(pts)
                gt_areas.append(gt_area)
                
                # 提取包围框用于 IoU 匹配
                gt_bboxes.append([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()])
            gt_bboxes = np.array(gt_bboxes)

        num_gt = len(gt_bboxes)
        
        # --- 2. 模型预测 ---
        result = inference_detector(model, img_path)
        scores = result.pred_instances.scores.cpu().numpy()
        masks = result.pred_instances.masks.cpu().numpy()
        bboxes = result.pred_instances.bboxes.cpu().numpy()
        
        valid_inds = np.where(scores > SCORE_THR)[0]
        pred_bboxes = bboxes[valid_inds]
        num_pred = len(pred_bboxes)

        # 渲染出图逻辑
        color_mask_layer = np.zeros_like(img)
        stone_pixels_idx = np.zeros(img.shape[:2], dtype=bool)
        for i in valid_inds:
            mask = masks[i]
            color = np.random.randint(50, 256, (3,)).tolist()
            color_mask_layer[mask] = color
            stone_pixels_idx = stone_pixels_idx | mask
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, color, 1)

        img[stone_pixels_idx] = img[stone_pixels_idx] * (1 - ALPHA) + color_mask_layer[stone_pixels_idx] * ALPHA
        cv2.imwrite(os.path.join(out_vis_dir, img_name), img)

        # --- 3. 像素面积误差对比 (写入 TXT) ---
        f_out.write(f"▶ 图片: {img_name} (人工标注: {num_gt}块 | 预测: {num_pred}块)\n")
        f_out.write("-" * 80 + "\n")
        
        detected_gt_count = 0        

        if num_gt > 0 and num_pred > 0:
            iou_matrix = calculate_iou(gt_bboxes, pred_bboxes)
            for i in range(num_gt):
                # 找到与当前真实石头 IoU 最高的预测框
                best_pred_idx = np.argmax(iou_matrix[i])
                best_iou = iou_matrix[i, best_pred_idx]
                
                gt_area = gt_areas[i]

                if best_iou > IOU_THR:
                    detected_gt_count += 1
                    # 获取模型预测出的这个石头的实际像素面积
                    pred_mask = masks[valid_inds[best_pred_idx]]
                    pred_area = np.sum(pred_mask)
                    
                    diff = pred_area - gt_area
                    diff_pct = (diff / gt_area * 100) if gt_area > 0 else 0
                    f_out.write(f"  [匹配成功] 石头_{i:03d} | IoU: {best_iou:.2f} | 真实面积: {gt_area:>6.0f} px | 预测面积: {pred_area:>6.0f} px | 面积误差: {diff:>+6.0f} px ({diff_pct:>+6.1f}%)\n")
                else:
                    f_out.write(f"  [漏检/偏离] 石头_{i:03d} | 最大IoU: {best_iou:.2f} | 真实面积: {gt_area:>6.0f} px | 预测未匹配\n")
        f_out.write("\n")

        total_gt += num_gt
        total_pred_valid += num_pred
        total_detected += detected_gt_count

    # --- 4. 写入全局汇总 ---
    summary = "=" * 100 + "\n"
    summary += "📊 【全局汇总报告】\n"
    summary += f"总计人工标注石头 (GT):     {total_gt} 块\n"
    summary += f"模型总计有效预测数:        {total_pred_valid} 块\n"
    summary += f"成功匹配/检出石头数:       {total_detected} 块\n"
    summary += f"👉 整体真实检出率 (Recall): {(total_detected / total_gt * 100) if total_gt > 0 else 0:.2f}%\n"
    f_out.write(summary)

print(f"🎉 全部处理完成！详细误差报告保存在: {out_txt_file}")
