import os
import cv2
import numpy as np
from mmdet.apis import init_detector, inference_detector

# ================= 1. 配置路径 =================
config_file = '/group/chenjinming/SOTA_MMD/A-config/point_rend_r50_150e_aug.py'
checkpoint_file = '/group/chenjinming/SOTA_MMD/A-output/point_rend_r50_150e_aug/epoch_150.pth'
img_dir = '/group/chenjinming/Datas/test-img'
# 创建专门存放高清自定义可视化图片的文件夹
out_dir = '/group/chenjinming/SOTA_MMD/A-predict/pointrend_custom_vis'
os.makedirs(out_dir, exist_ok=True)

# ================= 2. 视觉参数调节 =================
SCORE_THR = 0.15
ALPHA = 0.35  # 掩码透明度：0.35 代表颜色占 35%，原图底色占 65%

# ================= 3. 加载模型 =================
print("🚀 正在加载 PointRend 终极版模型...")
model = init_detector(config_file, checkpoint_file, device='cuda:0')

img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
print(f"🔍 开始生成极致观感的预测图，共 {len(img_files)} 张...")

# ================= 4. 自定义绘画循环 =================
for img_name in img_files:
    img_path = os.path.join(img_dir, img_name)
    img = cv2.imread(img_path)
    
    # 提取预测结果
    result = inference_detector(model, img_path)
    pred_instances = result.pred_instances
    
    scores = pred_instances.scores.cpu().numpy()
    masks = pred_instances.masks.cpu().numpy()
    
    # 筛选出大于阈值的石头
    valid_inds = np.where(scores > SCORE_THR)[0]
    
    # 创建一个纯黑的彩色图层，用于画半透明蒙版
    color_mask_layer = np.zeros_like(img)
    # 创建一个布尔矩阵，记录哪些像素有石头，以免把纯黑背景也混合了
    stone_pixels_idx = np.zeros(img.shape[:2], dtype=bool)

    for i in valid_inds:
        mask = masks[i]
        
        # 🎨 核心1：为每一块独立的石头生成随机的亮丽颜色 (B, G, R)
        # 规避掉太暗的颜色 (从 50~255 中随机)
        color = np.random.randint(50, 256, (3,)).tolist()
        
        # 将随机颜色涂在对应的石头掩码区域
        color_mask_layer[mask] = color
        stone_pixels_idx = stone_pixels_idx | mask
        
        # 🔪 核心2：使用 OpenCV 提取并绘制 1 像素宽的锐利边缘
        # 这能极致地展现 PointRend 切开粘连石头的“刀工”
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, color, 1)

    # 👻 核心3：Alpha 半透明混合 (只混合有石头的区域，保持原图背景清晰)
    img[stone_pixels_idx] = img[stone_pixels_idx] * (1 - ALPHA) + color_mask_layer[stone_pixels_idx] * ALPHA
    
    # 保存结果
    out_path = os.path.join(out_dir, img_name)
    cv2.imwrite(out_path, img)
    print(f"✅ 已保存: {img_name} (独立检出 {len(valid_inds)} 块石头)")

print(f"🎉 全部处理完成！请前往 {out_dir} 查看结果。")
