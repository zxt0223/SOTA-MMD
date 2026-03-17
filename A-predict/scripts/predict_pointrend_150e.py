import os
from mmdet.apis import DetInferencer

# ================= 1. 配置路径 =================
config_file = '/group/chenjinming/SOTA_MMD/A-config/point_rend_r50_150e_aug.py'
checkpoint_file = '/group/chenjinming/SOTA_MMD/A-output/point_rend_r50_150e_aug/epoch_150.pth'
img_dir = '/group/chenjinming/Datas/test-img'
# 建立全新的输出文件夹，保存最高清的 Mask
out_dir = '/group/chenjinming/SOTA_MMD/A-predict/pointrend_150e_aug_results'

# ================= 2. 初始化推理器 =================
print("🚀 正在加载 PointRend 终极版模型 (边缘重采样)...")
inferencer = DetInferencer(model=config_file, weights=checkpoint_file, device='cuda:0')

# ================= 3. 执行批量推理 =================
print(f"🔍 开始生成高精度掩码预测图...")
# 保持和评估时一样的 0.15 阈值，释放最大的检出率
inferencer(inputs=img_dir, out_dir=out_dir, pred_score_thr=0.15, no_save_vis=False, no_save_pred=False)

print("✅ 预测全部完成！请前往 pointrend_150e_aug_results/vis 文件夹查看极致边缘效果。")
