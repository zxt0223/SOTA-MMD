import os
from mmdet.apis import DetInferencer

# ================= 1. 配置路径 =================
# 指向我们刚刚跑完的 150 轮强力增强版配置文件
config_file = '/group/chenjinming/SOTA_MMD/A-config/cascade_mask_rcnn_r50_150e_aug.py'
# 指向第 150 轮的最终权重 (如果因为设置了验证集保存了 best_xxx.pth，请根据实际文件名修改)
checkpoint_file = '/group/chenjinming/SOTA_MMD/A-output/cascade_mask_rcnn_r50_150e_aug/epoch_150.pth'

# 预测图片来源 (你指定的测试图文件夹)
img_dir = '/group/chenjinming/Datas/test-img'
# 新的输出文件夹，防止覆盖之前的 24e 结果
out_dir = '/group/chenjinming/SOTA_MMD/A-predict/cascade_150e_aug_results'

# ================= 2. 初始化推理器 =================
print("🚀 正在加载 150 轮极限增强版 Cascade Mask R-CNN 模型，请稍候...")
# 使用 0 号显卡进行预测
inferencer = DetInferencer(model=config_file, weights=checkpoint_file, device='cuda:0')

# ================= 3. 执行批量推理 =================
print(f"🔍 开始对 {img_dir} 内的图片进行预测...")
print(f"📁 结果将保存在: {out_dir}")

# no_save_vis=False: 保存画好掩码的图片用于肉眼观察
# no_save_pred=False: 保存 JSON 坐标用于后续物理计算
inferencer(inputs=img_dir, out_dir=out_dir, no_save_vis=False, no_save_pred=False)

print("✅ 预测全部完成！请前往文件夹查看结果。")
