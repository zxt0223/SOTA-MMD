import os
from mmdet.apis import DetInferencer

# ================= 1. 配置路径 =================
config_file = '/group/chenjinming/SOTA_MMD/A-config/cascade_mask_rcnn_r50_24e.py'
# 注意：这里使用的是你提供的 epoch_3.pth。如果你后续训练完了完整的 24 轮，记得改成 epoch_24.pth 或 best.pth
checkpoint_file = '/group/chenjinming/SOTA_MMD/A-output/cascade_mask_rcnn_r50_24e/epoch_3.pth'
img_dir = '/group/chenjinming/Datas/test-img'
out_dir = '/group/chenjinming/SOTA_MMD/A-predict/cascade_results'

# ================= 2. 初始化推理器 =================
print("🚀 正在加载 Cascade Mask R-CNN 模型，请稍候...")
# device='cuda:0' 表示使用第一张显卡进行预测。如果工厂电脑没有显卡，这里可以改为 'cpu'
inferencer = DetInferencer(model=config_file, weights=checkpoint_file, device='cuda:0')

# ================= 3. 执行批量推理 =================
print(f"🔍 开始对 {img_dir} 内的图片进行预测...")
print(f"📁 结果将保存在: {out_dir}")

# no_save_vis=False: 保存画好框和掩码的图片
# no_save_pred=False: 保存包含坐标和掩码点信息的 JSON 文件 (后续计算体积会用到)
inferencer(inputs=img_dir, out_dir=out_dir, no_save_vis=False, no_save_pred=False)

print("✅ 预测全部完成！请前往文件夹查看结果。")
