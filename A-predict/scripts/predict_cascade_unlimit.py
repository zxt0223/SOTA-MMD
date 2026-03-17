import os
from mmdet.apis import DetInferencer

config_file = '/group/chenjinming/SOTA_MMD/A-config/cascade_150e_test_unlimit.py'
checkpoint_file = '/group/chenjinming/SOTA_MMD/A-output/cascade_mask_rcnn_r50_150e_aug/epoch_150.pth'
img_dir = '/group/chenjinming/Datas/test-img'
# 建立全新的输出文件夹，方便对比
out_dir = '/group/chenjinming/SOTA_MMD/A-predict/cascade_150e_unlimit_results'

print("🚀 正在加载 [解除 100 个上限封印] 的 Cascade 模型...")
inferencer = DetInferencer(model=config_file, weights=checkpoint_file, device='cuda:0')

print(f"🔍 开始高密度预测，请稍候...")
# 【关键】pred_score_thr 默认是 0.3，我们降到 0.15，把被阴影遮挡、得分偏低的真实小石头放出来
inferencer(inputs=img_dir, out_dir=out_dir, pred_score_thr=0.15, no_save_vis=False, no_save_pred=False)

print("✅ 预测全部完成！封印已解除，请前往 cascade_150e_unlimit_results 文件夹查看。")
