#!/bin/bash
# 进入 SOTA_MMD 根目录
cd /group/chenjinming/SOTA_MMD

# 设定使用 2 张空闲的显卡进行通信和流转测试（例如 6 和 7 号卡）
export CUDA_VISIBLE_DEVICES=6,7
GPUS=2

echo "========================================"
echo "🚀 开始 Mask R-CNN 冒烟测试..."
echo "========================================"
bash tools/dist_train.sh A-config/mask_rcnn_smoke.py $GPUS --work-dir A-train-multigpu/work_dirs/smoke_mask_rcnn

echo "========================================"
echo "🚀 开始 SOLO 冒烟测试..."
echo "========================================"
bash tools/dist_train.sh A-config/solo_smoke.py $GPUS --work-dir A-train-multigpu/work_dirs/smoke_solo

echo "========================================"
echo "🚀 开始 YOLACT 冒烟测试..."
echo "========================================"
bash tools/dist_train.sh A-config/yolact_smoke.py $GPUS --work-dir A-train-multigpu/work_dirs/smoke_yolact

echo "========================================"
echo "✅ 所有模型的冒烟测试脚本已执行完毕！"
echo "请检查上方日志是否有异常报错。"
echo "========================================"