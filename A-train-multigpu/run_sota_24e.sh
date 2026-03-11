#!/bin/bash
cd /group/chenjinming/SOTA_MMD

# 征用 4 张显卡跑正式实验，速度拉满
export CUDA_VISIBLE_DEVICES=4,5,6,7
GPUS=4

echo "🚀 [1/3] 开始训练 Mask R-CNN (24 Epochs)..."
bash tools/dist_train.sh A-config/mask_rcnn_24e.py $GPUS --work-dir A-train-multigpu/work_dirs/sota_mask_rcnn

echo "🚀 [2/3] 开始训练 SOLO (24 Epochs)..."
bash tools/dist_train.sh A-config/solo_24e.py $GPUS --work-dir A-train-multigpu/work_dirs/sota_solo

echo "🚀 [3/3] 开始训练 YOLACT (24 Epochs)..."
bash tools/dist_train.sh A-config/yolact_24e.py $GPUS --work-dir A-train-multigpu/work_dirs/sota_yolact

echo "✅ 大满贯！所有 SOTA 模型训练完成，请前往 work_dirs 查看成绩单！"