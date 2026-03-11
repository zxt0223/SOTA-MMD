#!/bin/bash
cd /group/chenjinming/SOTA_MMD

# 创建存放运行日志的文件夹
mkdir -p A-train-multigpu/work_dirs/logs_parallel

echo "========================================"
echo "🚀 启动三线并发 SOTA 训练任务..."
echo "========================================"

# 1. 训练 Vanilla Mask R-CNN (挂载在 GPU 1)
export CUDA_VISIBLE_DEVICES=1
nohup python tools/train.py A-config/mask_rcnn_vanilla_24e.py --work-dir A-train-multigpu/work_dirs/vanilla_mask_rcnn > A-train-multigpu/work_dirs/logs_parallel/mask_rcnn.log 2>&1 &
PID1=$!
echo "✅ [GPU 1] Vanilla Mask R-CNN 已启动 (PID: $PID1)"

# 2. 训练 抢救版 SOLO (挂载在 GPU 2)
export CUDA_VISIBLE_DEVICES=2
nohup python tools/train.py A-config/solo_24e.py --work-dir A-train-multigpu/work_dirs/fixed_solo > A-train-multigpu/work_dirs/logs_parallel/solo.log 2>&1 &
PID2=$!
echo "✅ [GPU 2] SOLO 已启动 (PID: $PID2)"

# 3. 训练 抢救版 YOLACT (挂载在 GPU 3)
export CUDA_VISIBLE_DEVICES=3
nohup python tools/train.py A-config/yolact_24e.py --work-dir A-train-multigpu/work_dirs/fixed_yolact > A-train-multigpu/work_dirs/logs_parallel/yolact.log 2>&1 &
PID3=$!
echo "✅ [GPU 3] YOLACT 已启动 (PID: $PID3)"

echo "========================================"
echo "🎉 恭喜！三个模型正在各自的显卡上疯狂运算！"
echo "你可以随时输入以下命令之一来偷看训练进度："
echo "tail -f A-train-multigpu/work_dirs/logs_parallel/mask_rcnn.log"
echo "tail -f A-train-multigpu/work_dirs/logs_parallel/solo.log"
echo "tail -f A-train-multigpu/work_dirs/logs_parallel/yolact.log"