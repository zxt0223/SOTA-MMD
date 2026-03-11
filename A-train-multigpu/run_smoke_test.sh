#!/bin/bash
# 进入 SOTA_MMD 根目录执行
cd /group/chenjinming/SOTA_MMD

# 设定使用的 GPU 数量 (我们先用 2 张 4090 测试 NCCL 通信即可)
GPUS=2
CONFIG="A-config/mask_rcnn_r50_smoke_test.py"
# 将日志和权重定向到我们专属的目录
WORK_DIR="A-train-multigpu/work_dirs/smoke_test"

# 调用官方分布式启动脚本
bash tools/dist_train.sh $CONFIG $GPUS --work-dir $WORK_DIR