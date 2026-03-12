#!/bin/bash

export CUDA_VISIBLE_DEVICES="7,6,5,4"
export NCCL_P2P_DISABLE="1"
GPUS=4
OUTPUT_BASE="/group/chenjinming/SOTA_MMD/A-output"

CONFIGS=(
    "A-config/mask_rcnn_r50_24e.py"
    "A-config/mask_rcnn_r18_24e.py"
    "A-config/solo_24e.py"
    "A-config/yolact_24e.py"
)

for CONFIG in "${CONFIGS[@]}"; do
    MODEL_NAME=$(basename $CONFIG .py)
    WORK_DIR="${OUTPUT_BASE}/${MODEL_NAME}"
    mkdir -p ${WORK_DIR}
    
    echo "============================================="
    echo "🚀 开始正式训练: ${MODEL_NAME}"
    echo "============================================="
    
    # 核心：使用 tee 保存屏幕的所有终端输出
    bash tools/dist_train.sh ${CONFIG} ${GPUS} --work-dir ${WORK_DIR} 2>&1 | tee ${WORK_DIR}/terminal_full_log.txt
    
    echo ">>> ${MODEL_NAME} 训练结束，准备切换下一个模型..."
    sleep 5
done
echo "🎉 所有模型正式训练完毕！"
