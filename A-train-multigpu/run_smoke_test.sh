#!/bin/bash

# ==========================================
# 1. 指定使用的 GPU 物理编号 (2号和3号)
export CUDA_VISIBLE_DEVICES="2,3"

# 2. 【关键修复】禁用 P2P，防止特定拓扑(如SYS连接)下 NCCL 死锁
export NCCL_P2P_DISABLE="1"

# 3. 对应的 GPU 数量为 2
GPUS=2
# ==========================================

OUTPUT_BASE="A-output"

echo "============================================="
echo "开始执行 MMDetection 冒烟测试 (1 Epoch) "
echo "正在使用的显卡编号: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NCCL_P2P_DISABLE 状态: ${NCCL_P2P_DISABLE}"
echo "============================================="

# 定义需要测试的配置列表
CONFIGS=(
    "A-config/mask_rcnn_r50_smoke.py"
    "A-config/mask_rcnn_r18_smoke.py"
    "A-config/solo_smoke.py"
    "A-config/yolact_smoke.py"
)

for CONFIG in "${CONFIGS[@]}"; do
    MODEL_NAME=$(basename $CONFIG .py)
    WORK_DIR="${OUTPUT_BASE}/${MODEL_NAME}"
    
    echo ">>> 正在测试: ${MODEL_NAME}"
    
    # 启动分布式训练
    bash tools/dist_train.sh ${CONFIG} ${GPUS} --work-dir ${WORK_DIR}
    
    # 检查是否报错
    if [ $? -eq 0 ]; then
        echo ">>> [成功] ${MODEL_NAME} 测试通过！结果保存在 ${WORK_DIR}"
    else
        echo ">>> [失败] ${MODEL_NAME} 测试报错，请向上翻阅检查报错原因！"
        exit 1 # 一旦报错立刻停止
    fi
    echo "---------------------------------------------"
done

echo "所有冒烟测试已完成！"
