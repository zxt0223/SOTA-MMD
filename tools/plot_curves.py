import json
import glob
import os
import matplotlib.pyplot as plt

def get_log_data(work_dir):
    # 自动寻找最新的 json 格式训练日志
    json_files = glob.glob(os.path.join(work_dir, '*/vis_data/scalars.json'))
    if not json_files:
        return None, None
    latest_json = sorted(json_files, key=os.path.getmtime)[-1]
    
    epochs_map = []
    map_vals = []
    epochs_loss = []
    loss_vals = []
    
    with open(latest_json, 'r') as f:
        for line in f:
            data = json.loads(line)
            # 提取 mAP
            if 'coco/bbox_mAP' in data:
                epochs_map.append(data['step'])
                map_vals.append(data['coco/bbox_mAP'])
            # 提取 Loss
            elif 'loss' in data and 'step' in data:
                epochs_loss.append(data['step'])
                loss_vals.append(data['loss'])
                
    return (epochs_loss, loss_vals), (epochs_map, map_vals)

def main():
    models = {
        "ResNet-50 (Baseline)": "A-output/mask_rcnn_r50_24e",
        "LEGNet (Ours)": "A-output/mask_rcnn_legnet_24e"
    }
    
    plt.figure(figsize=(14, 6))
    
    # 1. 画 Loss 曲线
    plt.subplot(1, 2, 1)
    for name, path in models.items():
        loss_data, _ = get_log_data(path)
        if loss_data:
            # Loss 点太多了，我们每 50 个点采样一次让曲线平滑
            plt.plot(loss_data[0][::50], loss_data[1][::50], label=name, alpha=0.8)
    plt.title('Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 2. 画 mAP 曲线
    plt.subplot(1, 2, 2)
    for name, path in models.items():
        _, map_data = get_log_data(path)
        if map_data:
            plt.plot(map_data[0], map_data[1], marker='o', label=name)
    plt.title('Validation bbox_mAP')
    plt.xlabel('Epochs')
    plt.ylabel('mAP')
    plt.legend()
    plt.grid(True)
    
    out_path = "A-output/Vis_Results/Curve_Comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"✅ 对比曲线图已保存至: {out_path}")

if __name__ == "__main__":
    main()
