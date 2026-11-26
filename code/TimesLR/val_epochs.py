import os
import csv
import time
import glob
import joblib
import random
import configparser
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from Dataset import Dataset
from TimesFM_lora import TimesFMForClassification

# ========= 1. 配置加载与环境初始化 =========
config = configparser.ConfigParser()
config.read('config.ini')


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    print('Random seed :', seed)


my_seed = config.getint('general', 'my_seed')
set_seed(my_seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

path_to_files = config.get('general', 'path_to_files')
encoder = joblib.load(config.get('general', 'path_to_encoder'))
encoder_unknown = joblib.load(config.get('general', 'path_to_encoder_open'))
label_allow_list = encoder.classes_
open_list = encoder_unknown.classes_
batch_size = config.getint('general', 'batch_size_for_train')
num_classes = config.getint('general', 'num_classes')

# 🎛️【全新相似度评估空间选择开关】
# 可选值: 'feat' (使用64维空间特征值) 或 'logits' (使用分类别对数几率)
SIM_SPACE = 'feat'


# ========= 2. DataLoader 加载 (完全静态，移除 Shuffle) =========
def set_eval_dataloader():
    train_file_paths, train_labels = [], []
    val_file_paths, val_labels = [], []
    unknown_file_paths, unknown_labels = [], []

    for parent_path in glob.glob(path_to_files):
        label = os.path.basename(parent_path)

        if label not in label_allow_list:
            unknown_parent_path = os.path.join(parent_path, 'test/*')
            for files_path in glob.glob(unknown_parent_path):
                for file_path in glob.glob(os.path.join(files_path, '*.h5')):
                    unknown_file_paths.append(file_path)
                    unknown_labels.append(label)
        else:
            train_parent_path = os.path.join(parent_path, 'train/*')
            for files_path in glob.glob(train_parent_path):
                for file_path in glob.glob(os.path.join(files_path, '*.h5')):
                    train_file_paths.append(file_path)
                    train_labels.append(label)
            val_parent_path = os.path.join(parent_path, 'test/*')
            for files_path in glob.glob(val_parent_path):
                for file_path in glob.glob(os.path.join(files_path, '*.h5')):
                    val_file_paths.append(file_path)
                    val_labels.append(label)

    train_labels = encoder.transform(train_labels)
    val_labels = encoder.transform(val_labels)
    unknown_labels = encoder_unknown.transform(unknown_labels)

    print('train nums for MAV:', len(train_labels))
    print('val nums:', len(val_labels))
    print('unknown nums:', len(unknown_labels))

    train_set = Dataset(train_file_paths, train_labels)
    val_set = Dataset(val_file_paths, val_labels)
    unknown_set = Dataset(unknown_file_paths, unknown_labels)

    # 推理流全部关闭 shuffle，使用双倍 Batch 加速
    train_loader_eval = DataLoader(train_set, batch_size=batch_size * 2, shuffle=False, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=batch_size * 2, shuffle=False)
    unknown_loader = DataLoader(unknown_set, batch_size=batch_size * 2, shuffle=False)

    return train_loader_eval, val_loader, unknown_loader


# ========= 3. 初始化空白基础模型 =========
model = TimesFMForClassification(num_classes, device=device).to(device)
criterion = nn.CrossEntropyLoss().to(device)

if __name__ == '__main__':
    train_loader_eval, val_loader, unknown_loader = set_eval_dataloader()

    # 配置独立的重验证日志保存路径
    re_val_log_path = f"./re_validation_{SIM_SPACE}_report.csv"
    csv_file = open(re_val_log_path, 'w', newline='')
    fieldnames = ['epoch', 'val_loss', 'val_accuracy', 'val_acc_under_thresh',
                  'auto_threshold', 'total_unk_rejection_acc', 'open_set_auroc', 'epoch_time']
    writer_csv = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer_csv.writeheader()

    # 自动检索所有保存的二阶段模型权重文件
    stage2_model_folder = config.get('general', 'path_to_model_folder')
    ckpt_pattern = os.path.join(stage2_model_folder, "model_epoch_*.pth")
    ckpt_files = glob.glob(ckpt_pattern)

    if not ckpt_files:
        print(f"❌ 错误: 未能在路径 {stage2_model_folder} 下找到任何 stage2_epoch_*.pth 文件。")
        exit()

    # 按 epoch 数字大小升序排序
    ckpt_files.sort(key=lambda x: int(os.path.basename(x).split('_')[2].split('.')[0]))

    print(f"\n====== 找到 {len(ckpt_files)} 个 Epoch 权重，基于 【{SIM_SPACE.upper()}】 空间计算相似度 ======")

    for ckpt_path in ckpt_files:
        epoch = int(os.path.basename(ckpt_path).split('_')[2].split('.')[0])
        start_time = time.time()

        # 💥 加载当前 Epoch 权重
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # 切换到彻底的静态评估模式
        model.eval()

        # 根据用户选择的空间，动态分配类中心的张量大小
        # feat 对应 64维，logits 对应 闭集类别数
        eval_dim = 64 if SIM_SPACE == 'feat' else num_classes
        train_feature_sums = torch.zeros(num_classes, eval_dim).to(device)
        train_class_counts = torch.zeros(num_classes, 1).to(device)

        # ----------------------------------------------------------------
        # 1. 独立推理：无损重构当前选定空间 (feat/logits) 的 MAV
        # ----------------------------------------------------------------
        with torch.no_grad():
            for data, targets in train_loader_eval:
                inputs = data.float().to(device)
                targets = targets.to(device)

                # 💥 接收双返回值解包
                feat, logits = model(inputs)

                # 依据开关决定把谁拿出来做 MAV 空间重构
                target_vectors = feat if SIM_SPACE == 'feat' else logits

                ones = torch.ones(targets.size(0), 1).to(device)
                train_class_counts.scatter_add_(0, targets.view(-1, 1), ones)
                target_expanded = targets.view(-1, 1).expand(-1, target_vectors.size(1))
                train_feature_sums.scatter_add_(0, target_expanded, target_vectors)

        MAV = train_feature_sums / (train_class_counts + 1e-10)
        norm_mav = F.normalize(MAV, p=2, dim=1, eps=1e-10)

        # ----------------------------------------------------------------
        # 2. 独立推理：Val 验证集多分类性能测试与得分收集
        # ----------------------------------------------------------------
        val_loss_total, val_correct, val_total = 0, 0, 0
        all_val_max_scores = []
        all_val_predictions_correct = []

        with torch.no_grad():
            for data, targets in val_loader:
                inputs = data.float().to(device)
                targets = targets.to(device)

                # 💥 接收双返回值解包
                feat, logits = model(inputs)
                v_loss = criterion(logits, targets)

                val_loss_total += v_loss.item() * targets.size(0)
                _, predicted = torch.max(logits.data, 1)
                val_correct += (predicted == targets).sum().item()
                val_total += targets.size(0)

                # 提取目标评估向量进行矩阵点积
                eval_vectors = feat if SIM_SPACE == 'feat' else logits
                norm_eval_vectors = F.normalize(eval_vectors, p=2, dim=1, eps=1e-10)
                sim_matrix = torch.mm(norm_eval_vectors, norm_mav.t())
                max_sims, max_idxs = torch.max(sim_matrix, dim=1)

                all_val_max_scores.extend(max_sims.cpu().numpy())
                all_val_predictions_correct.extend((max_idxs == targets).cpu().numpy())

        avg_val_loss = val_loss_total / val_total
        val_accuracy = val_correct / val_total

        # ----------------------------------------------------------------
        # 3. 独立推理：Unknown 未知类测试与得分收集
        # ----------------------------------------------------------------
        all_unknown_max_scores = []
        unknown_class_sims = defaultdict(list)

        with torch.no_grad():
            for data, targets in unknown_loader:
                inputs = data.float().to(device)

                # 💥 接收双返回值解包
                feat, logits = model(inputs)

                eval_vectors = feat if SIM_SPACE == 'feat' else logits
                norm_eval_vectors = F.normalize(eval_vectors, p=2, dim=1, eps=1e-10)

                cos_sim_matrix = torch.mm(norm_eval_vectors, norm_mav.t())
                max_sims, _ = torch.max(cos_sim_matrix, dim=1)

                max_sims_cpu = max_sims.detach().cpu().numpy()
                targets_cpu = targets.detach().cpu().numpy()

                all_unknown_max_scores.extend(max_sims_cpu)
                for k in range(targets.size(0)):
                    unknown_class_sims[targets_cpu[k].item()].append(max_sims_cpu[k].item())

        # ----------------------------------------------------------------
        # 4. 数理推演：解算全局自适应黄金阈值与学术指标
        # ----------------------------------------------------------------
        val_scores_arr = np.array(all_val_max_scores)
        unk_scores_arr = np.array(all_unknown_max_scores)
        val_correct_mask = np.array(all_val_predictions_correct)

        threshold_grid = np.linspace(0.0, 1.0, 1000)
        best_threshold = 0.0
        min_diff = float('inf')

        eps = 1e-12
        total_unknown = len(unk_scores_arr) + eps
        total_known = len(val_scores_arr) + eps

        for th in threshold_grid:
            current_tur = np.sum(unk_scores_arr < th) / total_unknown
            current_tkr = np.sum((val_scores_arr >= th) & val_correct_mask) / total_known

            diff = np.abs(current_tkr - current_tur)
            if diff < min_diff:
                min_diff = diff
                best_threshold = th

        # 计算考虑黄金阈值拦截下的已知类最终实际准确率
        val_correct_under_thresh = np.sum((val_scores_arr >= best_threshold) & val_correct_mask)
        val_accuracy_under_thresh = val_correct_under_thresh / total_known

        # 利用决断出的最佳黄金阈值，计算总未知类拦截准确率
        total_correct_unknown_rejections = np.sum(unk_scores_arr < best_threshold)
        total_unk_rejection_acc = total_correct_unknown_rejections / total_unknown

        # 计算 Generalized Open-Set AUROC
        y_true_binary = np.concatenate([np.ones_like(val_scores_arr), np.zeros_like(unk_scores_arr)])
        y_true_binary[:len(val_scores_arr)] = np.where(val_correct_mask, 1, 0)
        all_scores_concat = np.concatenate([val_scores_arr, unk_scores_arr])

        try:
            generalized_open_set_auroc = roc_auc_score(y_true_binary, all_scores_concat)
        except ValueError:
            generalized_open_set_auroc = 0.5

        # 统计各个未知类拦截准确率 (U-Acc)
        unknown_info_list = []
        for class_id, sims_list in unknown_class_sims.items():
            sims_tensor = torch.tensor(sims_list)
            correct_preds = (sims_tensor < best_threshold).sum().item()
            acc_u = correct_preds / len(sims_list) if len(sims_list) > 0 else 0.0
            class_name = encoder_unknown.inverse_transform([class_id])[0]
            unknown_info_list.append(f"{class_name}: {acc_u * 100:.2f}%")

        unknown_info = ", ".join(unknown_info_list)
        epoch_time = time.time() - start_time

        # ----------------------------------------------------------------
        # 5. 打印及落盘当前 Epoch 的重验证结果
        # ----------------------------------------------------------------
        print(
            f"[Re-Val] Epoch {epoch} Done, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val Acc(T): {val_accuracy_under_thresh:.4f} | "
            f"Auto-Thresh: {best_threshold:.4f}, Total U-Acc: {total_unk_rejection_acc * 100:.2f}, Gen-AUROC: {generalized_open_set_auroc:.4f} | "
            f"U-Acc(<Thresh): [{unknown_info}], Time: {epoch_time:.2f}s")

        writer_csv.writerow({
            'epoch': epoch, 'val_loss': avg_val_loss, 'val_accuracy': val_accuracy,
            'val_acc_under_thresh': val_accuracy_under_thresh,
            'auto_threshold': best_threshold,
            'total_unk_rejection_acc': total_unk_rejection_acc,
            'open_set_auroc': generalized_open_set_auroc,
            'epoch_time': epoch_time
        })
        csv_file.flush()

    csv_file.close()
    print(f"\n🎉 验证报告全部生成完毕，已成功保存至: {re_val_log_path}")