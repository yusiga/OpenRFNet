import ast
import csv
import os
import time
from collections import defaultdict

import torch
import torch.nn as nn
import glob
import joblib
from Dataset import Dataset
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import OneCycleLR
import random
import numpy as np
import configparser
from TimesFM_lora import TimesFMForClassification
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

# ========= 配置 =========
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

max_epochs = config.getint('general', 'max_epochs')
max_lr = config.getfloat('general', 'max_lr')

accumulation_steps = 1

# ========= Stage2: 创建模型并加载第一阶段权重 =========
model = TimesFMForClassification(num_classes, device=device).to(device)

stage1_ckpt_path = config.get('general', 'path_to_model')
ckpt = torch.load(stage1_ckpt_path, map_location=device)
state_dict = ckpt["model_state_dict"]
print(stage1_ckpt_path)

model_dict = model.state_dict()
filtered_dict = {}
for k, v in state_dict.items():
    if "feature_extractor" in k or "fc_head" in k:
        continue  # 过滤后端层重新初始化
    if k in model_dict and model_dict[k].shape == v.shape:
        filtered_dict[k] = v

model_dict.update(filtered_dict)
model.load_state_dict(model_dict)

# ========= Stage2: 冻结参数 =========
for name, p in model.named_parameters():
    if "lora_A" in name or "lora_B" in name:
        p.requires_grad = False
    elif "core" in name:
        p.requires_grad = False

# 激活后端可训练模块
for name, p in model.feature_extractor.named_parameters():
    p.requires_grad = True
for name, p in model.fc_head.named_parameters():
    p.requires_grad = True


# ========= DataLoader =========
def set_dataloader():
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

    print('train nums:', len(train_labels))
    print('val nums:', len(val_labels))
    print('unknown nums:', len(unknown_labels))

    train_set = Dataset(train_file_paths, train_labels)
    val_set = Dataset(val_file_paths, val_labels)
    unknown_set = Dataset(unknown_file_paths, unknown_labels)

    train_loader = DataLoader(train_set, batch_size=batch_size*2, shuffle=False, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=batch_size * 2, shuffle=False)
    unknown_loader = DataLoader(unknown_set, batch_size=batch_size * 2, shuffle=False)
    return train_loader, val_loader, unknown_loader


def entropy_maximization_loss(logits):
    log_probs = F.log_softmax(logits, dim=1)
    bs, n_cls = logits.shape
    uniform_target = torch.full((bs, n_cls), 1.0 / n_cls).to(logits.device)
    loss_u = F.kl_div(log_probs, uniform_target, reduction='batchmean')
    return loss_u


def cache_backbone_features(data_loader, name, cache_path):
    """Cache frozen-backbone 1280-D representations on CPU."""

    if os.path.exists(cache_path):
        cache = torch.load(cache_path, map_location='cpu')
        cached_reps = cache['reps']
        cached_targets = cache['targets']
        print(f'{name} loaded reps: {tuple(cached_reps.shape)}')
        return TensorDataset(cached_reps, cached_targets)

    cached_reps, cached_targets = [], []
    model.eval()
    with torch.no_grad():
        for data, targets in data_loader:
            inputs = data.float().to(device)
            x_reduced = model.reducer(inputs)
            reps = model._encode_magnitude(x_reduced)
            cached_reps.append(reps.cpu())
            cached_targets.append(targets.cpu())

    cached_reps = torch.cat(cached_reps, dim=0)
    cached_targets = torch.cat(cached_targets, dim=0)
    print(f'{name} cached reps: {tuple(cached_reps.shape)}')

    torch.save({
        'reps': cached_reps,
        'targets': cached_targets
    }, cache_path)

    return TensorDataset(cached_reps, cached_targets)


# ========= 训练主循环 =========
if __name__ == '__main__':

    train_loader, val_loader, unknown_loader = set_dataloader()

    # 根据第一阶段模型文件名区分不同的第一阶段 Epoch
    stage1_model_name = os.path.splitext(os.path.basename(stage1_ckpt_path))[0]
    cache_dir = os.path.join(
        os.path.dirname(stage1_ckpt_path),
        'reps_cache',
        stage1_model_name
    )
    os.makedirs(cache_dir, exist_ok=True)

    # Stage-II freezes the backbone and LoRA, so their 1280-D outputs are fixed.
    # Cache them once and train/evaluate only the two task-specific heads afterwards.
    train_cache = cache_backbone_features(
        train_loader,
        'train',
        os.path.join(cache_dir, 'train_cache.pt')
    )
    val_cache = cache_backbone_features(
        val_loader,
        'val',
        os.path.join(cache_dir, 'val_cache.pt')
    )
    unknown_cache = cache_backbone_features(
        unknown_loader,
        'unknown',
        os.path.join(cache_dir, 'unknown_cache.pt')
    )

    train_loader = DataLoader(
        train_cache, batch_size=batch_size, shuffle=True, drop_last=True
    )
    train_loader_eval = DataLoader(
        train_cache, batch_size=batch_size * 2, shuffle=False, drop_last=False
    )
    val_loader = DataLoader(
        val_cache, batch_size=batch_size * 2, shuffle=False, drop_last=False
    )
    unknown_loader = DataLoader(
        unknown_cache, batch_size=batch_size * 2, shuffle=False, drop_last=False
    )

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=max_lr,
        weight_decay=0
    )

    print("可训练参数列表：")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  - {name}")

    total_update_steps = (len(train_loader) // accumulation_steps) * max_epochs
    onecycle_scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_update_steps,
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=10,
        final_div_factor=10
    )

    # ========= Log 配置 =========
    csv_file = open(config.get('general', 'path_to_log'), 'w', newline='')
    # 💥 增加 total_unknown_accuracy 字段记录总未知类拦截率
    fieldnames = [
        'epoch',
        'train_loss',
        'train_unknown_loss',
        'train_accuracy',
        'val_loss',
        'val_accuracy',
        'val_acc_under_thresh',
        'total_unknown_accuracy',
        'auto_threshold',
        'open_set_auroc',
        'epoch_time',
        'lr'
    ]
    writer_csv = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer_csv.writeheader()

    print("====== Stage2 Train Begin (验证部分切换至 FEAT 空间) ======")

    for epoch in range(max_epochs):
        start_time = time.time()
        model.train()

        train_loss_total, train_u_loss_total, train_correct, train_total, gen_total = 0, 0, 0, 0, 0
        optimizer.zero_grad()

        # ----------------------------------------------------------------
        # 1. 动态边界差值训练 Phase
        # ----------------------------------------------------------------
        for i, (data, targets) in enumerate(train_loader):
            reps_clean = data.float().to(device)
            targets = targets.to(device)
            lr = optimizer.param_groups[0]['lr']

            synthetic_groups = []
            for _ in range(1):
                perm = torch.randperm(targets.size(0), device=device)
                diff_mask = (targets != targets[perm])
                if diff_mask.any():
                    reps_a = reps_clean[diff_mask]
                    reps_b = reps_clean[perm][diff_mask]
                    lam = torch.distributions.Beta(2.0, 2.0).sample(
                        (reps_a.size(0), 1)
                    ).to(device)
                    lam = lam.clamp(0.3, 0.7)
                    # alpha = 2.0
                    # lam = np.random.beta(alpha, alpha)
                    # lam = max(0.3, min(0.7, lam))
                    synthetic_groups.append(lam * reps_a + (1.0 - lam) * reps_b)

            feat_clean = model.feature_extractor(reps_clean)
            logits_clean = model.fc_head(feat_clean)
            loss_ce = criterion(logits_clean, targets)

            generated_count = sum(group.size(0) for group in synthetic_groups)
            if generated_count == 0:
                loss_u = torch.tensor(0.0, device=device)
                loss = loss_ce
            else:
                reps_synthetic = torch.cat(synthetic_groups, dim=0)
                feat_synthetic = model.feature_extractor(reps_synthetic)
                logits_synthetic = model.fc_head(feat_synthetic)
                loss_u = entropy_maximization_loss(logits_synthetic)
                loss = loss_ce + 1 * loss_u

            loss_scaled = loss / accumulation_steps
            loss_scaled.backward()

            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
                optimizer.step()
                onecycle_scheduler.step()
                optimizer.zero_grad()

            train_loss_total += loss.item() * targets.size(0)
            train_u_loss_total += loss_u.item() * generated_count
            _, predicted = torch.max(logits_clean.data, 1)
            train_correct += (predicted == targets).sum().item()
            train_total += targets.size(0)
            gen_total += generated_count

        avg_train_loss = train_loss_total / train_total
        avg_train_u_loss = train_u_loss_total / (gen_total + 1e-9)
        train_accuracy = train_correct / train_total

        # 模型权重固化，切入静态数理评估模式
        model.eval()

        # ----------------------------------------------------------------
        # 2. 独立推理：用训练集在 FEAT 空间（64维）无损重构 MAV
        # ----------------------------------------------------------------
        train_feature_sums = torch.zeros(num_classes, 64).to(device)
        train_class_counts = torch.zeros(num_classes, 1).to(device)

        with torch.no_grad():
            for data, targets in train_loader_eval:
                reps = data.float().to(device)
                targets = targets.to(device)

                feat = model.feature_extractor(reps)

                ones = torch.ones(targets.size(0), 1).to(device)
                train_class_counts.scatter_add_(0, targets.view(-1, 1), ones)
                target_expanded = targets.view(-1, 1).expand(-1, feat.size(1))
                train_feature_sums.scatter_add_(0, target_expanded, feat)

        MAV = train_feature_sums / (train_class_counts + 1e-10)
        norm_mav = F.normalize(MAV, p=2, dim=1, eps=1e-10)

        # ----------------------------------------------------------------
        # 3. 独立推理：Val 验证集多分类性能测试与得分收集 (收拢回 FEAT 维)
        # ----------------------------------------------------------------
        val_loss_total, val_correct, val_total = 0, 0, 0
        all_val_max_scores = []
        all_val_predictions_correct = []

        with torch.no_grad():
            for data, targets in val_loader:
                reps = data.float().to(device)
                targets = targets.to(device)

                feat = model.feature_extractor(reps)
                logits = model.fc_head(feat)
                v_loss = criterion(logits, targets)

                val_loss_total += v_loss.item() * targets.size(0)
                _, predicted = torch.max(logits.data, 1)
                val_correct += (predicted == targets).sum().item()
                val_total += targets.size(0)

                norm_feat = F.normalize(feat, p=2, dim=1, eps=1e-10)
                sim_matrix = torch.mm(norm_feat, norm_mav.t())
                max_sims, max_idxs = torch.max(sim_matrix, dim=1)

                all_val_max_scores.extend(max_sims.cpu().numpy())
                all_val_predictions_correct.extend((max_idxs == targets).cpu().numpy())

        avg_val_loss = val_loss_total / val_total
        val_accuracy = val_correct / val_total

        # ----------------------------------------------------------------
        # 4. 独立推理：Unknown 未知类测试与得分收集 (收拢回 FEAT 维)
        # ----------------------------------------------------------------
        all_unknown_max_scores = []
        unknown_class_sims = defaultdict(list)

        with torch.no_grad():
            for data, targets in unknown_loader:
                reps = data.float().to(device)

                feat = model.feature_extractor(reps)
                norm_feat = F.normalize(feat, p=2, dim=1, eps=1e-10)

                cos_sim_matrix = torch.mm(norm_feat, norm_mav.t())
                max_sims, _ = torch.max(cos_sim_matrix, dim=1)

                max_sims_cpu = max_sims.detach().cpu().numpy()
                targets_cpu = targets.detach().cpu().numpy()

                all_unknown_max_scores.extend(max_sims_cpu)
                for k in range(targets.size(0)):
                    unknown_class_sims[targets_cpu[k].item()].append(max_sims_cpu[k].item())

        # ----------------------------------------------------------------
        # 5. 数理推演：解算全局自适应黄金阈值与学术指标
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
        val_correct_under_thresh = np.sum(
            (val_scores_arr >= best_threshold) & val_correct_mask
        )
        val_accuracy_under_thresh = val_correct_under_thresh / total_known

        # 💥 【核心修改】：利用最佳黄金阈值，解算总未知类拦截准确率 (Total U-Acc)
        total_correct_unknown_rejections = np.sum(
            unk_scores_arr < best_threshold
        )
        total_unknown_accuracy = total_correct_unknown_rejections / total_unknown

        # 计算 Generalized Open-Set AUROC
        y_true_binary = np.concatenate([
            np.ones_like(val_scores_arr),
            np.zeros_like(unk_scores_arr)
        ])
        y_true_binary[:len(val_scores_arr)] = np.where(
            val_correct_mask,
            1,
            0
        )
        all_scores_concat = np.concatenate([
            val_scores_arr,
            unk_scores_arr
        ])

        try:
            generalized_open_set_auroc = roc_auc_score(
                y_true_binary,
                all_scores_concat
            )
        except ValueError:
            generalized_open_set_auroc = 0.5

        # 统计各个未知类拦截准确率 (U-Acc)
        unknown_info_list = []
        for class_id, sims_list in unknown_class_sims.items():
            sims_tensor = torch.tensor(sims_list)
            correct_preds = (sims_tensor < best_threshold).sum().item()
            acc_u = correct_preds / len(sims_list) if len(sims_list) > 0 else 0.0
            class_name = encoder_unknown.inverse_transform([class_id])[0]
            unknown_info_list.append(
                f"{class_name}: {acc_u * 100:.2f}%"
            )

        unknown_info = ", ".join(unknown_info_list)
        epoch_time = time.time() - start_time

        # ----------------------------------------------------------------
        # 6. 日志输出与落盘 (已在 Val Acc(T) 后面紧跟 Total U-Acc 打印)
        # ----------------------------------------------------------------
        print(
            f"[Stage2] Epoch {epoch}, Loss: {avg_train_loss:.4f}, U-Loss: {avg_train_u_loss:.4f}, Acc: {train_accuracy:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val Acc(T): {val_accuracy_under_thresh:.4f}, "
            f"Total U-Acc: {total_unknown_accuracy * 100:.2f}% | "
            f"Auto-Thresh: {best_threshold:.4f}, Gen-AUROC: {generalized_open_set_auroc:.4f} | "
            f"U-Acc(<Thresh): [{unknown_info}], Time: {epoch_time:.2f}s, lr: {lr:.6f}"
        )

        writer_csv.writerow({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'train_unknown_loss': avg_train_u_loss,
            'train_accuracy': train_accuracy,
            'val_loss': avg_val_loss,
            'val_accuracy': val_accuracy,
            'val_acc_under_thresh': val_accuracy_under_thresh,
            'total_unknown_accuracy': total_unknown_accuracy,
            'auto_threshold': best_threshold,
            'open_set_auroc': generalized_open_set_auroc,
            'epoch_time': epoch_time,
            'lr': lr
        })
        csv_file.flush()

        save_dir = config.get('general', 'path_to_model_folder_stage2')
        os.makedirs(save_dir, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
        }, os.path.join(save_dir, f"stage2_epoch_{epoch}.pth"))

    csv_file.close()