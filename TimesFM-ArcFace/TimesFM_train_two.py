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
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
import random
import numpy as np
import configparser
from My_TimesFM_lora import TimesFMForClassification
from pytorch_metric_learning import losses
import torch.nn.functional as F

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

# GPU configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

path_to_files = config.get('general', 'path_to_files')
encoder = joblib.load(config.get('general', 'path_to_encoder'))
encoder_unknown = joblib.load(config.get('general', 'path_to_encoder_open'))
label_allow_list = encoder.classes_
open_list = encoder_unknown.classes_
batch_size = config.getint('general', 'batch_size_for_train')
num_classes = config.getint('general', 'num_classes')
input_shape = ast.literal_eval(config.get('general', 'input_shape'))

max_epochs = config.getint('general', 'max_epochs')
max_lr = config.getfloat('general', 'max_lr')

accumulation_steps = 1

# ========= Stage2: 创建模型并加载第一阶段权重 =========
model = TimesFMForClassification(num_classes, device=device).to(device)

stage1_ckpt_path = config.get('general', 'path_to_model')
ckpt = torch.load(stage1_ckpt_path, map_location=device)
state_dict = ckpt["model_state_dict"]

model_dict = model.state_dict()
filtered_dict = {}
for k, v in state_dict.items():
    if "feature_extractor" in k:
        continue
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

for name, p in model.feature_extractor.named_parameters():
    p.requires_grad = True

# ========= DataLoader =========
def set_dataloader():
    train_file_paths, train_labels = [], []
    val_file_paths, val_labels = [], []
    unknown_file_paths, unknown_labels = [], []

    for parent_path in glob.glob(path_to_files):
        label = os.path.basename(parent_path)

        if label not in label_allow_list:
            unknown_parent_path = os.path.join(parent_path, 'val/*')
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
            val_parent_path = os.path.join(parent_path, 'val/*')
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

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    unknown_loader = DataLoader(unknown_set, batch_size=batch_size)
    return train_loader, val_loader, unknown_loader


# ========= 训练主循环 =========
if __name__ == '__main__':

    train_loader, val_loader, unknown_loader = set_dataloader()

    criterion = losses.ArcFaceLoss(
        num_classes=num_classes,
        embedding_size=64,
        margin=28.6,
        scale=16
    ).to(device)

    optimizer = torch.optim.Adam([
        {'params': filter(lambda p: p.requires_grad, model.parameters())},
        {'params': criterion.parameters()}
    ], lr=max_lr)

    print("可训练参数列表：")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  - {name}")

    for name, param in criterion.named_parameters():
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

    import torch.nn.functional as F


    def entropy_maximization_loss(logits):
        log_probs = F.log_softmax(logits, dim=1)
        bs, num_classes = logits.shape
        uniform_target = torch.full((bs, num_classes), 1.0 / num_classes).to(logits.device)
        loss_u = F.kl_div(log_probs, uniform_target, reduction='batchmean')
        return loss_u

    # ========= Log 配置 (新增 train_unknown_loss 字段) =========
    csv_file = open(config.get('general', 'path_to_log'), 'w', newline='')
    fieldnames = ['epoch', 'train_loss', 'train_unknown_loss', 'train_accuracy', 'val_loss', 'val_accuracy', 'epoch_time', 'lr']
    writer_csv = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer_csv.writeheader()

    print("====== Stage2 Train Begin ======")

    for epoch in range(max_epochs):
        start_time = time.time()
        model.train()

        # 初始化统计变量
        train_loss_total, train_u_loss_total, train_correct, train_total,gen_total = 0, 0, 0, 0,0
        optimizer.zero_grad()

        for i, (data, targets) in enumerate(train_loader):
            inputs = data.float().to(device)
            targets = targets.to(device)
            lr = optimizer.param_groups[0]['lr']

            # with torch.no_grad():
            x_reduced = model.reducer(inputs)
            reps_clean = model._encode_magnitude(x_reduced)

            perm = torch.randperm(targets.size(0))
            diff_mask = (targets != targets[perm])
            if diff_mask.sum() == 0:
                feat_clean = model.feature_extractor(reps_clean)
                logits_clean = criterion.get_logits(feat_clean)
                loss = criterion(feat_clean, targets)
                loss_u = torch.tensor(0.0).to(device)
            else:
                reps_a = reps_clean[diff_mask]
                reps_b = reps_clean[perm][diff_mask]
                alpha = 2.0
                lam = np.random.beta(alpha, alpha)
                lam = max(0.3, min(0.7, lam))

                reps_synthetic = lam * reps_a + (1 - lam) * reps_b

                feat_clean = model.feature_extractor(reps_clean)
                feat_synthetic = model.feature_extractor(reps_synthetic)

                logits_clean = criterion.get_logits(feat_clean)
                loss_arc = criterion(feat_clean, targets)

                logits_synthetic = criterion.get_logits(feat_synthetic)
                loss_u = entropy_maximization_loss(logits_synthetic)

                # 总损失
                loss = loss_arc + 0.5 * loss_u

            loss_scaled = loss / accumulation_steps
            loss_scaled.backward()

            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
                optimizer.step()
                onecycle_scheduler.step()
                optimizer.zero_grad()

            # 统计累加
            train_loss_total += loss.item() * targets.size(0)
            train_u_loss_total += loss_u.item() * diff_mask.sum()
            _, predicted = torch.max(logits_clean.data, 1)
            train_correct += (predicted == targets).sum().item()
            train_total += targets.size(0)
            gen_total += diff_mask.sum()
        # 计算 Epoch 平均指标
        avg_train_loss = train_loss_total / train_total
        avg_train_u_loss = train_u_loss_total / (gen_total + 1e-9)
        train_accuracy = train_correct / train_total

        # ========= Val =========
        model.eval()
        val_loss_total, val_correct, val_total = 0, 0, 0
        all_embeddings, all_targets = [], []
        feature_sums = torch.zeros(num_classes, 64).to(device)
        class_counts = torch.zeros(num_classes, 1).to(device)

        with torch.no_grad():
            for data, targets in val_loader:
                inputs = data.float().to(device)
                targets = targets.to(device)
                feat = model(inputs)

                v_loss = criterion(feat, targets)
                v_logits = criterion.get_logits(feat)
                val_loss_total += v_loss.item() * targets.size(0)
                _, predicted = torch.max(v_logits.data, 1)
                val_correct += (predicted == targets).sum().item()
                val_total += targets.size(0)

                norm_feats = F.normalize(feat, p=2, dim=1)
                all_embeddings.append(norm_feats)
                all_targets.append(targets)

                ones = torch.ones(targets.size(0), 1).to(device)
                class_counts.scatter_add_(0, targets.view(-1, 1), ones)
                target_expanded = targets.view(-1, 1).expand(-1, norm_feats.size(1))
                feature_sums.scatter_add_(0, target_expanded, norm_feats)

        # 计算阈值
        MAV = feature_sums / (class_counts + 1e-10)
        norm_mav = F.normalize(MAV, p=2, dim=1)
        all_embeddings_cat = torch.cat(all_embeddings)
        sim_matrix = torch.mm(all_embeddings_cat, norm_mav.t())
        max_sims, _ = torch.max(sim_matrix, dim=1)
        threshold = torch.quantile(max_sims, 0.05).item()

        # Unknown test
        unknown_class_sims = defaultdict(list)
        with torch.no_grad():
            for data, targets in unknown_loader:
                inputs = data.float().to(device)
                u_feat = model(inputs)
                norm_u = F.normalize(u_feat, p=2, dim=1)
                cos_sim_u = torch.mm(norm_u, norm_mav.t())
                max_u, _ = torch.max(cos_sim_u, dim=1)
                for k in range(targets.size(0)):
                    unknown_class_sims[targets[k].item()].append(max_u[k].item())

        unknown_info = ", ".join([
            f"{encoder_unknown.inverse_transform([cid])[0]}: {torch.quantile(torch.tensor(sims), 0.95).item():.4f}"
            for cid, sims in unknown_class_sims.items()
        ])

        avg_val_loss = val_loss_total / val_total
        val_accuracy = val_correct / val_total
        epoch_time = time.time() - start_time

        # 打印日志 (含 U-Loss)
        print(f"[Stage2] Epoch {epoch}, Loss: {avg_train_loss:.4f}, U-Loss: {avg_train_u_loss:.4f}, "
              f"Acc: {train_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}, "
              f"K-Thresh: {threshold:.4f}, U-Thresh(95%): [{unknown_info}], Time: {epoch_time:.2f}s, lr: {lr:.6f}")

        # 写入 CSV
        writer_csv.writerow({
            'epoch': epoch, 'train_loss': avg_train_loss, 'train_unknown_loss': avg_train_u_loss,
            'train_accuracy': train_accuracy, 'val_loss': avg_val_loss, 'val_accuracy': val_accuracy,
            'epoch_time': epoch_time, 'lr': lr
        })
        csv_file.flush()

        save_dir = config.get('general', 'path_to_model_folder_stage2')
        os.makedirs(save_dir, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'criterion_state_dict': criterion.state_dict(),
        }, os.path.join(save_dir, f"stage2_epoch_{epoch}.pth"))

    csv_file.close()