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
input_shape = ast.literal_eval(config.get('general', 'input_shape'))  # e.g. [2, 40000]

max_epochs = config.getint('general', 'max_epochs')
max_lr = config.getfloat('general', 'max_lr')

model = TimesFMForClassification(num_classes, device=device).to(device)
accumulation_steps = 1


# ========= DataLoader =========
def set_dataloader():
    train_file_paths, train_labels = [], []
    val_file_paths, val_labels = [], []
    unknown_file_paths, unknown_labels = [], []
    for parent_path in glob.glob(path_to_files):
        label = os.path.basename(parent_path)
        if label not in label_allow_list:
            # ===== unknown =====
            unknown_parent_path = os.path.join(parent_path, 'val/*')
            for files_path in glob.glob(unknown_parent_path):
                h5_file_path = os.path.join(files_path, '*.h5')
                for id, file_path in enumerate(glob.glob(h5_file_path)):
                    unknown_file_paths.append(file_path)
                    unknown_labels.append(label)
        else:
            train_parent_path = os.path.join(parent_path, 'train/*')
            for files_path in glob.glob(train_parent_path):
                h5_file_path = os.path.join(files_path, '*.h5')
                for id, file_path in enumerate(glob.glob(h5_file_path)):
                    train_file_paths.append(file_path)
                    train_labels.append(label)

            # ===== val =====
            val_parent_path = os.path.join(parent_path, 'val/*')
            for files_path in glob.glob(val_parent_path):
                h5_file_path = os.path.join(files_path, '*.h5')
                for id, file_path in enumerate(glob.glob(h5_file_path)):
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
    print('train batch nums:', len(train_loader))
    print('val batch nums:', len(val_loader))

    criterion = losses.ArcFaceLoss(
        num_classes=num_classes,
        embedding_size=64,
        margin=28.6,  # 角度间隔
        scale=16  # 缩放因子
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
    optimizer.zero_grad()
    # 日志
    csv_file = open(config.get('general', 'path_to_log'), 'w', newline='')
    fieldnames = ['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy', 'epoch_time', 'lr']
    writer_csv = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer_csv.writeheader()

    # 新增：两个 best 指标
    best_val_loss = float('inf')
    best_val_acc = -1.0

    print(f"Train begin {accumulation_steps * batch_size}")

    for epoch in range(max_epochs):
        start_time = time.time()
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        lr = optimizer.param_groups[0]['lr']
        # ---- Train ----
        for i, (data, targets) in enumerate(train_loader):
            inputs = data.float().to(device)
            targets = targets.to(device)
            embeddings = model(inputs)
            logits = criterion.get_logits(embeddings)
            loss = criterion(embeddings, targets)
            loss_scaled = loss / accumulation_steps
            loss_scaled.backward()

            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
                optimizer.step()
                onecycle_scheduler.step()
                optimizer.zero_grad()

            train_loss += loss.item() * targets.size(0)
            _, predicted = torch.max(logits.data, 1)
            train_correct += (predicted == targets).sum().item()
            train_total += targets.size(0)

        train_accuracy = train_correct / train_total
        train_loss = train_loss / train_total

        # ---- Val ----
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        all_embeddings = []
        all_targets = []
        feature_sums = torch.zeros(num_classes, 64).to(device)
        class_counts = torch.zeros(num_classes, 1).to(device)

        with torch.no_grad():
            for data, targets in val_loader:
                inputs = data.float().to(device)
                targets = targets.to(device)

                # 1. 提取特征
                embeddings = model(inputs)
                loss = criterion(embeddings, targets)
                logits = criterion.get_logits(embeddings)
                val_loss += loss.item() * targets.size(0)
                _, predicted = torch.max(logits.data, 1)
                val_correct += (predicted == targets).sum().item()
                val_total += targets.size(0)

                norm_feats = F.normalize(embeddings, p=2, dim=1)

                # 2. 存起来供后面算分布（避免二次加载数据）
                all_embeddings.append(norm_feats)
                all_targets.append(targets)

                # 3. 累加计算类中心 (MAV)
                ones = torch.ones(targets.size(0), 1).to(device)
                class_counts.scatter_add_(0, targets.view(-1, 1), ones)
                target_expanded = targets.view(-1, 1).expand(-1, norm_feats.size(1))
                feature_sums.scatter_add_(0, target_expanded, norm_feats)

        # --- 循环结束，开始矩阵化计算 ---

        # 4. 计算并标准化 MAV
        MAV = feature_sums / (class_counts + 1e-10)
        norm_mav = F.normalize(MAV, p=2, dim=1)  # [num_classes, 64]

        # 5. 拼接所有验证集特征并划定已知类 5% 分位数阈值
        all_embeddings_cat = torch.cat(all_embeddings)  # [N, 64]
        sim_matrix = torch.mm(all_embeddings_cat, norm_mav.t())
        max_sims, _ = torch.max(sim_matrix, dim=1)
        threshold = torch.quantile(max_sims, 0.05).item()

        end_time = time.time()
        epoch_time = end_time - start_time

        val_accuracy = val_correct / val_total
        val_loss = val_loss / val_total

        # ---- Unknown 测试核心修改：计算未知类拦截准确率 (U-Acc) ----
        unknown_class_sims = defaultdict(list)
        with torch.no_grad():
            for data, targets in unknown_loader:
                inputs = data.float().to(device)

                embeddings = model(inputs)
                norm_embeddings = F.normalize(embeddings, p=2, dim=1)

                # 计算至已知类中心的余弦相似度
                cos_sim_matrix = torch.mm(norm_embeddings, norm_mav.t())
                max_sims, _ = torch.max(cos_sim_matrix, dim=1)

                max_sims_cpu = max_sims.detach().cpu()
                targets_cpu = targets.detach().cpu()

                for k in range(len(targets_cpu)):
                    class_id = targets_cpu[k].item()
                    unknown_class_sims[class_id].append(max_sims_cpu[k].item())

        # 将原先计算 95% 分位数改为计算拦截率
        unknown_info_list = []
        for class_id, sims_list in unknown_class_sims.items():
            sims_tensor = torch.tensor(sims_list)
            # 核心逻辑：余弦相似度小于已知类阈值，则视为成功判定为未知类(拦截成功)
            correct_preds = (sims_tensor < threshold).sum().item()
            acc_u = correct_preds / len(sims_list) if len(sims_list) > 0 else 0.0
            class_name = encoder_unknown.inverse_transform([class_id])[0]
            unknown_info_list.append(f"{class_name}: {acc_u * 100:.2f}%")

        unknown_info = ", ".join(unknown_info_list)

        # 打印包含全新拦截准确率的日志 (U-Thresh 更名为 U-Acc)
        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, "
              f"K-Thresh(5%): {threshold:.4f}, "
              f"U-Acc(<Thresh): [{unknown_info}], "
              f"Time: {epoch_time:.2f}s, lr: {lr:.6f}")

        writer_csv.writerow({
            'epoch': epoch, 'train_loss': train_loss, 'train_accuracy': train_accuracy,
            'val_loss': val_loss, 'val_accuracy': val_accuracy,
            'epoch_time': epoch_time, 'lr': lr
        })
        csv_file.flush()

        # 模型存储逻辑
        save_dir = config.get('general', 'path_to_model_folder')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_filename = f"model_epoch_{epoch}.pth"
        save_path = os.path.join(save_dir, save_filename)

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'criterion_state_dict': criterion.state_dict(),
        }
        torch.save(checkpoint, save_path)

    csv_file.close()