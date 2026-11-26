import ast
import os
import glob
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timesfm
from timesfm.torch import util

revin = util.revin


# ========================================================
# 1. 完好保留的残差特征提取网络（无 Dropout）
# ========================================================
class SmoothFeatureExtractor(nn.Module):
    def __init__(self, d_model=1280):
        super().__init__()
        # 1280 -> 512 -> 128 -> 64 (平滑流形压缩，强迫特征向 MAV 极致靠拢)
        self.fc1 = nn.Linear(d_model, 512)
        self.ln1 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 128)
        self.ln2 = nn.LayerNorm(128)
        self.fc_out = nn.Linear(128, 64)

        # 快捷残差分支：保证底座大模型最原始的宏观时序特征直接无损传导到 64 维空间
        self.shortcut = nn.Linear(d_model, 64)

    def forward(self, x):
        res = self.shortcut(x)

        # 主干路径 (完全确定性映射，无随机失真)
        x = F.gelu(self.ln1(self.fc1(x)))
        x = F.gelu(self.ln2(self.fc2(x)))
        x = self.fc_out(x)

        # 残差融合
        return x + res


# ========================================================
# 2. 纯特征探查版（无 LoRA）的 TimesFM 分类模型
# ========================================================
class TimesFMForClassification(nn.Module):
    def __init__(self, num_classes, device="cuda", max_context=16000, pool="mean"):
        super().__init__()
        self.device = torch.device(device)
        self.max_context = max_context
        self.pool = pool
        self.p = 32

        # 降维与时序预处理器
        self.reducer = nn.Sequential(
            nn.AvgPool1d(kernel_size=4, stride=4),
            nn.AdaptiveAvgPool1d(self.max_context),
        )

        # ===== TimesFM Backbone 初始化 =====
        self.backbone = timesfm.TimesFM_2p5_200M_torch()
        self.backbone.load_checkpoint(path="../model/TIMESFM-2.5-200M/model.safetensors")

        self.backbone.compile(
            timesfm.ForecastConfig(
                max_context=max_context,
                max_horizon=1,
                normalize_inputs=True,
                use_continuous_quantile_head=False,
                force_flip_invariance=False,
                infer_is_positive=False,
                fix_quantile_crossing=False,
            )
        )

        self.core: nn.Module = self.backbone.model

        # ===== 彻底全量冻结大模型底座所有参数 =====
        for p in self.core.parameters():
            p.requires_grad = False

        # ===== 1. 重新请回你的残差特征提取层（输出固定的 64 维特征） =====
        d_model = 1280
        self.feature_extractor = SmoothFeatureExtractor(d_model=d_model)

        # ===== 2. 额外外挂的独立线性分类头（接收 64 维，输出类别数） =====
        self.fc_head = nn.Linear(64, num_classes)

        self.to(self.device)

    def encode_torch_new(
            self,
            inputs: torch.Tensor,
            masks: torch.Tensor | None = None,
            pool: str = "mean",
    ) -> torch.Tensor:
        inputs = inputs.to(self.device)
        if masks is None:
            masks = torch.zeros_like(inputs, dtype=torch.bool)
        masks = masks.to(self.device)

        batch_size, context = inputs.shape[0], inputs.shape[1]
        num_input_patches = context // self.p

        patched_inputs = torch.reshape(inputs, (batch_size, -1, self.p))
        patched_masks = torch.reshape(masks, (batch_size, -1, self.p))

        # 滚动计算时序非平稳状态 (REvIN 前置)
        n = torch.zeros(batch_size, device=inputs.device)
        mu = torch.zeros(batch_size, device=inputs.device)
        sigma = torch.zeros(batch_size, device=inputs.device)
        patch_mu, patch_sigma = [], []
        for i in range(num_input_patches):
            (n, mu, sigma), _ = util.update_running_stats(
                n, mu, sigma, patched_inputs[:, i], patched_masks[:, i]
            )
            patch_mu.append(mu)
            patch_sigma.append(sigma)
        context_mu = torch.stack(patch_mu, dim=1)
        context_sigma = torch.stack(patch_sigma, dim=1)

        # REvIN 归一化
        normed_inputs = revin(patched_inputs, context_mu, context_sigma, reverse=False)
        normed_inputs = torch.where(patched_masks, 0.0, normed_inputs)

        # 前向流
        (_, output_embeddings, _, _), _ = self.core(
            normed_inputs,
            patched_masks,
            decode_caches=None
        )

        patch_valid = (~patched_masks[..., -1]).to(output_embeddings.dtype)

        # 特征池化
        if pool == "last":
            idxs = torch.clamp(patch_valid.cumsum(dim=1).argmax(dim=1), min=0)
            gathered = []
            for b in range(batch_size):
                j = int(idxs[b].item())
                if patch_valid[b].sum() == 0:
                    j = 0
                gathered.append(output_embeddings[b, j])
            reps = torch.stack(gathered, dim=0)
        else:
            denom = patch_valid.sum(dim=1, keepdim=True).clamp_min(1.0)
            reps = (output_embeddings * patch_valid.unsqueeze(-1)).sum(dim=1) / denom

        return reps

    def _encode_torch(self, inputs_2d: torch.Tensor, pool="mean") -> torch.Tensor:
        m = torch.zeros_like(inputs_2d, dtype=torch.bool, device=self.device)
        return self.encode_torch_new(inputs_2d, m, pool=pool)

    def _encode_magnitude(self, x_1c_1d: torch.Tensor) -> torch.Tensor:
        seqs = x_1c_1d[:, 0, :]
        return self._encode_torch(seqs, pool=self.pool)

    def forward(self, x: torch.Tensor):
        x = x.to(self.device)
        x = self.reducer(x)
        reps = self._encode_magnitude(x)

        # 1. 经过残差提取器，拿到受残差保护、确定凝聚的 64 维特征
        feat64 = self.feature_extractor(reps)

        # 2. 送入外挂分类头得到决策 logits [Batch_Size, num_classes]
        logits = self.fc_head(feat64)

        return logits