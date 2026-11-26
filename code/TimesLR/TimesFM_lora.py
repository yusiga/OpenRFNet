import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timesfm
from timesfm.torch import util
revin = util.revin

# =========================
# LoRA Linear
# =========================
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 1.0
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # freeze base
        for p in self.base.parameters():
            p.requires_grad = False

        in_features = base.in_features
        out_features = base.out_features
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        y = self.base(x)
        if self.r > 0:
            delta = (self.dropout(x) @ self.lora_A.t()) @ self.lora_B.t()
            y = y + delta * self.scaling
        return y


def get_parent_module(root: nn.Module, full_name: str):
    parts = full_name.split(".")
    parent = root
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    return parent, parts[-1]


def inject_lora_by_names(root: nn.Module, target_full_names, r=8, alpha=16, dropout=0):
    for full_name in target_full_names:
        parent, child = get_parent_module(root, full_name)
        layer = getattr(parent, child)
        if not isinstance(layer, nn.Linear):
            raise TypeError(f"{full_name} is {type(layer)} not nn.Linear")
        setattr(parent, child, LoRALinear(layer, r=r, alpha=alpha, dropout=dropout))


def build_last_k_layers_qv(total_layers: int, k: int = 20):
    assert total_layers > 0
    k = min(k, total_layers)
    start = total_layers - k
    targets = []
    for i in range(start, total_layers):
        targets.append(f"stacked_xf.{i}.attn.query")
        targets.append(f"stacked_xf.{i}.attn.value")
    return targets


class TimesFMForClassification(nn.Module):
    def __init__(
            self,
            num_classes,
            device="cuda",
            max_context=16000,
            pool="mean",
            dropout=0,
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0,
            lora_last_k=20,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.max_context = max_context
        self.pool = pool
        self.p = 32
        self.reducer = nn.Sequential(
            nn.AvgPool1d(kernel_size=4, stride=4),
            nn.AdaptiveAvgPool1d(self.max_context),
        )

        # ===== TimesFM backbone =====
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
        for p in self.core.parameters():
            p.requires_grad = False

        total_layers = len(self.core.stacked_xf)
        lora_targets = build_last_k_layers_qv(total_layers=total_layers, k=lora_last_k)

        inject_lora_by_names(self.core, lora_targets, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)

        for n, p in self.core.named_parameters():
            if ("lora_A" in n) or ("lora_B" in n):
                p.requires_grad = True

        # ===== 特征提取层 =====
        d_model = 1280
        self.feature_extractor = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            #nn.LayerNorm(64),
        )
        self.fc_head = nn.Linear(64, num_classes)
        self.to(self.device)

    def encode_torch_new(
            self,
            inputs: torch.Tensor,  # [B, T]
            masks: torch.Tensor | None = None,  # [B, T]  bool, True=masked(无效/补零)
            pool: str = "mean",  # "mean" | "last"
    ) -> torch.Tensor:
        """
        提取 TimesFM 最后一层 Transformer 的 patch 级 hidden states，并做池化得到 [B, md] 表征。
        - inputs/masks 会按 patch(长度 self.p) 重排，和 decode() 完全一致的预处理/归一化；
        - 与 decode() 一样执行 REvIN 归一化；
        - 返回 shape: [B, md]，可直接接分类头。
        """
        inputs = inputs.to(self.device)
        if masks is None:
            masks = torch.zeros_like(inputs, dtype=torch.bool)
        masks = masks.to(self.device)

        batch_size, context = inputs.shape[0], inputs.shape[1]
        num_input_patches = context // self.p

        # 1) 补齐到 patch 的整数倍（与 decode 保持一致：前补零，并将 mask=1）
        patched_inputs = torch.reshape(inputs, (batch_size, -1, self.p))
        patched_masks = torch.reshape(masks, (batch_size, -1, self.p))

        # 2) 运行统计（与 decode 完全一致）
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

        # 3) REvIN 归一化 + mask 置零
        normed_inputs = revin(patched_inputs, context_mu, context_sigma, reverse=False)
        normed_inputs = torch.where(patched_masks, 0.0, normed_inputs)

        # 4) 只跑一次前向（不自回归），拿到最后一层 hidden states
        #    forward() 返回: (input_emb, output_emb, point, quant_spread)
        (_, output_embeddings, _, _), _ = self.core(
            normed_inputs,
            patched_masks,
            decode_caches=None
        )
        # output_embeddings: [B, num_patches, md]
        # 注意：Transformer 里用的是每个 patch 的最后一个 mask 位作为该 patch 的有效标记
        patch_valid = (~patched_masks[..., -1]).to(output_embeddings.dtype)  # [B, num_patches]

        # 5) 池化到 [B, md]
        if pool == "last":
            # 取最后一个有效 patch 的表征
            # (计算每个样本的最后一个有效下标)
            idxs = torch.clamp(patch_valid.cumsum(dim=1).argmax(dim=1), min=0)
            gathered = []
            for b in range(batch_size):
                j = int(idxs[b].item())
                # 如果全 0（极端情况），退化为第 0 个
                if patch_valid[b].sum() == 0:
                    j = 0
                gathered.append(output_embeddings[b, j])
            reps = torch.stack(gathered, dim=0)  # [B, md]
        else:
            # mean 池化（按有效 patch 加权平均）
            denom = patch_valid.sum(dim=1, keepdim=True).clamp_min(1.0)  # 防 0
            reps = (output_embeddings * patch_valid.unsqueeze(-1)).sum(dim=1) / denom  # [B, md]

        # 6) 返还特征（已是归一化后的 hidden states）
        return reps

    def _encode_torch(self, inputs_2d: torch.Tensor, pool="mean") -> torch.Tensor:
        m = torch.zeros_like(inputs_2d, dtype=torch.bool, device=self.core.device)
        reps = self.encode_torch_new(inputs_2d, m, pool=pool)
        return reps

    def _encode_magnitude(self, x_1c_1d: torch.Tensor) -> torch.Tensor:
        seqs = x_1c_1d[:, 0, :]
        return self._encode_torch(seqs, pool=self.pool)

    def forward(self, x: torch.Tensor, label: torch.Tensor = None):
        x = x.to(self.device)
        x = self.reducer(x)
        reps = self._encode_magnitude(x)
        feat = self.feature_extractor(reps)
        logits = self.fc_head(feat)
        return feat,logits