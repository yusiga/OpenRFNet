import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timesfm
from pytorch_metric_learning import losses


class TimesFMForClassification(nn.Module):
    def __init__(
            self,
            num_classes,
            device="cuda",
            max_context=16000,
            pool="mean",
            dropout=0.3
    ):
        super().__init__()
        self.device = torch.device(device)
        self.max_context = max_context
        self.pool = pool

        self.reducer = nn.Sequential(
            nn.AvgPool1d(kernel_size=4, stride=4),
            nn.AdaptiveAvgPool1d(self.max_context),
        )

        # ===== TimesFM backbone（全冻结，不参与训练）=====
        self.backbone = timesfm.TimesFM_2p5_200M_torch()
        self.backbone.load_checkpoint(
            path="../model/TIMESFM-2.5-200M/model.safetensors"
        )
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

        # ===== 分类头（仅这部分参与训练）=====
        d_model = 1280  # TimesFM 默认输出维度
        self.feature_extractor = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
        )
        self.to(self.device)

    # ======================================================
    # 使用官方 backbone.encode 接口，支持长序列切片
    # ======================================================
    @torch.no_grad()
    def _encode_magnitude(self, x_1c_1d: torch.Tensor):
        """
        x_1c_1d: 形状 [B, 1, L]
        返回: torch Tensor [B, d_model]
        """
        B, C, L = x_1c_1d.shape
        assert C == 1, "当前实现只支持单通道输入"

        seqs = x_1c_1d[:, 0, :]  # [B, L]

        # 情况 1：长度不超过 max_context
        if L <= self.max_context:
            seqs_np = seqs.detach().cpu().numpy()
            reps_np = self.backbone.encode(seqs_np, pool=self.pool)  # [B, d_model]
            return torch.from_numpy(reps_np).to(self.device).float()

        # 情况 2：L > max_context，按 max_context 切分
        chunk_reps = []
        for start in range(0, L, self.max_context):
            end = min(start + self.max_context, L)
            chunk_np = seqs[:, start:end].detach().cpu().numpy()

            # 对每一段进行编码
            rep_chunk_np = self.backbone.encode(chunk_np, pool=self.pool)
            chunk_reps.append(torch.from_numpy(rep_chunk_np))

        # 段间聚合：取平均
        reps = torch.stack(chunk_reps, dim=0).mean(dim=0)  # [B, d_model]
        return reps.to(self.device).float()

    def forward(self, x: torch.Tensor, label: torch.Tensor = None):
        x = self.reducer(x)
        reps = self._encode_magnitude(x)  # [B, 1280]
        feat = self.feature_extractor(reps)  # [B, 64]
        return feat