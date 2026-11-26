# -*- coding: utf-8 -*-
# THOP 统计模型复杂度：参数量 / 可训练参数 / MACs / FLOPs(=2*MACs)
# 按你给的调用方式：StudentResNet1D -> load_state_dict -> eval
#
# 依赖：
#   pip install thop
#
# 使用：
#   1) 修改下面的 num_classes、ckpt_path、dummy_input_shape
#   2) 直接运行：python thop_profile_student_resnet1d.py

import torch
from thop import profile
import ast
# ====== 你自己的模型 ======
from ResNet_model import ResNetSTFT  # 按你的工程实际路径改\
import configparser

config = configparser.ConfigParser()
config.read('config.ini')


def count_params(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ====== 你需要改的三项 ======
    num_classes = 8
    ckpt_path = config.get('general', 'path_to_model_acc')  # 对应 config.get('general','path_to_model_acc')
    dummy_input_shape = (1, 1, 448, 448)  # (B, C, L) 按你的真实输入改：通道数/长度
    # ==========================
    input_shape = config.get('general', 'input_shape')
    input_shape = ast.literal_eval(input_shape)
    # 1) build + load + eval（按你给的方式）
    model = ResNetSTFT(input_shape, num_classes).to(device)
    model_weights = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(model_weights, strict=True)
    model.eval()

    # 2) 参数量
    total_params, trainable_params, frozen_params = count_params(model)

    # 3) THOP 统计 MACs / FLOPs
    dummy = torch.randn(*dummy_input_shape, device=device)
    macs, _ = profile(model, inputs=(dummy,), verbose=False)
    flops = 2 * macs  # 论文里常用：FLOPs = 2 * MACs

    # 4) 输出（“能找到啥写啥” -> 全给你）
    print("========== Model Complexity (THOP) ==========")
    print(f"Device              : {device}")
    print(f"Model               : {model.__class__.__name__}")
    print(f"Num classes         : {num_classes}")
    print(f"Checkpoint          : {ckpt_path}")
    print(f"Dummy input shape   : {dummy_input_shape} (B,C,L)")
    print("--------------------------------------------")
    print(f"Total Params        : {total_params}  ({total_params / 1e6:.4f} M)")
    print(f"Trainable Params    : {trainable_params}  ({trainable_params / 1e6:.4f} M)")
    print(f"Frozen Params       : {frozen_params}  ({frozen_params / 1e6:.4f} M)")
    print("--------------------------------------------")
    print(f"MACs                : {macs:.4f}  ({macs / 1e9:.4f} G MACs)")
    print(f"FLOPs (2*MACs)      : {flops:.4f}  ({flops / 1e9:.4f} GFLOPs)")
    print("Note                : FLOPs are reported as 2×MACs (common in papers).")
    print("============================================")


if __name__ == "__main__":
    main()
