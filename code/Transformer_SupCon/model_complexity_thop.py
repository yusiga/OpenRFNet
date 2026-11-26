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
from torch import nn

from Sup_TransResNet_model import TransResNet, Classifier_gan
import configparser

config = configparser.ConfigParser()
config.read('config.ini')


class CascadedSignalModel(nn.Module):
    def __init__(self, sup_model, classifier_model):
        super(CascadedSignalModel, self).__init__()
        # 将外部实例化好、加载完权重的模型传进来
        self.sup_model = sup_model
        self.classifier_model = classifier_model

    def forward(self, x):
        # 1. 第一步：数据送入 Sup_model 提取高维特征
        # 这里的 sup_feat 就是你要传给下一个模型的“输入”
        sup_feat = self.sup_model(x)

        # 2. 第二步：将特征直接送入 Classifier_model 得到分类结果
        # 注意：如果你的 Classifier_model 的 forward 正常只接收一个输入，直接传入即可
        # 如果 Classifier_model 会返回 (logits, reps) 等多个值，按原样接收即可
        out = self.classifier_model(sup_feat)

        return out
def count_params(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ====== 你需要改的三项 ======
    num_classes = 21
    dummy_input_shape = (1, 1, 775, 775)  # (B, C, L) 按你的真实输入改：通道数/长度
    # ==========================
    input_shape = config.get('general', 'input_shape')
    input_shape = ast.literal_eval(input_shape)
    # 1) build + load + eval（按你给的方式）
    Classifier_model = Classifier_gan(num_classes).to(device)
    model_weights = torch.load(config.get('general', 'path_to_ce_gan'),map_location=torch.device('cpu'))
    Classifier_model.load_state_dict(model_weights)
    Classifier_model.eval()

    Sup_model = TransResNet(input_shape).to(device)
    model_weights = torch.load(config.get('general', 'path_to_model'),map_location=torch.device('cpu'))
    Sup_model.load_state_dict(model_weights)
    Sup_model.eval()

    model = CascadedSignalModel(Sup_model, Classifier_model).to(device)

    # ========= 4. 切换为测试状态 =========
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
