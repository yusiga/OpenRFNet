import torch
import random
import numpy as np
import configparser
from TimesFM_lora import TimesFMForClassification
config = configparser.ConfigParser()
config.read('config.ini')

seed = 42


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def reset_trainable_heads(model):

    def _reset(module):
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()

    model.feature_extractor.apply(_reset)
    model.fc_head.reset_parameters()



def compare_state_dict(dict1, dict2, name):

    print("\n======================")
    print(name)
    print("======================")

    total_diff = 0

    for k in dict1.keys():

        if k not in dict2:
            print("Missing:", k)
            continue

        diff = torch.abs(dict1[k] - dict2[k]).mean().item()

        total_diff += diff

        print(
            f"{k:50s} diff={diff:.10f}"
        )

    print("----------------------")
    print("Average diff:", total_diff / len(dict1))



# =====================================================
# A方案
# 创建模型后的初始化参数
# =====================================================

set_seed(seed)

print("Create model A")

model_A = TimesFMForClassification(
    num_classes=6,
    device="cuda"
)


feature_A = {
    k:v.clone().detach().cpu()
    for k,v in model_A.feature_extractor.state_dict().items()
}

fc_A = {
    k:v.clone().detach().cpu()
    for k,v in model_A.fc_head.state_dict().items()
}



# =====================================================
# B方案
# 创建模型
# load checkpoint
# reset
# =====================================================

set_seed(seed)

print("\nCreate model B")

model_B = TimesFMForClassification(
    num_classes=6,
    device="cuda"
)


# 加载你的stage1模型
stage1_ckpt_path = config.get('general', 'path_to_model')
ckpt = torch.load(stage1_ckpt_path, map_location="cuda")

model_B.load_state_dict(
    ckpt["model_state_dict"]
)


reset_trainable_heads(model_B)



feature_B = {
    k:v.clone().detach().cpu()
    for k,v in model_B.feature_extractor.state_dict().items()
}

fc_B = {
    k:v.clone().detach().cpu()
    for k,v in model_B.fc_head.state_dict().items()
}



# =====================================================
# 比较
# =====================================================

compare_state_dict(
    feature_A,
    feature_B,
    "feature_extractor"
)


compare_state_dict(
    fc_A,
    fc_B,
    "fc_head"
)