from sklearn.preprocessing import LabelEncoder
import joblib

# 定义需要识别的无人机类别
close_list = ['mavic3_1','mavic3S_1','mavicAir2_1','mini3pro_2','mini4PRO_1', 'mini5PRO_1']
open_list = ['mavic3C_1','mavicAir2s_1']

all_list = close_list + open_list
unknown_list = close_list + ['unknowns']


def fit_and_print(name, encoder_obj, data_list, save_path):
    """
    封装的函数：执行拟合、打印类别映射关系、并保存模型
    """
    print(f"=== {name} (Length: {len(data_list)}) ===")
    encoded_labels = encoder_obj.fit_transform(data_list)

    # 打印每个数字编码对应的原始类别
    print(encoder_obj.classes_)
    # 保存模型
    joblib.dump(encoder_obj, save_path)


# 1. label_encoder_close
encoder = LabelEncoder()
fit_and_print('label_encoder_close', encoder, close_list, '../model/label_encoder_close.joblib')

# 2. label_encoder_open
encoder1 = LabelEncoder()
fit_and_print('label_encoder_open', encoder1, open_list, '../model/label_encoder_open.joblib')

# 3. label_encoder_all
encoder2 = LabelEncoder()
fit_and_print('label_encoder_all', encoder2, all_list, '../model/label_encoder_all.joblib')

# 4. label_encoder_unknown
encoder3 = LabelEncoder()
fit_and_print('label_encoder_unknown', encoder3, unknown_list, '../model/label_encoder_unknown.joblib')

