import numpy as np
import torch.nn as nn
from track_trans import TrackTrans
import torch
import pandas as pd

device = torch.device("cuda:0")

# Input dimensions: 256 (features) +参考代码 (timestamp)
# 输入序列长度为257
input_dim = 257
# 序列为3，需要输入3个以前的图像特征图序列，预测下一个图像位置
seq_length = 3
# 输出维度为4
output_dim = 4
deep = 2
num_heads = 1
# 时差，输入序列和预测序列差多少，1意味着预测下一帧物体目标，2意味着预测下两帧物体目标
time_difference = 1

track_trans = TrackTrans(input_dim, seq_length, time_difference, output_dim, deep, num_heads)
# load model weights
model_weight_path = "models/first_model/first_model.pt"
track_trans.load_state_dict(torch.load(model_weight_path, map_location=device))
track_trans.to(device)
track_trans.eval()
# 读取数据
feather_file1 = "test_data/basketball_1.feather"
feather_file2 = "test_data/basketball_1.feather"
feather_file3 = "test_data/basketball_1.feather"
feather_file4 = "test_data/basketball_1.feather"

df1 = pd.read_feather(feather_file1)
tensor1 = torch.tensor(df1.to_numpy())
df2 = pd.read_feather(feather_file2)
tensor2 = torch.tensor(df2.to_numpy())
df3 = pd.read_feather(feather_file3)
tensor3 = torch.tensor(df3.to_numpy())
df4 = pd.read_feather(feather_file4)
tensor4 = torch.tensor(df4.to_numpy())
input_data = [[tensor1, tensor2, tensor3]]

with torch.no_grad():
    output = track_trans(input_data)
    print(output)
    print(tensor4[-1, 0:4])
