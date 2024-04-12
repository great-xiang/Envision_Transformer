import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset

def get_pandas(path):
    pandas_file = []  # pandas文件名列表
    for file in os.listdir(path):
        pandas_file.append(file)
    # 二维列表，第一维存储类别，第二维存储pandas数据
    pandas_list = []

    for index2, name in enumerate(pandas_file):
        file_path = path + '/' + pandas_file[index2]
        pandas_list.append([])
        for img_path in sorted(os.listdir(file_path), key=lambda x: int(x.split('.')[0])):
            pandas_list[index2].append(file_path + '/' + str(img_path))
    return pandas_list

class MyDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path  # 数据集根目录
        self.pandas_list = get_pandas(self.data_path)
        self.seq_length = 4

    def __len__(self):
        # 返回数据集的数据数量
        return len(self.pandas_list)

    def __number__(self, n):
        # 返回第n类物体的数据数量
        return len(self.pandas_list[n])

    def __getitem__(self, index):
        result_list = []
        start_index = random.randint(0, len(self.pandas_list[index]) - self.seq_length)
        for i in range(start_index, start_index + self.seq_length):
            df = pd.read_feather(self.pandas_list[index][i])
            result_list.append(torch.tensor(df.to_numpy()))
        return torch.stack(result_list, dim=0)

    def __set__(self, seq_length):
        self.seq_length = seq_length