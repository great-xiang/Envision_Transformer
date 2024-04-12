import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda:0")


class TrackTrans(nn.Module):
    def __init__(self, input_dim=257, seq_length=3, time_difference=1, output_dim=4, deep=1, num_heads=1):
        super(TrackTrans, self).__init__()
        self.seq_length = seq_length
        self.time_difference = time_difference
        self.pos_embedding = nn.Parameter(torch.randn(196, 256))
        self.time_embedding = nn.Parameter(torch.randn(seq_length, 197, 1))
        self.pos_encoding = PositionalEncoding()
        self.time_encoding = TimeEncoding(self.seq_length, self.time_difference)
        self.attentions = nn.ModuleList([
            nn.MultiheadAttention(input_dim, num_heads) for _ in range(deep)
        ])
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.linear1 = nn.Linear(input_dim, 2048)
        self.linear2 = nn.Linear(2048, input_dim)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, input_data):
        # 添加位置编码信息,batch_size*seq_length*197*256
        img_feature_pos = self.pos_encoding(input_data)
        # 添加时间信息，batch_size*seq_length*197*256->batch_size*seq_length*197*257
        img_feature_pos_time = self.time_encoding(img_feature_pos)
        # 将4维tensor展平为3维tensor,[B,S,197,257]->[B,S*197,257]
        img_feature_pos_time = img_feature_pos_time.view(len(input_data), self.seq_length * 197, 257)
        img_feature_pos_time = img_feature_pos_time.to(device)
        # 循环遍历多个自注意力层
        for attention_layer in self.attentions:
            # self.attention用来计算自注意力,接受三个输入参数：query、key 和 value 通常是相同的，使用相同的张量 img_feature_pos_time 作为所有三个参数。
            x, _ = attention_layer(img_feature_pos_time, img_feature_pos_time, img_feature_pos_time)
            # 残差连接
            img_feature_pos_time = x + img_feature_pos_time
            # 层归一化
            img_feature_pos_time = self.norm1(img_feature_pos_time)
            # 两次线性变换并激活
            x = F.relu(self.linear1(img_feature_pos_time))
            x = self.linear2(x)
            # 再次残差连接并归一化
            img_feature_pos_time = x + img_feature_pos_time
            img_feature_pos_time = self.norm2(img_feature_pos_time)
        # 获取最终的token，此时img_feature_pos_time的形状为(S*197, 257)
        final_token = img_feature_pos_time[:, -1]
        outputs = self.linear(final_token)
        return outputs

class PositionalEncoding(nn.Module):
    def __init__(self):
        super(PositionalEncoding, self).__init__()
        # 创建一个形状为(len, d_model)的全零张量，用于存储位置编码
        self.pos_embedding = torch.zeros(196, 256)
        position = torch.arange(0, self.pos_embedding.size(0)).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.pos_embedding.size(1), 2) * -(math.log(10000.0) / self.pos_embedding.size(1)))
        self.pos_embedding[:, 0::2] = torch.sin(position * div_term)
        self.pos_embedding[:, 1::2] = torch.cos(position * div_term)

    def forward(self, input_data):
        # x.size(0)表示输入序列的长度。self.pe[:x.size(0), :]：表示将位置编码张量self.pe按照输入序列的长度进行截取，取前x.size(0)个位置，保留所有维度的内容
        img_feature_pos = []
        for index, batch_data in enumerate(input_data):
            img_feature_pos.append([])
            for img_feature in batch_data:
                x = img_feature[0:196, :] + self.pos_embedding[0:196, :]
                x = torch.cat((x, img_feature[196].unsqueeze(0)), dim=0)
                img_feature_pos[index].append(x)
        return img_feature_pos


class TimeEncoding(nn.Module):
    def __init__(self, seq_length, time_difference):
        super(TimeEncoding, self).__init__()
        self.seq_length = seq_length
        # 创建一个形状为(len, d_model)的全零张量，用于存储位置编码
        self.time_embedding = torch.zeros(seq_length, 197, 1)
        # 等差数列，获取时间戳
        time_sequence = np.linspace(0, 1, seq_length + time_difference)
        for i in range(seq_length):
            self.time_embedding[i, :, :] = time_sequence[i]

    def forward(self, img_feature_pos):
        img_feature_pos_time = []
        for index, batch_data in enumerate(img_feature_pos):
            img_feature_pos_time.append([])
            # 3*197*256->3*197*257
            for i in range(self.seq_length):
                img_feature_pos_time[index].append(torch.cat((batch_data[i], self.time_embedding[i]), dim=1))
            # 转换为tensor
            img_feature_pos_time[index] = torch.stack(img_feature_pos_time[index])
        img_feature_pos_time = torch.stack(img_feature_pos_time)
        return img_feature_pos_time
