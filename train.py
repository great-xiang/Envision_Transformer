import random
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
import os
import logging
from torch.optim.lr_scheduler import StepLR
from utils import MyDataset
from track_trans import TrackTrans
device = torch.device("cuda:0")
data_path = 'data_set/OTB100_result'

train_dataset = MyDataset(data_path)
# 创建DataLoader并设置collate_fn参数,禁用元组
dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True, collate_fn=default_collate)

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
track_trans.to(device)

loss_function = torch.nn.MSELoss()
optimizer = optim.Adam(track_trans.parameters(), lr=0.001)
# 创建日志记录器
logging.basicConfig(filename='training.log', level=logging.INFO)
# 模型保存路径
model_path = 'models/'
os.makedirs(model_path, exist_ok=True)
# 添加学习率调度器
scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
# 训练模型
num_epochs = 500
for epoch in range(num_epochs):
    track_trans.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()

    # 返回的pandas_list是[batch,seq_length,197,256]维的tensor数组
    for batch_idx, pandas_list in enumerate(dataloader):
        # 随机生成这个批次数据的长度
        train_dataset.__set__(random.randint(4, 11))

        optimizer.zero_grad()

        # 将pandas_list的最后一个数据作为label
        outputs = track_trans(pandas_list[:, 0:-1]).to('cpu')

        # 取每个batch的最后一个seq的最后一个维度的前四个数值作为label
        loss = loss_function(outputs, pandas_list[:, -1, -1, 0:4])
        loss = torch.mean(loss)  # 计算平均损失
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += pandas_list.__len__()

    epoch_loss = running_loss / len(dataloader)
    # 更新学习率
    scheduler.step()

    # 记录日志
    end_time = time.time()
    epoch_time = end_time - start_time
    logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Time: {epoch_time:.2f}s")
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Time: {epoch_time:.2f}s")
    # 每隔一定周期保存模型
    if (epoch + 1) % 50 == 0:
        model_file = os.path.join(model_path, f'model_epoch_{epoch + 1}.pt')
        torch.save(track_trans.state_dict(), model_file)

