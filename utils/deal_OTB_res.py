import os
import torch
from my_resnet import my_resnet
from PIL import Image
from torchvision import transforms
import cv2
import pandas as pd

device = torch.device("cuda:0")
# 加载resnet
model = my_resnet()
model.load_state_dict(torch.load("resNet34-pre.pth"), strict=False)
model.to(device)
model.eval()

# 将图片的较短边缩小至224，再截取224*224的图像
data_transform = transforms.Compose(
    [transforms.Resize(224),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

file_path = 'data_set/OTB100'

# 打开每个文件
for img_file in os.listdir(file_path):
    print(img_file)
    img_path = []
    Label = []
    image_index = 0
    # 保存物体位置编码
    temp = torch.zeros(256)
    # 获取图片路径
    image_folder = file_path + '/' + img_file + "/img/"
    for img in os.listdir(image_folder):
        img_path.append(image_folder + img)
    # 获取图片宽高
    image = cv2.imread(img_path[-1])
    size = image.shape
    w = size[1]
    h = size[0]
    this_img_size = max(w, h)

    label_file = file_path + '/' + img_file + '/groundtruth_rect.txt'
    for label in open(label_file):
        s = label.split(',')
        # 删除回车号
        s[3] = s[3][0:-2]
        # 变成数字，s分别为X11,Y11,X22,Y22
        s = list(map(int, s))
        # 将坐标变换为相对位置
        s[0] = (s[0] - (w / 2)) / w
        s[1] = (s[1] - (h / 2)) / h
        s[2] = (s[2] - (w / 2)) / w
        s[3] = (s[3] - (h / 2)) / h
        for i in range(4):
            temp[i] = s[i]

    for image in img_path:
        image_index += 1
        img = Image.open(image)
        img = data_transform(img)
        img = torch.unsqueeze(img, dim=0)
        with torch.no_grad():
            # img_feature维度为196*256
            output_3d = torch.squeeze(model(img.to(device))).cpu()
            output_2d = output_3d.view(256, -1)
            img_feature = output_2d.permute(1, 0)
            # 添加物体位置编码
            img_feature_list = torch.cat((img_feature, temp.unsqueeze(0)), dim=0)
            df = pd.DataFrame(img_feature_list.numpy())
            column_names = [str(i) for i in range(256)]
            df.columns = column_names

            # 保存数组到feather文件
            feather_file = str(image_index) + '.feather'
            # 要保存的文件夹路径
            folder_path = img_file

            # 如果文件夹不存在，可以先创建文件夹
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # 保存到指定文件夹中的 Feather 文件
            feather_file = os.path.join(folder_path, feather_file)

            df.to_feather(feather_file)

    # 从feather文件加载数组
    # feather_file = "your_file.feather"
    # df = pd.read_feather(feather_file)
