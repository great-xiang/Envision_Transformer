import os
import torch
from vit import VisionTransformer
from PIL import Image
from torchvision import transforms
import cv2
import csv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 加载VIT
model = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, representation_size=None,
                          num_classes=1000).to(device)
model.load_state_dict(torch.load("vit_base_patch16_224.pth", map_location=device))
model.eval()

# 将图片的较短边缩小至224，再截取224*224的图像
data_transform = transforms.Compose(
    [transforms.Resize(224),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

file_path = 'data_set/OTB100'

# 打开每个文件
for img_file in os.listdir(file_path):
    print(img_file)
    img_path = []
    Label = []
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
            Label.append(s[i])

    for image in img_path:
        img = Image.open(image)
        img = data_transform(img)
        img = torch.unsqueeze(img, dim=0)
        with torch.no_grad():
            # 特征向量维度为768
            img_feature = torch.squeeze(model(img.to(device))).cpu()
            img_feature_list = img_feature.tolist()
            # 添加标签数据,添加四个，向量维度为772
            for i in range(4):
                img_feature_list.append(Label[i])
            del Label[0:4]
            # 保存数组到 CSV 文件
            with open(img_file + '.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(img_feature_list)

    # 从CSV文件加载数组
    # loaded_data = []
    # with open('data.csv', 'r') as f:
    #     reader = csv.reader(f)
    #     for row in reader:
    #         loaded_data.append([int(i) for i in row])
