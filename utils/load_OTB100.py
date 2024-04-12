import os
import cv2


# 传入文件路径，传出两个数组，第一个数组是图片路径，第二个数组是标签
def load_OTB100(file_path):
    img_path = []
    Label = []
    # 打开每个文件
    for img_file in os.listdir(file_path):
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
            Label.append(s)
    return img_path, Label

# print(load_OTB100('data_set/OTB100'))
