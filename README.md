## <div style="text-align:center;">Visual Object Tracking Transformer With Time Token</div>
<div style="text-align:center;"> auther：Xiang shihao </div>

1. 模型工作原理，将图像输入resnet，获得特征图，将多帧连续图像特征图、物体坐标作为模型输入，加入时间编码、位置编码，最终预测物体位置
2. track_trans.py是模型定义文件，train.py为训练文件，predict.py为预测文件，utils.py为工具文件，包含数据集加载器和将来的一些其它文件，
3. utils文件夹下有一些预处理文件,resnet和vit文件是图像预处理模型，deal_OTB_res.py是使用resnet处理OTB数据集
4. models文件夹下有训练好的一个模型
5. data_set文件夹下是数据集
