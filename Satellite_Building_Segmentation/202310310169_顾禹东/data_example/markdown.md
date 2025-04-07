# 数据集说明
## 数据集来源
#####  网址： https://project.inria.fr/aerialimagelabeling/

## 数据集样本总数
##### Inria Aerial Image Labeling Dataset(360张卫星图，512x512分辨率)
 
## 数据集任务类型
##### 1.dataset/images & dataset/masks：主要划分为训练集和验证集两部分
训练集用来模型训练模型分割，主要用于对于卫星遥感图中建筑物的目标检测及分割，验证集用来验证模型的准确率
##### 2.dataset/tests：主要作为测试集来模型预测输出结果
测试集用来通过模型输出结果

## 训练集，验证集和测试集划分
##### 1.训练集和验证集在代码中划分，训练集和验证集的比例为8:2，统一在dataset/images和dataset/masks里划分
##### 2.验证集因为下载数据集时直接给我了测试集这个文件夹所以我直接用了