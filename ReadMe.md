# 僵尸画像及分类

该项目是第十一届全国服务外包大赛题目：A09——僵尸企业画像及分类。该项目共3个功能：

1. 僵尸企业识别——二分类
2. 僵尸企业分类——多分类
3. 非僵尸企业风险预测

第一个功能使用xgboost作为预测模型，采用stacking策略增强模型鲁棒性。具体实现方法可以查看`分类方法原理及参数调优概述.docx`

第二、三个功能不在机器学习范畴内，属于数学建模，根据已有论文做的。

运行代码的话 可以直接运行`predict.py`或者运行`train.py`查看准确度。

准确率、召回率、F1_score都有99.98%，最高到过100%，这么高的准确率我都怀疑是不是过拟合了，但是十折交叉验证之后，还是这么高，后期企业给了测试数据，还是100%，所以可能是数据有问题，用线性回归做baseline，净利润的权重是1，其他都是0.001左右，有很大的可能数据是伪造的。