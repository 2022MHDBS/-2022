# 1.tree目录展示
!tree work/ -d

# 2.样本的可视化展示
import cv2
import matplotlib.pyplot as plt
%matplotlib inline
plt.imshow(cv2.imread("work/train_data/train.jpg"))

# 3.文本数据集：使用jieba分词,统计词频和词云可视化
# jieba进行文本数据的分词
import jieba

with open('work/train_data/1~18岁男童身高.txt', mode='r', encoding='utf-8') as f:
    text = f.read()
    seg_list = jieba.cut(text, cut_all=True)
    print("[Full Mode]" + "/ ".join(seg_list))     # 全模式

    seg_list = jieba.cut(text, cut_all=False)
    print("[Default Mode]" + "/ ".join(seg_list))  # 精确模式

    seg_list = jieba.cut_for_search(text)          # 搜索引擎模式
    print("[Search Mode]" + "/ ".join(seg_list))

# 导入飞桨
import paddle
# 输入数据：儿童年龄
data_x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,]
# 输入数据：儿童身高
data_y = [50.4,76.5,88.5,96.8,104.1,111.3,117.7,124.0,
130.0,135.4,140.2,145.3,151.9,159.5,165.9,169.8,171.6,172.3,172.7,]
# 定义网络
net = paddle.nn.Linear(in_features=1,out_features=1)
# 定义损失函数
loss_func = paddle.nn.MSELoss()
# 定义优化器
opt = paddle.optimizer.SGD(parameters=net.parameters())
# 组建预训练模型
for epoch in range(100):
    for x,y in zip(data_x,data_y):
        x = paddle.to_tensor([x],dtype="float32")
        y = paddle.to_tensor([y],dtype="float32")
        infer_y = net(x)
        loss = loss_func(infer_y,y)
        loss.backward()
        opt.step()
        opt.clear_gradients()
        print(f"Epoch: {epoch}\t loss: {loss.numpy()}")

# 测试模型效果
x = paddle.to_tensor([18],dtype="float32")
infer_y = net(x)
print(f"18岁对应的身高为：{infer_y.numpy()}cm。")