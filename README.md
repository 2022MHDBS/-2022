## 1.数据集
1.1 选择数据集
选择1至18岁男女标准身高数据集

1.2 项目背景/意义:
随着人们生活水平的不断提高，孩子们的身高也有了显著的提升，但是其中也包含着一些问题，例如，家长们如何知道自家孩子的身高在相应的年龄段是否偏矮，达标或者超高，以及如何通过记录孩子成长时期的身高去预测未来孩子的身高，如果发现孩子的身高出现了问题，应当及时采取相应的措施，保障孩子的健康成长。

1.3项目拟使用的方法：
采用传统自然语言处理的方法。

## 2.数据集预处理
（1）数据集清洗与介绍（解压数据集、tree命令查看目录结构、样本的可视化展示等）。

（2）文本数据集：使用jieba分词并统计词频。

In [31]
#### 1.tree目录展示
!tree work/ -d
work/
└── train_data

1 directory
In [33]
#### 2.样本的可视化展示
In [4]
import cv2
import matplotlib.pyplot as plt
%matplotlib inline
plt.imshow(cv2.imread("work/train_data/train.jpg"))

#### 3.文本数据集：使用jieba分词,统计词频和词云可视化
#### jieba进行文本数据的分词
import jieba

with open('work/train_data/1~18岁男童身高.txt', mode='r', encoding='utf-8') as f:
    text = f.read()
    seg_list = jieba.cut(text, cut_all=True)
    print("[Full Mode]" + "/ ".join(seg_list))     # 全模式

    seg_list = jieba.cut(text, cut_all=False)
    print("[Default Mode]" + "/ ".join(seg_list))  # 精确模式

    seg_list = jieba.cut_for_search(text)          # 搜索引擎模式
    print("[Search Mode]" + "/ ".join(seg_list))

#### jieba词频统计
from jieba import analyse

extract_tags = analyse.extract_tags(text, withWeight=True)
for i, j in extract_tags:
    print(i, j)

#### 词云可视化
from wordcloud import WordCloud

result = {}
for word in extract_tags:
    result[word[0]] = word[1]

wordcloud = WordCloud(
    background_color="white",
    max_font_size=50,
    font_path='work/train_data/simkai.ttf')
wordcloud.generate_from_frequencies(result)

plt.figure()
plt.axis('off')
plt.imshow(wordcloud)
plt.show()

## 3.选择模型并完成训练选择模型并完成训练
3.1 选择模型。
3.2 配置超参数并训练模型。
3.3 测试模型效果。

#### 导入飞桨
import paddle
#### 输入数据：儿童年龄
data_x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,]
#### 输入数据：儿童身高
data_y = [50.4,76.5,88.5,96.8,104.1,111.3,117.7,124.0,
130.0,135.4,140.2,145.3,151.9,159.5,165.9,169.8,171.6,172.3,172.7,]
#### 定义网络
net = paddle.nn.Linear(in_features=1,out_features=1)
#### 定义损失函数
loss_func = paddle.nn.MSELoss()
#### 定义优化器
opt = paddle.optimizer.SGD(parameters=net.parameters())
#### 组建预训练模型
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

#### 测试模型效果
x = paddle.to_tensor([18],dtype="float32")
infer_y = net(x)
print(f"18岁对应的身高为：{infer_y.numpy()}cm。")
18岁对应的身高为：[175.75696]cm。
## 4 总结
本项目希望通过对孩子身高的预测，去更好的帮助孩子的成长，梅西在10岁的时候身高才有1米25，11岁时被确认患有侏儒症，幸好得到了及时的治疗，成就了今天的伟大球星梅西。此外，本项目还有许多的不足之处，后续也将多多地学习，不断改进项目。
## 5 个人总结
感谢AI达人创造营。
AIStudio个人主页：https://aistudio.baidu.com/aistudio/usercenter
