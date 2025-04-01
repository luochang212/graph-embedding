# graph-embedding

> GNN 处理图数据的方式还是很符合直觉的，基本沿袭了 CNN 的思路：每个神经元只看局部信息，通过层层汇聚掌握全貌。

本文做了什么：

- 对 GraphSAGE 的简单实现做逐行注释
- 在 Docker 环境运行 GraphSAGE 的原版示例
- 用 <a href="https://pytorch-geometric.readthedocs.io/en/latest/" target="_blank">PyG</a> 实现了 GCN 和 GAT
- 为运行 PyG 写了一些 pipeline 代码

✨ 注意：运行以下代码依赖 <a href="https://github.com/luochang212/graph-embedding/blob/main/util.py" target="_blank">util.py</a> 文件。

## 一、GraphSAGE 的简单实现

主流图算法大致分两种：

- 图嵌入算法 (GE): DeepWalk, Node2Vec 等
- 图神经网络算法 (GNN): GraphSAGE, GCN, GAT 等

### 1.1 绪论：图神经网络

图神经网络算法做的事，相当于把图这种复杂的数据结构，转换成低维向量，而低维向量往往是很好用的。

拿到图嵌入可以做很多事情，比如：

- 节点分类
- 链接预测
- 社区发现
- 相似度量

总之，图嵌入是一种非常有用的特征。在实践中，甚至可以将图嵌入和其他特征 concat 起来，训练更复杂的模型。

#### 1.1.1 GNN 和 CNN

GNN 和 CNN 的思路还挺像的，可以看作 CNN 在图数据上的推广。

CNN 有平移不变性和局部性。其中 **局部性** 指：每个神经元每次只看一小块像素，随着神经元层数的堆叠，层级越高的神经元，“看到”的像素量越多（如下图）。

![partial](/img/partial.svg)

假设 $C_1$ 能看到 4 个像素，$C_2$ 能看到 4 个像素。由于 $B_1$ 能看到 $C_1$ 和 $C_2$，相当于 $B_1$ 能够“看到” `4*2=8` 个像素。

依此类推，$A$ 能看到 $B_1$ 和 $B_2$，相当于能“看到” `8*2=16` 个像素。

层级越高的神经元，底下的“小弟”越多。尽管每个小弟只负责看一部分，但是高层级的神经元汇总了小弟们看见的部分，因此能够看到全局。这好比 merge 周报的老板，虽然细节知道的不如员工多，但比员工更了解公司的全貌。

#### 1.1.2 Aggregate 操作

图神经网络也有和 CNN 类似的汇聚过程。图神经网络会执行一种叫 Aggregate 的操作。虽然每次 Aggregate 只是把邻居节点的特征聚合到自己身上。但随着聚合次数增加，四跳五跳甚至十八跳节点的信息，也会一步一步“挪”过来。

更牛的是，Aggregate 是并行的。这意味着每轮迭代，所有节点都会收集邻居节点的信息。每个节点，都会构建一个属于自己的深度网络。而且随着迭代轮次增加，网络的深度和范围也会同步增加。

> 迭代次数与聚合范围的关系是线性的：
>
> - 迭代 1 个轮次，将 1 跳之内的节点信息聚合到本节点
> - 迭代 2 个轮次，将 2 跳之内的节点信息聚合到本节点
> - ... ...
> - 迭代 n 个轮次，将 n 跳之内的节点信息聚合到本节点

### 1.2 GraphSAGE 的基本思想

GraphSAGE 是一种可实时计算的图嵌入算法。这意味着它不需要知道节点所在的整张图的信息，只需要知道某节点 n 跳之内的拓扑关系和节点特征，就能计算该节点的嵌入。

它是如何做到这一点的呢？

GraphSAGE 的基本思想是：不学节点的下标，只学节点的特征。

|姓名|唱歌|跳舞|Rap|打篮球|
| -- | -- | -- | -- | -- |
|小真|1|1|1|1|
|小弥|1|1|1|1|
|小鲲|1|1|1|0|

如上表，我们只需要知道一个人是不是喜欢唱跳 rap 篮球就好了，至于这个人是谁，我们并不关心。

### 1.3 GraphSAGE 的简单实现

#### 1.3.1 Cora 数据集

[Cora 数据集](https://paperswithcode.com/dataset/cora) 是一个经典的图数据集，GraphSAGE 的作者也在 Demo 中用了这个数据集。

> **Note:** Cora 是一个论文引用关系数据集。它由 2708 篇科学论文组成。这些论文被分成 7 个类别，类别包括神经网络、强化学习、规则学习等。每篇论文由一个 1433 维的词向量表示，该向量的每一个元素对应一个词，元素值为 0 或 1，表示该词在论文中的是否存在。所有词均来自一个 1433 个词的字典。

Cora 包含两个数据文件：

- `cora.content`：记录论文的编号、特征和类别标签，每行代表一篇论文
- `cora.cites`：记录论文间的引用关系，每一行表示一条引用关系

#### 1.3.2 逐行注释

论文代码中定义了三个类：

|class|说明|
| -- | -- |
|`MeanAggregator`|聚合器，聚合邻居节点特征，相当于用邻居节点的表征来表示自己|
|`Encoder`|编码器，concat 当前节点的邻居特征和原始特征|
|`SupervisedGraphSage`|有监督学习器，用全连接层将嵌入结果 Embedding 与标号 label 连接，再用梯度下降更新 Embedding 以对齐 label |

这三个类很重要。只用这三个类，就可以把整个算法串起来。

1. **MeanAggregator** 的作用是将邻居节点特征转换为本节点的表征。由于是邻居的特征，可以把把这种表征记为节点的“关联表征”
2. 然后由 **Encoder** 将关联表征与原始特征 concat 起来。concat 后的结果输入带 ReLU 的全连接层。这一步的输出其实已经是节点的 Embedding 了。只不过还要经过监督学习的更新，才能保证 Embedding 具有良好的分类性能
3. 监督学习 **SupervisedGraphSage** 赋予 Embedding 分类性能。它用全连接层将图嵌入 Embedding 和标号 label 连接起来。经过多轮训练，最终得到包含 节点特征信息 和 关联信息，还对由 label 定义的类别具有良好分类性能的 Embedding


## 二、GraphSAGE 的原版实现

上一节我们尝试了 [GraphSAGE 的简单版实现](https://github.com/williamleif/graphsage-simple)。在简单实现中，一些与理解无关的复杂性被省略了。作为一名严肃的工程师，当然要跑跑 [原版 GraphSAGE](https://github.com/williamleif/GraphSAGE) 实现。

### 2.1 配置 Docker 环境

看了作者的 `requirements.txt` 里一堆不知道是啥还锁版本的包就头大，所幸作者提供了 Docker。

首先来到 Dockerfile 所在目录。

> **Note:** 如果你是中国大陆网友，请等一等，你需要先修改 Dockerfile，否则大概率会运行失败。
>
> 你需要打开 Dockerfile，为 pip 换源：
>
> ```
> # 用阿里源
> RUN pip install networkx==1.11 \
>     -i https://mirrors.aliyun.com/pypi/simple/
> ```

现在可以运行：

```
docker build -t graphsage .
```

一旦镜像安装完成，运行以下代码，启动容器：

```
# 映射到 9999 是因为我本地 8888 已经开了一个 Jupyter
# 如果你啥也没开，映射到 8888 就好
# docker run -it -p 8888:8888 graphsage

docker run -it -p 9999:8888 graphsage
```

容器启动后，在浏览器打开 [http://localhost:9999](http://localhost:9999)。网页会要求输入 token，用容器日志里那个就行。

### 2.2 运行原版示例

作者给了一些 demo，让我们能快速跑起来 ᕕ( ᐛ )ᕗ

#### 2.2.1 有监督学习

在容器提供的 Jupyter Notebook 里打开 Terminal，运行：

```
python -m graphsage.supervised_train \
    --train_prefix ./example_data/toy-ppi \
    --model graphsage_mean \
    --sigmoid
```

#### 2.2.2 无监督学习

```
python -m graphsage.unsupervised_train \
    --train_prefix ./example_data/toy-ppi \
    --model graphsage_mean \
    --max_total_steps 1000 \
    --validate_iter 10
```

没有问题，都成功了。唯一需要注意的是，源码 `ppi` 前漏了一个 `toy-`。应该是作者光改数据忘改代码了。


## 三、图卷积网络 GCN

- PyG: [conv.GCNConv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html#torch-geometric-nn-conv-gcnconv)
- arXiv: [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)
- GitHub: [tkipf/gcn](https://github.com/tkipf/gcn)

1. 加载数据的极简示例
2. 加载 Cora 数据集
3. 训练 GCN 模型
4. 预测与评估


## 四、图注意力网络 GAT

- PyG: [conv.GATConv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATConv.html#torch-geometric-nn-conv-gatconv)
- arXiv: [Graph Attention Networks](https://arxiv.org/abs/1710.10903)
- GitHub: [PetarV-/GAT](https://github.com/PetarV-/GAT)

1. 加载 Cora 数据集
2. 训练 GAT 模型
3. 预测与评估
