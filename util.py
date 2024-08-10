import os
import random

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.data import Data

import matplotlib.pyplot as plt


def gen_abspath(directory: str, rel_path: str) -> str:
    """由相对路径，生成绝对路径"""
    abs_dir = os.path.abspath(directory)
    return os.path.join(abs_dir, rel_path)


class Animator:
    """在动画中绘制数据"""

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):

        # 增量地绘制多条线
        if legend is None:
            legend = []
        self.use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]

        # 使用lambda函数捕获参数
        self.config_axes = lambda: self.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
        
    @staticmethod
    def use_svg_display():
        """Use the svg format to display a plot in Jupyter.

        Defined in :numref:`sec_calculus`"""
        from matplotlib_inline import backend_inline
        backend_inline.set_matplotlib_formats('svg')
    
    @staticmethod
    def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
        """Set the axes for matplotlib.

        Defined in :numref:`sec_calculus`"""
        axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
        axes.set_xscale(xscale), axes.set_yscale(yscale)
        axes.set_xlim(xlim),     axes.set_ylim(ylim)
        if legend:
            axes.legend(legend)
        axes.grid()

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


class Accumulator:
    """在n个变量上累加"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """

    def __init__(self, features, cuda=False, gcn=False): 
        """
        Initializes the aggregator for a specific graph.

        :param features: function mapping LongTensor of node ids to FloatTensor of feature values.
        :param cuda: whether to use GPU
        :param gcn: whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator, self).__init__()

        self.features = features
        self.cuda = cuda
        self.gcn = gcn

    def forward(self, nodes, to_neighs, num_sample=10):
        """
        :param nodes: list of nodes in a batch
        :param to_neighs: list of sets, each set is the set of neighbors for node in batch
        :param num_sample: number of neighbors to sample. No sampling if None.
        """

        # Local pointers to functions (speed hack)
        _set = set
        if not num_sample is None:
            # 对批量内每一个 node 的邻居，抽 num_sample 个样本
            _sample = random.sample
            samp_neighs = [_set(_sample(list(to_neigh), 
                            num_sample,
                            )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs

        if self.gcn:
            # 邻居节点 + 本节点
            samp_neighs = [samp_neigh + set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
        unique_nodes_list = list(set.union(*samp_neighs))

        # 建一个字典，存节点到节点编码的映射
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}

        # mask 是一个全零矩阵，shape 是 (批量大小, 无重复的节点数)
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))

        # 把邻居列表摊平，并将每个邻居元素换成它对应的编码 i
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]   
        
        # 一个数列，长度是 节点数 * 节点邻居数，相当于每个邻居在数列中被其节点的编号表示
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        
        # 一个稀疏矩阵，行表示节点，列表示邻居，节点与邻居之间有边的时候，对应矩阵元素值为 1
        mask[row_indices, column_indices] = 1
        if self.cuda:
            mask = mask.cuda()

        # 沿着 1 轴的方向（按行）求和，并保留维度
        num_neigh = mask.sum(1, keepdim=True)

        # 利用广播机制，求行平均
        mask = mask.div(num_neigh)

        # self.features 是将节点 id 转换为 节点特征的函数
        if self.cuda:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
        else:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))

        # 矩阵乘法 (节点个数, 所有邻居节点个数) @ (所有邻居节点个数, 特征维数) => （节点个数, 特征维数）
        to_feats = mask.mm(embed_matrix)

        return to_feats


class Encoder(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """

    def __init__(self, features, feature_dim, embed_dim,
                 adj_lists, aggregator, num_sample=10,
                 gcn=False, cuda=False):
        """初始化

        :param features: 特征矩阵
        :param feature_dim: 特征数
        :param embed_dim: 嵌入维度
        :param adj_lists: 节点间关联关系，被存成值为集合的字典
        :param aggregator: 聚合器，用于生成邻居节点的嵌入
        :param num_sample: 邻居节点抽样个数
        :param gcn: 是否仅使用关联信息
        :param cuda: 是否使用 cuda
        """
        super(Encoder, self).__init__()

        self.features = features
        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.aggregator = aggregator
        self.num_sample = num_sample

        self.gcn = gcn
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.aggregator.cuda = cuda
        self.weight = nn.Parameter(
            torch.FloatTensor(embed_dim, self.feat_dim if self.gcn else 2 * self.feat_dim))
        init.xavier_uniform_(self.weight)  # 对权重做初始化

    def forward(self, nodes):
        """
        Generates embeddings for a batch of nodes.

        :param nodes: list of nodes
        """
        # 邻居特征
        neigh_feats = self.aggregator.forward(nodes, [self.adj_lists[int(node)] for node in nodes], self.num_sample)
        # 表格特征
        if not self.gcn:
            if self.cuda:
                self_feats = self.features(torch.LongTensor(nodes).cuda())
            else:
                self_feats = self.features(torch.LongTensor(nodes))
            # 将邻居特征和表格特征 concat 起来
            combined = torch.cat([self_feats, neigh_feats], dim=1)
        else:
            # 只有邻居特征
            combined = neigh_feats
        # 全连接层加一个 ReLU 激活函数
        combined = F.relu(self.weight.mm(combined.t()))
        return combined


class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        # 获取节点的嵌入表示
        embeds = self.enc(nodes)
        # 全连接层
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())


def net(features, num_feats, adj_lists, label_cnt, use_cuda):
    agg1 = MeanAggregator(features, cuda=use_cuda)
    enc1 = Encoder(features, num_feats, 128, adj_lists, agg1, gcn=True, cuda=use_cuda)
    agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=use_cuda)
    enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
                   gcn=True, cuda=use_cuda)
    enc1.num_samples = 5
    enc2.num_samples = 5

    return SupervisedGraphSage(label_cnt, enc2)


def num_gpus():
    """Get the number of available GPUs"""
    return torch.cuda.device_count()


def load_cora(edge_path, feat_path):
    """加载 cora 数据集

    :param edge_path: cora 数据集边文件 
    :param feat_path: cora 数据集节点特征文件

    :return node_to_noi: 节点到节点下标的字典
    :return noi_to_feat: 节点下标到节点特征的字典
    :return noi_to_label: 节点下标到节点标签的字典
    :return label_map: 节点标签值到节点标签的字典
    """

    # noi: node_index
    node_to_noi = dict()
    noi_to_feat = dict()
    noi_to_label = dict()
    label_map = dict()
    with open(feat_path) as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()

            node_to_noi[info[0]] = i
            noi_to_feat[i] = [int(e) for e in info[1:-1]]
            if info[-1] not in label_map:
                label_map[info[-1]] = len(label_map)
            noi_to_label[i] = label_map[info[-1]]

    edge_list = list()
    with open(edge_path) as fp:
        for line in fp:
            info = line.strip().split()

            node_a = node_to_noi[info[0]]
            node_b = node_to_noi[info[1]]
            edge_list.append([node_a, node_b])

    return edge_list, noi_to_feat, noi_to_label, label_map


def create_pyg_cora_data(edge_list, feat_dict, label_dict):
    """把 cora 数据集存成 PyG data"""
    edge_index = torch.tensor(edge_list, dtype=torch.long)

    feat_list = [f[1] for f in sorted(feat_dict.items(), key=lambda e: e[0])]
    label_list = [l[1] for l in sorted(label_dict.items(), key=lambda e: e[0])]
    x = torch.tensor(feat_list, dtype=torch.float)
    y = torch.tensor(label_list, dtype=torch.long)

    return Data(x=x, y=y, edge_index=edge_index.t().contiguous())


def random_split_cora_data(data, train_rate, val_rate):
    """随机分割训练集、验证集和测试集"""
    assert train_rate + val_rate < 1
    N = data.num_nodes

    # 样本量
    num_train = int(N * train_rate)
    num_val = int(N * val_rate)
    num_test = N - num_train - num_val

    # 掩码初始化
    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask = torch.zeros(N, dtype=torch.bool)
    test_mask = torch.zeros(N, dtype=torch.bool)

    # 索引随机化
    perm = torch.randperm(N)

    # 按样本量分割数据集
    train_mask[perm[:num_train]] = True
    val_mask[perm[num_train:num_train+num_val]] = True
    test_mask[perm[num_train+num_val:]] = True

    # 将掩码添加到 data
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data


def train(model, optimizer, data, num_epoch):
    model.train()
    train_loss_list = list()
    val_loss_list = list()
    for epoch in range(num_epoch):
        optimizer.zero_grad()
        out = model(data)
        train_loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        train_loss.backward()
        optimizer.step()
        
        # 记录一下 train loss
        train_loss_list.append(train_loss.item())
    
        # 记录一下 validate loss
        val_loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask])
        val_loss_list.append(val_loss.item())

    return model, train_loss_list, val_loss_list


def plot_loss(train_loss, val_loss, title):
    plt.plot(train_loss, label='train loss')
    plt.plot(val_loss, label='val loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(title)
    plt.legend()
    plt.show()
