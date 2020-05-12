import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adadelta
import sys
from torch.nn import init

def log(*args, **kwargs):
    print(file=sys.stdout, flush=True, *args, **kwargs)

class Bottle(nn.Module):
    ''' Perform the reshape routine before and after an operation '''

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0] * size[1], -1))
        return out.view(size[0], size[1], -1)

class XavierLinear(nn.Module):
    '''
    Simple Linear layer with Xavier init
    Paper by Xavier Glorot and Yoshua Bengio (2010):
    Understanding the difficulty of training deep feedforward neural networks
    http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    '''

    def __init__(self, in_features, out_features, bias=True):
        super(XavierLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_normal_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)

class OrthogonalLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(OrthogonalLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.orthogonal_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)

class BottledOrthogonalLinear(Bottle, OrthogonalLinear):
    pass

class BottledXavierLinear(Bottle, XavierLinear):
    pass

class HighWay(nn.Module):
    def __init__(self, size, num_layers=1, dropout_ratio=0.5):
        super(HighWay, self).__init__()
        self.size = size
        self.num_layers = num_layers
        self.trans = nn.ModuleList()
        self.gate = nn.ModuleList()
        self.dropout = dropout_ratio

        for i in range(num_layers):
            tmptrans = BottledXavierLinear(size, size)
            tmpgate = BottledXavierLinear(size, size)
            self.trans.append(tmptrans)
            self.gate.append(tmpgate)

    def forward(self, x):
        '''
        forward this module
        :param x: torch.FloatTensor, (N, D) or (N1, N2, D)
        :return: torch.FloatTensor, (N, D) or (N1, N2, D)
        '''

        g = F.sigmoid(self.gate[0](x))
        h = F.relu(self.trans[0](x))
        x = g * h + (1 - g) * x

        for i in range(1, self.num_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            g = F.sigmoid(self.gate[i](x))
            h = F.relu(self.trans[i](x))
            x = g * h + (1 - g) * x

        return x


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, edge_types, dropout=0.5, bias=True, use_bn=False,
                 device=torch.device("cpu")):
        """
        Single Layer GraphConvolution
        :param in_features: The number of incoming features
        :param out_features: The number of output features
        :param edge_types: The number of edge types in the whole graph
        :param dropout: Dropout keep rate, if not bigger than 0, 0 or None, default 0.5
        :param bias: If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        """
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.edge_types = edge_types
        self.dropout = dropout if type(dropout) == float and -1e-7 < dropout < 1 + 1e-7 else None
        # parameters for gates
        self.Gates = nn.ModuleList()
        # parameters for graph convolutions
        self.GraphConv = nn.ModuleList()
        # batch norm
        self.use_bn = use_bn
        if self.use_bn:
            self.bn = nn.BatchNorm1d(self.out_features)

        for _ in range(edge_types):
            self.Gates.append(BottledOrthogonalLinear(in_features=in_features,
                                                      out_features=1,
                                                      bias=bias))
            self.GraphConv.append(BottledOrthogonalLinear(in_features=in_features,
                                                          out_features=out_features,
                                                          bias=bias))
        self.device = device
        self.to(device)

    def forward(self, input, adj):
        """
        :param input: FloatTensor, input feature tensor, (batch_size, seq_len, hidden_size)
        :param adj: FloatTensor (sparse.FloatTensor.to_dense()), adjacent matrix for provided graph of padded sequences, (batch_size, edge_types, seq_len, seq_len)
        :return: output
            - **output**: FloatTensor, output feature tensor with the same size of input, (batch_size, seq_len, hidden_size)
        """

        adj_ = adj.transpose(0, 1)  # (edge_types, batch_size, seq_len, seq_len)
        ts = []
        for i in range(self.edge_types):
            gate_status = F.sigmoid(self.Gates[i](input))  # (batch_size, seq_len, 1)
            adj_hat_i = adj_[i] * gate_status  # (batch_size, seq_len, seq_len)
            ts.append(torch.bmm(adj_hat_i, self.GraphConv[i](input)))
        ts = torch.stack(ts).sum(dim=0, keepdim=False).to(self.device)
        if self.use_bn:
            ts = ts.transpose(1, 2).contiguous()
            ts = self.bn(ts)
            ts = ts.transpose(1, 2).contiguous()
        ts = F.relu(ts)
        if self.dropout is not None:
            ts = F.dropout(ts, p=self.dropout, training=self.training)
        return ts

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'