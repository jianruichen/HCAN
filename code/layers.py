import numpy as np
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter, scatter_mean
from torch_geometric.utils import softmax
from typing import Union, Tuple, Optional, Any
from torch import Tensor
from torch.nn.modules.module import Module

class GraphConvolution(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        #is_sparse
        output = torch.spmm(adj.to_dense(), support)
        row_sum = adj.to_dense().sum(dim=1, keepdim=True)
        output = output / row_sum
        if self.bias is not None:
            return output + self.bias.squeeze()
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class SpGraphConvolutionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout):
        super(SpGraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        h = torch.mm(input, self.W)
        # h= input
        edge_index, edge_value, edge_size = adj.coo()
        del adj
        edge = torch.concat((edge_index.unsqueeze(0), edge_value.unsqueeze(0)), dim=0)

        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*Dim x E

        edge_e = torch.ones(edge_h.size()[1])

        # edge_e = torch.nan_to_num(edge_e, nan=0.0)
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
        # e_rowsum: N x 1

        e_rowsum = e_rowsum + torch.where(e_rowsum == 0, 1.0, 0)
        # precess isolated node

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)

        # h_prime = torch.nan_to_num(h_prime, nan=0.0)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

def glorot(value: Any):
    if isinstance(value, Tensor):
        stdv = math.sqrt(6.0 / (value.size(-2) + value.size(-1)))
        value.data.uniform_(-stdv, stdv)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            glorot(v)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            glorot(v)

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):

        grad_output = torch.nan_to_num(grad_output, nan=0.0)

        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
            # 梯度裁剪
            max_grad_norm = 1.0  # 设置裁剪的梯度范数阈值
            grad_values = torch.clamp(grad_values, -max_grad_norm, max_grad_norm)

            grad_values = torch.nan_to_num(grad_values, nan=0.0)
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)

            max_grad_norm = 1.0
            grad_b = torch.clamp(grad_b, -max_grad_norm, max_grad_norm)

        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

    
class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
                
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        h = torch.mm(input, self.W)

        edge_index, edge_value, edge_size = adj.coo()
        del adj
        edge = torch.concat((edge_index.unsqueeze(0), edge_value.unsqueeze(0)), dim=0)

        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*Dim x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        # edge_e = torch.nan_to_num(edge_e, nan=0.0)
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
        # e_rowsum: N x 1

        e_rowsum = e_rowsum + torch.where(e_rowsum==0, 1.0, 0)
        # precess isolated node

        #get attention matrix
        attention_e = edge_e.clone().unsqueeze(1)
        attention_e = attention_e / torch.index_select(e_rowsum, 0, edge[0])

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        # h_prime = torch.nan_to_num(h_prime, nan=0.0)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime), edge, attention_e
        else:
            # if this layer is last layer,
            return h_prime, edge, attention_e

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        h = torch.mm(input, self.W)

        edge_index, edge_value, edge_size = adj.coo()
        del adj
        edge = torch.concat((edge_index.unsqueeze(0), edge_value.unsqueeze(0)), dim=0)

        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*Dim x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        # edge_e = torch.nan_to_num(edge_e, nan=0.0)
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
        # e_rowsum: N x 1

        e_rowsum = e_rowsum + torch.where(e_rowsum == 0, 1.0, 0)
        # precess isolated node

        # get attention matrix
        attention_e = edge_e.clone().unsqueeze(1)
        attention_e = attention_e / torch.index_select(e_rowsum, 0, edge[0])

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        # h_prime = torch.nan_to_num(h_prime, nan=0.0)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime), edge, attention_e
        else:
            # if this layer is last layer,
            return h_prime, edge, attention_e

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpGraph_Mul_AttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraph_Mul_AttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a1 = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a1.data, gain=1.414)
        self.a2 = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a2.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, args, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        h = torch.mm(input, self.W)

        edge = args.edge

        #node_i-->e_(p)
        p_h = torch.mm(args.p_h, self.W)
        edge_col, row_i = args.edge_col, args.row_i
        # edge_col_h = torch.cat((p_h[edge_col[0, :], :], p_h[edge_col[1, :], :]), dim=1).t()
        edge_col_h = torch.cat((h[edge_col[0, :], :], p_h), dim=1).t()      #
        edge_col_e = torch.exp(-self.leakyrelu(self.a1.mm(edge_col_h).squeeze()))
        edge_col_e = softmax(edge_col_e, row_i, num_nodes=N)

        #node_i_e_(p)-->e_m
        edge_row = edge.clone()

        row_resort = args.row_resort
        new_h = torch.mm(args.new_h, self.W)
        edge_row = torch.stack((row_resort, edge_row[1, :]), 0)           #node to order
        edge_row_h = torch.cat((new_h[edge_row[0, :], :], h[edge_row[1, :], :]), dim=1).t()  #
        edge_row_e = torch.exp(-self.leakyrelu(self.a2.mm(edge_row_h).squeeze()))

        edge_col_e_expand = torch.index_select(edge_col_e, 0, row_resort)
        edge_e = edge_row_e * edge_col_e_expand

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
        # e_rowsum: N x 1

        e_rowsum = e_rowsum + torch.where(e_rowsum == 0, 1.0, 0)
        # precess isolated node

        # get attention matrix
        attention_e = edge_e.clone().unsqueeze(1)
        attention_e = attention_e / torch.index_select(e_rowsum, 0, edge[0])

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        # h_prime = torch.nan_to_num(h_prime, nan=0.0)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime), edge, attention_e
        else:
            # if this layer is last layer,
            return h_prime, edge, attention_e

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'