import numpy as np
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import GATConv
from torch_scatter import scatter_add,scatter
import math
from torch.nn import Parameter
from torch_geometric.nn.dense.linear import Linear
from torch_sparse import SparseTensor, set_diag

from torch_geometric.utils import softmax
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *

class GCN(nn.Module):
    def __init__(self, input, hid, output, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(input, hid)
        self.gc2 = GraphConvolution(hid, output)
        self.dropout = dropout

    def forward(self, adj, x):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

class GAT(nn.Module):
    def __init__(self, num_layers, init_dim, hid_dim, embed_dim, nheads, params=None):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.p = params
        self.dropout = self.p.dropout
        self.alpha = self.p.alpha

        self.attentions = [GraphAttentionLayer(init_dim, hid_dim, dropout=self.dropout, alpha=self.alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.hid_attentions = []
        for i in range(num_layers - 1):
            hid_attention = [GraphAttentionLayer(hid_dim, hid_dim, dropout=self.dropout, alpha=self.alpha, concat=True) for _ in range(nheads)]
            self.hid_attentions.append(hid_attention)

        self.out_att = GraphAttentionLayer(hid_dim * nheads, embed_dim, dropout=self.dropout, alpha=self.alpha, concat=False)

    def forward(self, adj, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        for i in range(len(self.hid_attentions)):
            x = torch.cat([att(x, adj) for att in self.hid_attentions[i]], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x

class SpGCN(nn.Module):
    def __init__(self, init_dim, hid_dim, embed_dim, dropout):
        """Sparse version of GAT."""
        super(SpGCN, self).__init__()
        self.dropout = dropout

        self.gc1 = SpGraphConvolutionLayer(init_dim, hid_dim, dropout=self.dropout)
        self.gc2 = SpGraphConvolutionLayer(hid_dim, embed_dim, dropout=self.dropout)

    def forward(self, adj, x):
        x = (self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = (self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        # x = F.relu(x)
        return x


class SpGAT(nn.Module):
    def __init__(self, num_layers, init_dim, hid_dim, embed_dim, nheads, params=None):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.p = params
        self.dropout = self.p.dropout
        self.alpha = self.p.alpha

        self.attentions = [SpGraphAttentionLayer(init_dim,
                                                 hid_dim,
                                                 dropout=self.dropout,
                                                 alpha=self.alpha,
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.hid_attentions = []
        for i in range(num_layers - 1):
            hid_attention = [SpGraphAttentionLayer(hid_dim,
                                                     hid_dim,
                                                     dropout=self.dropout,
                                                     alpha=self.alpha,
                                                     concat=True) for _ in range(nheads)]
            self.hid_attentions.append(hid_attention)

        self.out_att = SpGraphAttentionLayer(hid_dim * nheads,
                                             embed_dim,
                                             dropout=self.dropout,
                                             alpha=self.alpha,
                                             concat=False)

    def forward(self, adj, x):
        x = F.dropout(x, self.dropout, training=self.training)
        # x = torch.cat([att(x, adj) for att in self.attentions], dim=1)

        h = torch.tensor([])
        for att in self.attentions:
            h_prime, edge, attention_in = att(x, adj)
            h = torch.cat((h, h_prime), dim=1)
        x = h

        # if have hidden layers
        for i in range(len(self.hid_attentions)):
            h = torch.tensor([])
            for att in self.hid_attentions[i]:
                h_prime, edge, attention_hid = att(x, adj)
                h = torch.cat([h, h_prime], dim=1)
            x = h

        x = F.dropout(x, self.dropout, training=self.training)
        h_prime, edge, attention_out = self.out_att(x, adj)
        x = F.elu(h_prime)
        return x, edge, attention_out

class SpMGAT(nn.Module):
    def __init__(self, num_layers, init_dim, hid_dim, embed_dim, nheads, params=None):
        """Sparse version of GAT."""
        super(SpMGAT, self).__init__()
        self.p = params
        self.dropout = self.p.dropout
        self.alpha = self.p.alpha

        self.attentions = [SpGraph_Mul_AttentionLayer(init_dim,
                                                 hid_dim,
                                                 dropout=self.dropout,
                                                 alpha=self.alpha,
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.hid_attentions = []
        for i in range(num_layers - 1):
            hid_attention = [SpGraph_Mul_AttentionLayer(hid_dim,
                                                     hid_dim,
                                                     dropout=self.dropout,
                                                     alpha=self.alpha,
                                                     concat=True) for _ in range(nheads)]
            self.hid_attentions.append(hid_attention)

        self.out_att = SpGraph_Mul_AttentionLayer(hid_dim * nheads,
                                             embed_dim,
                                             dropout=self.dropout,
                                             alpha=self.alpha,
                                             concat=False)

    def forward(self, adj, x, args):
        x = F.dropout(x, self.dropout, training=self.training)
        # x = torch.cat([att(x, adj) for att in self.attentions], dim=1)

        h = torch.tensor([])
        for att in self.attentions:
            h_prime, edge, attention_in = att(args, x, adj)
            h = torch.cat((h, h_prime), dim=1)
        x = h

        # if have hidden layers
        for i in range(len(self.hid_attentions)):
            h = torch.tensor([])
            for att in self.hid_attentions[i]:
                h_prime, edge, attention_hid = att(args, x, adj)
                h = torch.cat([h, h_prime], dim=1)
            x = h

        x = F.dropout(x, self.dropout, training=self.training)
        h_prime, edge, attention_out = self.out_att(args, x, adj)
        x = F.elu(h_prime)
        return x, edge, attention_out


class HighAgg(nn.Module):
    """
    Aggregation functions to aggregate individuals' embeddings to higher_order embeddings.
    """
    def __init__(self, args):
        super(HighAgg, self).__init__()
        # For:　author and item liner layers
        self.embed_dim = args.embed_dim #* 2  #concat *2
        self.node_weight = Linear(self.embed_dim, self.embed_dim)
        self.high_weight = Linear(self.embed_dim, self.embed_dim)
        self.node_weight.reset_parameters()
        self.high_weight.reset_parameters()

        self.heads = args.att_head
        self.att_src = Parameter(torch.Tensor(1, self.heads, self.embed_dim))
        self.att_dst = Parameter(torch.Tensor(1, self.heads, self.embed_dim))
        self.lin_src = Linear(self.embed_dim, self.heads * self.embed_dim, False, weight_initializer='glorot')
        self.lin_dst = Linear(self.embed_dim, self.heads * self.embed_dim, False, weight_initializer='glorot')
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)

    def forward(self, n_features, n2h_graph, h_features=None):
        nn_features = torch.index_select(n_features, 0, n2h_graph[0])
        h_features = scatter(nn_features, n2h_graph[1], dim=0, reduce='mean')  # average individual emb to get initial h_features

        h_high = torch.index_select(h_features, 0, n2h_graph[1])      #high_order feature of every node
            #h_high = scatter(nn_high_features, n2h_graph[0], dim=0, reduce='mean')     #get all nodes included in higher_order

        h_node = self.lin_src(nn_features).view(-1, self.heads, self.embed_dim) #W*h_nodei
        h_high = self.lin_dst(h_high).view(-1, self.heads, self.embed_dim)     #W*h_high

        #n2h_agg
        alpha_src = (h_high * self.att_src).sum(-1)  # a^T * [W*h_high]
        alpha_dst = (h_node * self.att_dst).sum(-1)  # a^T * [W*h_nodei]
        alpha = alpha_src + alpha_dst
        alpha = F.leaky_relu(alpha, 0.2)  # Leaky_relu(a^T * [W*h_high +W*h_nodei])
        alpha = softmax(alpha, n2h_graph[1], num_nodes=int(n2h_graph[0].size(0))) #an attention of a node inside a higher order to that higher order; sigma operation
        #

        h_node = h_node * alpha.unsqueeze(-1)  # alpha * h_nodei
        h_node = h_node.mean(dim=1)  # (n_edge, dim); get mean values in all attention views
        h_features = scatter(h_node, n2h_graph[1], dim=0, reduce='sum') #sigma operation
        h_features = self.high_weight(h_features)   # W
        return h_features, alpha



class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        if args.Is_sparse == True:
            self.GraphBase_1 = SpGAT(args.num_layers, args.init_dim, args.hid_dim, args.embed_dim, args.head, args)
            self.GraphBase_2 = SpMGAT(args.num_layers, args.init_dim, args.hid_dim, args.embed_dim, args.head, args)
            self.GraphBase_0 = SpGCN(args.init_dim, args.hid_dim, args.embed_dim, args.gcn_dropout).requires_grad_(False)
        else:
            self.GraphBase_1 = GAT(args.num_layers, args.init_dim, args.hid_dim, args.embed_dim, args.head, args)
            self.GraphBase_2 = GAT(args.num_layers, args.init_dim, args.hid_dim, args.embed_dim, args.head, args)
            self.GraphBase_0 = GCN(args.init_dim, args.hid_dim, args.embed_dim, args.gcn_dropout).requires_grad_(False)
        self.id_num = args.node_number
        self.high_num = args.high_number
        self.node_embedding_layer = nn.Embedding(self.id_num, args.init_dim)
        self.node_embedding_layer.reset_parameters()
        self.high_embedding_layer = nn.Embedding(self.high_num, args.init_dim)
        self.high_embedding_layer.reset_parameters()
        self.n2h_agg = HighAgg(args)


    def forward(self, args, n2n_graph, n2h_graph, n2h_graph_neg, train_model=True):
        # n2h_graph_y = n2h_graph

        node_num = args.node_number
        n_features = self.node_embedding_layer(torch.LongTensor([idx for idx in range(node_num)]).to(args.device))
        if(train_model == True):
            h_features = self.high_embedding_layer(torch.LongTensor([idx for idx in range(args.train_high_number)]).to(args.device))
        else:
            h_features = self.high_embedding_layer(torch.LongTensor([idx for idx in range(args.train_high_number, args.high_number)]).to(args.device))

        features_union = torch.cat([n_features, h_features], dim=0)

        graph_union, range_len = args.graph_union, args.range_len

        col_h = torch.index_select(features_union, 0, args.edge_col_y[1, :])  # Characterization of the corresponding order for subsequent finding of the mean value

        p_h = scatter(col_h, args.new_indices, dim=0, reduce='max') - scatter(col_h, args.new_indices, dim=0, reduce='min')   #max-min

        row_indices = torch.arange(features_union.size(0))
        new_h = features_union[row_indices.repeat_interleave(torch.tensor(args.expand_counts))]  #

        args.p_h, args.new_h =p_h, new_h

        features_union_1, edge_view1, attention_1 = self.GraphBase_1(graph_union, features_union)

        features_union_2, edge_view2, attention_2 = self.GraphBase_2(graph_union, features_union, args)

        features_union_0 = self.GraphBase_0(graph_union, features_union)
        # no info_loss

        # features_union = torch.cat((features_union_1, features_union_2), dim=1)
        features_union = (features_union_1 + features_union_2)

        #single layer
        # features_union = features_union_1

        n_features = features_union[:node_num]
        h_features = features_union[node_num:]

        #agg together
        h_features_all, alpha = self.n2h_agg(n_features, n2h_graph_neg, h_features)

        if(train_model == True):
            return n_features, h_features_all, features_union_0, features_union_1, features_union_2, edge_view1, attention_1, attention_2
        else:
            return n_features, h_features_all


class Predictor(nn.Module):
    def __init__(self, args, pre_node=None):
        super(Predictor, self).__init__()
        self.dropout = args.dropout1
        self.sm = nn.Sigmoid()
        if(pre_node == True):
            dim = args.embed_dim * 2 #* 2  #concat时 *2
            self.predict = torch.nn.Linear(dim, 1).requires_grad_(False)
            self.predict.weight.data.uniform_(1. / (dim), 1. / (dim))

        else:
            dim = args.embed_dim #* 2
            self.predict = torch.nn.Linear(dim, 1).requires_grad_(False)
            self.predict.weight.data.uniform_(1. /dim, 1. /dim)

    def forward(self, x):
        x = F.sigmoid(self.predict(x))
        return x
