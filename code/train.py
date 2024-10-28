import random
import time
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import scipy.sparse as sp
import logging
from utils import *
from models import *
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import warnings
warnings.filterwarnings(action='ignore')
start = time.time()

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default="../data/", help='Path of loading Data.')
parser.add_argument('--task', type=str, default="predict", help="available task: [predict, class]")
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='[52, 42, 32, 22, 12]')
parser.add_argument('--epochs', type=int, default=501, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--Is_sparse', type=bool, default=True, help='Is sparse matrix?')
parser.add_argument('-dataset', default='email-Eu', type=str, help='email-Eu, email-Enron, congress-bills, tags-math-sx, tags-ask-ubuntu,'
                                                                      'contact-primary-school, contact-high-school, DAWN, NDC-substances')
parser.add_argument('-node_number', default='979', type=int, help='[979, 143, 1718, 1627, 3021, 242, 327, 2290, 3438]')
parser.add_argument('-temper', default=24.0, type=float, help='[32.0, 24.0, 16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.25]')
parser.add_argument('-gcn_dropout', default=0.2, type=int, help='[0.1, 0.2, 0.3, 0.4, 0.5]')
# #hns
# temper_list = [24.0, 16.0, 1.0, 4.0, 0.5, 2.0, 2.0, 8.0, 24.0]
# gcn_drop_list = [0.2, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.1]
# #mns
# # temper_list = [1.0, 16.0, 1.0, 4.0, 8.0, 2.0, 2.0, 2.0, 8.0]
# # gcn_drop_list = [0.5, 0.1, 0.1, 0.1, 0.1, 0.0, 0.1, 0.3, 0.1]
# #lns
# # temper_list = [2.0, 2.0, 32.0, 24.0, 24.0, 2.0, 0.25, 0.25, 1.0]
# # gcn_drop_list = [0.1, 0.1, 0.4, 0.1, 0.1, 0.1, 0.1, 0.3, 0.1]
#GAT layer
parser.add_argument('--num_layers', type=int, default=1, help='GAT_hidden_layers_number.')
parser.add_argument('--init_dim', type=int, default=128, help='Dimensions of initial features.')  #32
parser.add_argument('--hid_dim', type=int, default=128, help='Dimensions of hid features.')  #32
parser.add_argument('--embed_dim', type=int, default=128, help='Dimensions of embed features.')  #32
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=int, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--head', type=int, default=1, help='GAT_heads.')
#predict module
parser.add_argument('--mlp_hidden_dim', type=int, default=32, help='Dimensions of hidden units.')   #three layers mlp
parser.add_argument('--mlp_hidden_dim1', type=int, default=16, help='Dimensions of hidden units.')
parser.add_argument('--mlp_output_dim', type=int, default=1, help='Dimensions of output layer.')
parser.add_argument('--dropout1', type=float, default=0.5, help='Dropout rate (1 - keep probability).')

parser.add_argument('-view_num', default=4, type=int, help='View number')
parser.add_argument('-datasize', default=0.7, type=float, help='Train set ratio')
parser.add_argument('-att_head', default=1, type=int, help='Head number of att')

parser.add_argument('-max_order', default=0, type=int, help='order')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# load gpu
device = torch.device("gpu" if torch.cuda.is_available() else "cpu")
args.device = device

dataset = args.dataset
logging.info("--------------{}------------------".format(dataset))
args.dataset = dataset
node_number = args.node_number
print('==' * 40)
print('Loading {} dataset...'.format(dataset))
print("-----seed:{}------".format(args.seed))


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
seed_everything(seed=args.seed)

def train(epoch):
    model.train()
    predictor_high.train()
    predictor_node.train()
    optimizer.zero_grad()
    h_optimizer.zero_grad()
    n_optimizer.zero_grad()


    #disfusion
    high_train = pos_high_train + neg_high_train
    train_labels = torch.concat(
        (torch.unsqueeze(torch.ones(len(pos_high_train)), 1), torch.unsqueeze(torch.zeros(len(neg_high_train)), 1)),
        dim=0)
    sample_idx = np.array([i for i in range(len(high_train))])
    np.random.shuffle(sample_idx)
    high_train_shuffle = [high_train[i] for i in sample_idx]
    train_labels = train_labels[sample_idx]
    n2h_graph_train, n2h_dict_train = construct_incidence(args, high_train_shuffle)
    n2h_graph_train = torch.LongTensor(n2h_graph_train).t().to(device)

    # model
    n_features, h_features_all, features_union_0, features_union_1, features_union_2, edge, attention_1, attention_2 \
        = model(args, nn_graph, n2h_graph, n2h_graph_train, train_model=True)
    n_features, h_features_all, features_union_0, features_union_1, features_union_2 = \
        n_features.to(device), h_features_all.to(device), features_union_0.to(device), features_union_1.to(
            device), features_union_2.to(device)


    # InfoNCE loss
    loss_info = InfoNCE_loss(features_union_0, features_union_1, features_union_2, temperature=args.temper)  # 8.0
    # print("info_loss", loss_info)

    train_output = predictor_high(h_features_all)
    high_loss = compute_loss(args, train_output, train_labels)

    loss_train = high_loss + loss_info  # + node_loss
    loss_train.backward()
    optimizer.step()
    h_optimizer.step()
    n_optimizer.step()

    # evaluation

    AUC_train, AUC_PR_train, AP_train, precision_train, recall_train, f1_train = metrics(train_output.cpu().detach(),
                                                                                         train_labels)
    # print(f'EPOCH[{epoch}/{args.epochs}]')
    # print(f"[train_lOSS{epoch}] {loss_train}",
    #       f"[train_AUC{epoch}] {AUC_train, AP_train, AUC_PR_train}")

    # test
    loss_test, AUC_test, AUC_PR_test, AP_test, precision_test, recall_test, f1_test, \
    alpha, n2h_graph_test, test_labels, predict_labels, sample_idx = epoch_test(n_features)
    # print('==' * 27)
    # print(f"[test_lOSS{epoch}] {loss_test}",
    #       f"[test_AUC{epoch}] {AUC_test, AP_test, AUC_PR_test}"
    #       )
    # print('==' * 27)

    data_dict = {"alpha": alpha, "n2h_graph_test": n2h_graph_test, "test_labels": test_labels,
                 "predict_labels": predict_labels, "sample_idx": sample_idx,
                 "attention_1": attention_1, "attention_2": attention_1}

    metrics_dict = {'AUC_train': AUC_train, "AUC_PR_train": AUC_PR_train, "AP_train": AP_train,
                    "AUC_test": AUC_test, "AP_test": AP_test, "AUC_PR_test": AUC_PR_test,
                    "precision_test": precision_test, "recall_test": recall_test, "f1_test": f1_test,
                    }

    return loss_train, loss_test, data_dict, metrics_dict


def epoch_test(n_features):
    model.eval()
    predictor_high.eval()
    predictor_node.eval()

    # shuffle
    high_test = pos_high_test + neg_high_test
    test_labels = torch.concat(
        (torch.unsqueeze(torch.ones(len(pos_high_test)), 1), torch.unsqueeze(torch.zeros(len(neg_high_test)), 1)),
        dim=0)
    sample_idx = np.array([i for i in range(len(high_test))])
    np.random.shuffle(sample_idx)
    high_test_shuffle = [high_test[i] for i in sample_idx]
    test_labels = test_labels[sample_idx]
    n2h_graph_test, n2h_dict_test = construct_incidence(args, high_test_shuffle)
    n2h_graph_test = torch.LongTensor(n2h_graph_test).t().to(device)
    h_features = None
    h_features_all, alpha = model.n2h_agg(n_features, n2h_graph_test, h_features)

    test_output = predictor_high(h_features_all)
    high_loss = compute_loss(args, test_output, test_labels)

    # node loss
    binary_edges_ = torch.LongTensor(binary_edges).t()
    edge_features = torch.concat((n_features[binary_edges_[0]], n_features[binary_edges_[1]]), dim=1)
    node_pos_out = predictor_node(edge_features)
    node_pos_loss = -torch.log(node_pos_out + 1e-15).mean()
    # random sampling.
    neg_edge = torch.randint(0, args.node_number, binary_edges_.size(), dtype=torch.long)
    neg_edge_features = torch.concat((n_features[binary_edges_[0]], n_features[neg_edge[1]]), dim=1)
    node_neg_out = predictor_node(neg_edge_features)
    node_neg_loss = -torch.log(1 - node_neg_out + 1e-15).mean()
    node_loss = node_pos_loss + node_neg_loss

    loss_test = high_loss + node_loss

    # evaluation
    AUC_test, AUC_PR_test, AP_test, precision_value, recall_value, f1_value = metrics(test_output.cpu().detach(),
                                                                                      test_labels)

    # acc
    test_output_ = np.array(test_output.cpu().detach())
    div = np.mean(test_output_, axis=0)
    test_output_[test_output_ >= div] = 1.
    test_output_[test_output_ < div] = 0.

    return loss_test, AUC_test, AUC_PR_test, AP_test, precision_value, recall_value, f1_value, alpha, n2h_graph_test, test_labels, test_output_, sample_idx




# #load dataset
binary_edges, pos_high_train, pos_high_test, ii_graph, idx2high, i2h_graph, i2h_dict, node_degree = load_data(args,
                                                                                                              dataset)

# generate/load neg_sample
neg_type = 'hns'

    # first generate clu_att
    # neg_high_train, neg_high_test = process_sample_attention(args, binary_edges, pos_high_train, pos_high_test, node_degree, neg_type)

#load neg_sample
neg_high_train, neg_high_test = load_neg(args, neg_type)

# pos_sample
n2h_graph, n2h_dict = construct_incidence(args, pos_high_train)
nn_graph = torch.LongTensor(ii_graph).t().to(device)
n2h_graph = torch.LongTensor(n2h_graph).t().to(device)

high_index = []
for sub_list in pos_high_train:
    high_index.append(len(sub_list))
args.high_index = high_index

##order_attention' reprocess
args = mul_attention(args, nn_graph, n2h_graph, node_number)

model = Model(args)
#
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#
predictor_high = Predictor(args)
h_optimizer = torch.optim.Adam(predictor_high.parameters(), lr=args.lr, weight_decay=args.weight_decay)
predictor_node = Predictor(args, pre_node=True)
n_optimizer = torch.optim.Adam(predictor_node.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#
model, predictor_high, predictor_node = model.to(device), predictor_high.to(device), predictor_node.to(device)

# Train and test model
t_total = time.time()
loss_train_vector = []
AUC_vector, AP_vector = [], []
loss_test_vector = []
AUC_test_, AP_test_ = [], []

precision_test_list, recall_test_list, f1_test_list = [], [], []

for epoch in tqdm(range(args.epochs)):
    loss_train, loss_test, data_dict, metrics_dict = train(epoch)

    loss_train_vector.append(loss_train.data.cpu())

    AUC_vector.append(metrics_dict["AUC_train"])
    AP_vector.append(metrics_dict["AP_train"])
    AUC_test_.append(metrics_dict["AUC_test"])
    AP_test_.append(metrics_dict["AP_test"])

    loss_test_vector.append(loss_test.data.cpu())

    precision_test_list.append(metrics_dict["precision_test"])
    recall_test_list.append(metrics_dict["recall_test"])
    f1_test_list.append(metrics_dict["f1_test"])
    torch.cuda.empty_cache()

a = max(AUC_test_)
max_epoch = -1
for i in range(len(AUC_test_)):
    if (AUC_test_[i] == a):
        if (max_epoch < 0):
            max_epoch = i
        print("AUC:{:.4f}".format(max(AUC_test_)), "AP:{:.4f}".format(AP_test_[i]),
              "f1:{:.4f}".format(max(f1_test_list)))

