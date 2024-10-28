import numpy as np
import math
import copy
import random
from itertools import combinations
from torch_scatter import scatter_add,scatter
from torch_sparse import SparseTensor, set_diag

import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, precision_score, recall_score, f1_score, \
    accuracy_score, average_precision_score, silhouette_score
from sklearn.cluster import KMeans, DBSCAN
import scipy.sparse as sp
import torch
import pandas as pd


def metrics(output, order_labels):
    AUC_value = roc_auc_score(order_labels, output)
    precision, recall, thresholds = precision_recall_curve(order_labels, output)
    AUC_PR_value = auc(recall, precision)
    AP_value = average_precision_score(order_labels, output)

    output_binary = (output >= output.median()).int()
    precision_value = precision_score(order_labels, output_binary)
    recall_value = recall_score(order_labels, output_binary)
    f1_value = f1_score(order_labels, output_binary)

    return AUC_value, AUC_PR_value, AP_value, precision_value, recall_value, f1_value

# reprocess_raw(args, dataset, node_number)

def reprocess_raw(args, dataset, node_number):
    path = args.path
    print('Loading {} dataset...'.format(dataset))
    nodes = np.genfromtxt("{}{}/{}-nverts.txt".format(path, dataset, dataset))  # dtype=np.dtype(int)
    simplices = np.genfromtxt("{}{}/{}-simplices.txt".format(path, dataset, dataset))
    times = np.genfromtxt("{}{}/{}-times.txt".format(path, dataset, dataset))
    count = 0
    simplices_list = []
    for num in nodes:
        num = int(num)
        simplices_list.append(np.sort(simplices[count: count + num]).tolist())
        count = count + num
    del simplices
    #de-duplicate
    nodes_u = []
    simplices_list_u = []
    times_u = []
    for i, simplice in enumerate(simplices_list):
        if(simplice not in simplices_list_u and len(simplice) > 1):
            nodes_u.append(nodes[i])
            simplices_list_u.append(simplice)
            times_u.append(times[i])

    #judge new node number
    unique_numbers = set()
    for sublist in simplices_list_u:
        for number in sublist:
            unique_numbers.add(number)
    unique_numbers = np.array(list(unique_numbers))
    id2node = {i: unique_number for i, unique_number in enumerate(unique_numbers)}
    node2id = {unique_number: i for i, unique_number in enumerate(unique_numbers)}
    for i, sublist in enumerate(simplices_list_u):
        for index, item in enumerate(sublist):
            simplices_list_u[i][index] = node2id[item]
    #filter out


    #write file
    file_name = "{}{}/{}-nverts_new.txt".format(path, dataset, dataset)
    with open(file_name, "w") as file:
        for item in nodes_u:
            file.write(str(item) + "\n")
    file_name = "{}{}/{}-simplices_new.txt".format(path, dataset, dataset)
    with open(file_name, "w") as file:
        for item in simplices_list_u:
            for i in item:
                file.write(str(i) + "\n")
    file_name = "{}{}/{}-times_new.txt".format(path, dataset, dataset)
    with open(file_name, "w") as file:
        for item in times_u:
            file.write(str(item) + "\n")


def read_file_and_create_list(file_path):
    result_list = []
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        for line in lines:
            numbers = [int(float(num)) for num in line.strip().split()]
            result_list.append(numbers)

        return result_list
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def load_data(args, dataset):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    path = args.path
    # print('==' * 40)
    # print('Loading {} dataset...'.format(dataset))
    nodes = np.genfromtxt("{}{}/{}-nverts_new.txt".format(path, dataset, dataset))  # dtype=np.dtype(int)
    simplices = np.genfromtxt("{}{}/{}-simplices_new.txt".format(path, dataset, dataset))
    times = np.genfromtxt("{}{}/{}-times_new.txt".format(path, dataset, dataset))
    count = 0
    simplices_list = []
    for num in nodes:
        num = int(num)
        simplices_list.append(np.sort(simplices[count: count + num]).tolist())
        count = count + num
    del simplices

    # binary_edges_num = np.sum(nodes == 2)
    # high_order_num = np.sum(nodes > 2)

    binary_edges_train, binary_edges_test, high_order_train, high_order_test = [], [], [], []
    #div according times
    div = int(len(times) * args.datasize)
    for i, sub_list in enumerate(simplices_list):
        if(i < div and len(sub_list) == 2):
             binary_edges_train.append(sub_list)
        elif(i >= div and len(sub_list) == 2):
             binary_edges_test.append(sub_list)
        elif(i < div and len(sub_list) > 2):
            high_order_train.append(sub_list)
        elif(i >= div and len(sub_list) > 2):
            high_order_test.append(sub_list)
    binary_edges = binary_edges_train + binary_edges_test

    #filter out all then div 70%
    high_order = high_order_train + high_order_test
    high_order_copy = copy.deepcopy(high_order)
    order_temp = 0
    for sub_list in high_order_copy:
        num_combinations = math.comb(len(sub_list), 2)
        if(len(sub_list) > order_temp):
            order_temp = len(sub_list)
        for pair_edges in combinations(sub_list, 2):
            if(list(pair_edges) not in binary_edges):
                break
            else:
                num_combinations -= 1
                if(num_combinations == 0):
                    high_order.remove(sub_list)
    del high_order_copy
    args.max_order = order_temp

    unique_numbers = set()
    for sublist in (binary_edges + high_order):
        for number in sublist:
            unique_numbers.add(number)
    args.node_number = len(unique_numbers)

    high_order_train = high_order[:int(len(high_order) * 0.7)]
    high_order_test = high_order[int(len(high_order) * 0.7):]

    args.high_number = len(high_order_train + high_order_test)
    args.train_high_number = len(high_order_train)
    # print("node numbers: ", args.node_number)
    # print("all higher_order numbers: ", args.high_number)

    #count node's degree
    node_degree = {i : 0 for i in range(args.node_number)}
    for sub_list in (binary_edges + high_order_train):
        for node in sub_list:
            node_degree[node] = node_degree[node] + 1

    #construct node to node
    ii_graph = []
    for sub_list in binary_edges:
        ii_graph.append(sub_list)
        ii_graph.append([sub_list[1], sub_list[0]])

    #self-loop
    # for i in range(args.node_number):
    #     ii_graph.append([i, i])

    #convert high_order to index
    idx2high = {i: high for i, high in enumerate(high_order)}

    #node to high_order
    i2h_graph = []
    i2h_dict = {i: [] for i in range(args.node_number)}
    for i, sub_list in enumerate(high_order):
        for node in sub_list:
            i2h_graph.append([node, i])
            i2h_dict[node].append(i)

    return binary_edges, high_order_train, high_order_test, ii_graph, idx2high, i2h_graph, i2h_dict, node_degree


def load_neg(args, neg_type):
    path, dataset = args.path, args.dataset
    neg_high_train = read_file_and_create_list("{}{}/{}-{}_train.txt".format(path, dataset, dataset, neg_type))
    neg_high_test = read_file_and_create_list("{}{}/{}-{}_test.txt".format(path, dataset, dataset, neg_type))
    return neg_high_train, neg_high_test

def load_neg_random(args, pos_high_train, pos_high_test):
    neg_high_train, neg_high_test = [], []
    node_number = args.node_number
    candidate = [i for i in range(node_number)]
    for high in pos_high_train:

        neg_high_train.append(np.sort(np.random.choice(candidate, size=len(high), replace=False)).tolist())
    for high in pos_high_test:
        # candidate = [i for i in range(node_number)]
        neg_high_test.append(np.sort(np.random.choice(candidate, size=len(high), replace=False)).tolist())


    return neg_high_train, neg_high_test

def construct_incidence(args, high_order):
    node_number = args.node_number

    #node to high_order
    n2h_graph = []
    n2h_dict = {i: [] for i in range(node_number)}
    for i, sub_list in enumerate(high_order):
        for node in sub_list:
            n2h_graph.append([node, i])
            n2h_dict[node].append(i)


    return  n2h_graph, n2h_dict #, high_index

#generate negative samples
def process_sample_attention(args, binary_edges, high_order_train, high_order_test, node_degree, neg_type):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    high_order = high_order_train + high_order_test

    #higher_order's node's cluster sim
    edges = np.array(binary_edges)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(args.node_number, args.node_number), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    features_1 = np.array(adj.todense())
    del edges, adj

    adj_2 = np.dot(features_1, features_1)
    adj_2 = adj_2 - np.diag(np.diag(adj_2))
    adj_2 = (np.where(adj_2 > 0, 1, 0).astype(np.float32)) * 0.8
    features = torch.FloatTensor(features_1+adj_2-features_1*adj_2)
    del features_1, adj_2

    # find best cluster num
    min_cluster, min_evaluation = -1, 99999999
    for cluster_num in range(2, 20):
        kmeans = KMeans(n_clusters=cluster_num, random_state=args.seed).fit(features)
        nodes_labels = kmeans.labels_
        evaluation = silhouette_score(features, nodes_labels)
        if(evaluation < min_evaluation):
            min_evaluation = evaluation
            min_cluster = cluster_num

    # print("the best cluster nun", min_cluster)

    kmeans = KMeans(n_clusters=min_cluster, random_state=args.seed).fit(features)
    nodes_labels = kmeans.labels_
    labels_idx = {i : j for i, j in enumerate(nodes_labels)}
    high_order_labels = copy.deepcopy(high_order)
    for i, sub_list in enumerate(high_order_labels):
        for j, item in enumerate(sub_list):
            high_order_labels[i][j] = labels_idx[item]

    #cluster attention strategy
    cluster_node_dict = [[],[]]
    for node, label in enumerate(nodes_labels):
        cluster_node_dict[0].append(node)
        cluster_node_dict[1].append(label)
    cluster_node_dict = np.array(cluster_node_dict)
    c2n_dict = cluster_node_dict[:, np.argsort(cluster_node_dict[1])]
    c2n_dict = torch.LongTensor(c2n_dict)

    n_features = torch.index_select(features, 0, c2n_dict[0])
    c_features = scatter(n_features, c2n_dict[1], dim=0, reduce='mean')     #Q^c: C * Dim
    attention = torch.matmul(features, c_features.t()) / torch.sqrt(torch.tensor(c_features.shape[1]).float())
    attention = F.softmax(attention, dim=1)
    nodes_cluster_simliar = torch.matmul(attention, attention.t())

    # dis sim
    # nodes_cluster_simliar, sim_div = cluster_simliar(features, kmeans.cluster_centers_)
    nodes_cluster_simliar = np.array(nodes_cluster_simliar)


    high_order_sim = []
    for i, sub_list in enumerate(high_order):
        temp = []
        for pair_edges in combinations(sub_list, 2):
            pair_edges = np.array(pair_edges).astype(np.int32)
            temp.append(nodes_cluster_simliar[pair_edges[0]][pair_edges[1]])
        high_order_sim.append(temp)

    # print("共", len(high_order), "个")
    #generate neg sample
    neg_high_order = []
    for high_order_index, sub_list in enumerate(high_order):
        sub_list_, sub_list = np.copy(np.array(high_order_sim[high_order_index])), np.array(sub_list).astype(np.int32)
        sim_idx_dict = np.array([[sim, idx] for idx, sim in enumerate(sub_list_)])
        idx_sim_dict = {idx: sim for idx, sim in enumerate(sub_list_)}
        sub_list_ = np.sort(sub_list_)

        n = len(sub_list)
        rev_node_pair = []
        for i in range(len(sub_list_)//2, len(sub_list_)):
            index = np.argwhere(sim_idx_dict[:, 0]==sub_list_[len(sub_list_) - i])[0][0]
            row, col = matrix_idx(n, index)
            node_i, node_j= sub_list[row], sub_list[col]
            rev_node_pair.append([node_i, node_j])
        #find node's importance in higher_order
        unique_nodes = {}
        for i in range(len(rev_node_pair)):
            node1 = rev_node_pair[i][0]
            node2 = rev_node_pair[i][1]
            if(node1 not in unique_nodes):
                unique_nodes[node1] = 1
            else:
                unique_nodes[node1] = unique_nodes[node1] + 1
            if(node2 not in unique_nodes):
                unique_nodes[node2] = 1
            else:
                unique_nodes[node2] = unique_nodes[node2] + 1
        unique_nodes = np.array([[key, value] for key, value in unique_nodes.items()])
        unique_nodes_sort = unique_nodes[np.argsort(unique_nodes[:, 1])]

        rate, rate1 = 0.5, 0.7
        if(args.dataset in ["congress-bills", "NDC-substances"]):
            # select low importance nodes
            div = int(n * rate)
            high_impor_nodes = unique_nodes_sort[np.where(unique_nodes_sort[:, 1] >= div)[0], 0]
            low_impor_nodes = unique_nodes_sort[np.where(unique_nodes_sort[:, 1] < div)[0], 0]
            if (len(high_impor_nodes) == 0):
                num = max(int(len(low_impor_nodes) * (rate1)), 1)
                high_impor_nodes = np.random.choice(low_impor_nodes, num,
                                                    replace=False)
                low_impor_nodes = np.setdiff1d(low_impor_nodes, high_impor_nodes) # remove selected nodes in low
            if (len(low_impor_nodes) == 0):
                num = max(int(len(high_impor_nodes) * (1 - rate1)), 1)
                low_impor_nodes = np.random.choice(high_impor_nodes, num,
                                                   replace=False)
                high_impor_nodes = np.setdiff1d(high_impor_nodes, low_impor_nodes)

            # select nodes from top-k low_sim with low_impor_nodes; revision pos sample (sub_list)
            degree_div = np.sort(np.array([node_degree[node] for node in high_impor_nodes]))[0]
            rev_dict = construct_rev_dict(args, nodes_cluster_simliar, high_impor_nodes, low_impor_nodes, node_degree, degree_div)
            for i, node in enumerate(sub_list):
                if(node in rev_dict):
                    sub_list[i] = rev_dict[node]

            neg_high_order.append(sub_list.tolist())
        else:
            #select high importance nodes
            div = int(n * rate)
            high_impor_nodes = unique_nodes_sort[np.where(unique_nodes_sort[:, 1] > div)[0], 0]
            low_impor_nodes = unique_nodes_sort[np.where(unique_nodes_sort[:, 1] <= div)[0], 0]
            if(len(high_impor_nodes) == 0):
                num = max(int(len(low_impor_nodes) * (1 - rate1)), 1)
                high_impor_nodes = np.random.choice(low_impor_nodes, num,
                                                    replace=False)
                low_impor_nodes = np.setdiff1d(low_impor_nodes, high_impor_nodes) # remove selected nodes in low
            if(len(low_impor_nodes) == 0):
                num = max(int(len(low_impor_nodes) * (rate1)), 1)
                low_impor_nodes = np.random.choice(high_impor_nodes, num,
                                                   replace=False)
                high_impor_nodes = np.setdiff1d(high_impor_nodes, low_impor_nodes)

            # select nodes from top-k low_sim with low_impor_nodes; revision pos sample (sub_list)
            degree_div = np.sort(np.array([node_degree[node] for node in low_impor_nodes]))[0]
            rev_dict = construct_rev_dict(args, nodes_cluster_simliar, low_impor_nodes, high_impor_nodes, node_degree, degree_div)
            for i, node in enumerate(sub_list):
                if(node in rev_dict):
                    sub_list[i] = rev_dict[node]

            neg_high_order.append(sub_list.tolist())

    neg_high_train = neg_high_order[:len(high_order_train)]
    neg_high_test = neg_high_order[len(high_order_train):]

    #write HNS, MNS, LNS
    path = args.path
    dataset = args.dataset
    file_name = "{}{}/{}-{}_train.txt".format(path, dataset, dataset, neg_type)
    with open(file_name, "w") as file:
        for high in neg_high_train:
            for item in high:
                file.write(str(item) + " ")
            file.write("\n")
    file_name = "{}{}/{}-{}_test.txt".format(path, dataset, dataset, neg_type)
    with open(file_name, "w") as file:
        for high in neg_high_test:
            for item in high:
                file.write(str(item) + " ")
            file.write("\n")

    return neg_high_train, neg_high_test

# random generate neg sample
def random_process_sample(args, binary_edges, high_order_train, high_order_test):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    high_order = high_order_train + high_order_test

    neg_high_order = []
    index_list = np.array([i for i in range(args.node_number)])
    for high in high_order:
        n = len(high)
        random_selection = np.sort(np.random.choice(index_list, size=n, replace=False))
        neg_high_order.append(random_selection.tolist())

    neg_high_train = neg_high_order[:len(high_order_train)]
    neg_high_test = neg_high_order[len(high_order_train):]

    return neg_high_train, neg_high_test


#revised version---[:candidate_nodes_num]--->[args.node_number - candidate_nodes_num:]; add node filter;
def construct_rev_dict(args, nodes_cluster_simliar, resev_nodes, replace_nodes, node_degree, degree_div):
    candidate_nodes_num = int(args.node_number * 0.02) + 1
    candidate_nodes = np.array([])
    for node in resev_nodes:
        candidate_nodes = np.concatenate(
            (candidate_nodes, np.argsort(nodes_cluster_simliar[node])[int(args.node_number//2) : int(args.node_number//2) + candidate_nodes_num]), axis=0)
    candidate_nodes_unique = np.unique(candidate_nodes)

    mask = np.isin(candidate_nodes_unique, np.concatenate((replace_nodes, resev_nodes), axis=0))
    candidate_nodes_unique = candidate_nodes_unique[~mask]  # remove existed nodes in candidate sets

    # #filter out node's degree
    candidate_nodes_unique_c = candidate_nodes_unique.copy()
    for node in candidate_nodes_unique:
        if(node_degree[node] < degree_div):
            candidate_nodes_unique_c = np.delete(candidate_nodes_unique_c, np.argwhere(candidate_nodes_unique_c == node))

    # candidate node not enough
    if (len(candidate_nodes_unique_c) == 0):
        candidate_nodes_unique_c = candidate_nodes_unique
    if (len(candidate_nodes_unique_c) < len(replace_nodes)):
        add_nodes = np.random.choice(np.array([i for i in range(args.node_number)]),
                                     len(replace_nodes) - len(candidate_nodes_unique_c), replace=False)
        candidate_nodes_unique_c = np.concatenate((candidate_nodes_unique_c, add_nodes), axis=0)

    rev_nodes = np.random.choice(candidate_nodes_unique_c, len(replace_nodes), replace=False)
    rev_dict = {node: rev_nodes[i] for i, node in enumerate(replace_nodes)}
    return rev_dict

def cluster_simliar(nodes_embedding, cluster_center):
    distance = torch.sum(abs(nodes_embedding.unsqueeze(1) - cluster_center), 2)
    row_sum = torch.diag(torch.pow(torch.sum(distance,1),-1))
    norm_dis = torch.matmul(row_sum,distance)   #Ti:node-dis
    #|Ti-Tj|/max|Ti-Tj|
    size = norm_dis.shape[0]
    deviation = []
    max = 0
    for i in range(size):
        copy_norm_dis = norm_dis.clone()
        copy_norm_dis[:,] = norm_dis[i]
        cha = torch.abs(copy_norm_dis-norm_dis).sum(1)
        deviation.append(cha.tolist())
        cha_max = cha.numpy().max()
        if(cha_max>max):
            max = cha_max
    deviation = torch.tensor(deviation)
    clusters_sim = torch.ones((size, size)) - deviation / max   #cluster sim

    sim_sort = np.sort(np.array(clusters_sim.flatten()))
    sim_sort_u = np.unique(sim_sort)
    sim_div = sim_sort_u[(int(len(sim_sort_u)/2))]    #divide similiar degree

    return clusters_sim, sim_div

#order_attention reprocess
def mul_attention(args, nn_graph, n2h_graph, node_number):
    n2h_graph_ = torch.stack((n2h_graph[0], n2h_graph[1] + node_number))
    h2n_graph = torch.stack((n2h_graph_[1], n2h_graph_[0]))
    edge_uninon = torch.cat([nn_graph, n2h_graph_, h2n_graph], dim=-1)
    graph_union = SparseTensor(
        row=edge_uninon[0], col=edge_uninon[1],
        value=None,
        is_sorted=False)
    del edge_uninon
    #Rearrange the order according to the size of the order.
    row, col, _ = graph_union.coo()
    col_order = [2 if i < node_number else args.high_index[i - node_number] for i in col]
    sort_idx = np.lexsort((col_order, row))
    col_sorted = col[sort_idx]
    graph_union = SparseTensor(row=row, col=col_sorted, value=None, sparse_sizes=graph_union.sizes())
    args.graph_union, args.col_order = graph_union, np.array(col_order)[sort_idx]

    edge_index, edge_value, edge_size = graph_union.coo()
    edge = torch.concat((edge_index.unsqueeze(0), edge_value.unsqueeze(0)), dim=0)
    args.edge = edge
    order = torch.tensor(args.col_order, dtype=torch.int64)
    edge_col = edge.clone()
    args.edge_col_y = edge_col
    insert_index = []  # Avoid the case where nodes with degree 0 will be skipped
    for i in range(edge_col[0, :].max() + 1):
        if (len(np.where(edge_col[0, :] == i)[0]) == 0):
            insert_index.append(i + len(insert_index))

    range_indices = np.insert(np.where(np.diff(edge_col[0, :]) != 0)[0] + 1, 0, 0)
    range_indices = np.insert(range_indices, len(range_indices),
                              len(edge_col[1, :]))  # Boundaries for node changes [0,73,..],
    row_i, row_j = torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)
    change_list, row_resort = torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)
    range_len, temp, indice_max, new_indices = [0], 0, 0, []
    for i in range(len(range_indices) - 1):
        start, end = range_indices[i], range_indices[i + 1]
        lenth = len(torch.unique(order[start: end]))
        row_i = torch.concat((row_i, edge_col[0, :][start: start + lenth]), 0)  #  Only the number of nodes corresponding to the order is taken out
        unique_values, new_indice = np.unique(order[start: end], return_inverse=True)  # Order internal renumbering
        row_j = torch.concat((row_j, torch.tensor(unique_values, dtype=torch.int64)), 0)  # Order of nodes
        new_indices.extend((new_indice + indice_max).tolist())  #Node order renumbering
        indice_max += new_indice.max() + 1
        ##
        inner_change_indices = np.where(np.diff(order[range_indices[i]: range_indices[i + 1]]) != 0)[
                                   0] + 1  # Bounds on the change of order of a single node
        change_indices = np.insert(inner_change_indices + range_indices[i], 0, range_indices[i])  #all Bounds
        change_indices = torch.tensor(change_indices, dtype=torch.int64)
        range_len.append(temp + len(change_indices))  # order number
        temp += len(change_indices)
        change_list = torch.concat((change_list, change_indices), 0)
    change_list = torch.concat((change_list, torch.tensor([range_indices[-1]], dtype=torch.int64)), 0)
    args.new_indices = torch.tensor(new_indices, dtype=torch.int64)

    for i in range(len(change_list) - 1):
        number_list = torch.squeeze(torch.tensor([[i] * (change_list[i + 1] - change_list[i])], dtype=torch.int64), 0)
        row_resort = torch.concat((row_resort, number_list), 0)  #Node order renumbering
    expand_counts = [range_len[i + 1] - range_len[i] for i in range(len(range_len) - 1)]
    for item in insert_index:
        expand_counts.insert(item, 1)
    args.expand_counts = expand_counts
    args.edge_col = torch.stack((row_i, row_j), 0)  # node to order
    args.row_i, args.row_resort, args.range_len, args.range_indices = row_i, row_resort, range_len, range_indices
    return args


def matrix_idx(matrix_dim, idx):
    row = 0
    col = 0
    temp = idx
    if(temp == 0):
        col = 1
    else:
        for i in range(matrix_dim - 1, 1, -1):
            if (temp - i) > 0:
                row += 1
                temp = temp - i
            else:
                col = row + temp
                break
    return row, col

def compute_loss(args, output, labels):
    loss1 = F.binary_cross_entropy_with_logits(output, labels)
    return loss1

def InfoNCE_loss(features_0, features_1, features_2, temperature):
    # Calculate cosine similarities
    sim_00 = F.cosine_similarity(features_1, features_2, dim=1) / temperature
    sim_01 = F.cosine_similarity(features_0, features_1, dim=1) / temperature
    sim_02 = F.cosine_similarity(features_0, features_2, dim=1) / temperature
    nume = torch.exp(sim_00)
    denom = torch.exp(sim_01) + torch.exp(sim_02)

    # Calculate InfoNCE loss
    loss = -torch.log(nume / denom)

    return loss.mean()