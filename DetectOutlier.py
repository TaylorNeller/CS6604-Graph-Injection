import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import pad
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch, to_dense_adj
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import dense_to_sparse
from torch_geometric.utils import is_undirected
from collections import Counter

from scipy.stats import multivariate_normal

PAD_VAL = 0

def get_max_shape(dataset, attr):
    return [max(dim) for dim in zip(*[getattr(data, attr).shape for data in dataset])]

def get_padd_flattened_tensor(data, max_shape, attr):
    return torch.flatten(pad(getattr(data, attr), sum([[0, max_dim - curr_dim] for curr_dim, max_dim in zip(getattr(data, attr).shape[::-1], max_shape[::-1])], []), 'constant', PAD_VAL))

# def filter_outliers(dataset, true_len):
#     max_shape_x = get_max_shape(dataset, "x")
#     max_shape_y = get_max_shape(dataset, "y")
#     max_shape_ei = get_max_shape(dataset, "edge_index")
#     max_shape_ea = get_max_shape(dataset, "edge_attr")
    
#     padded_tensors = [
#         # torch.cat(
#         #     (get_padd_flattened_tensor(data, max_shape_x, "x"),
#         #     get_padd_flattened_tensor(data, max_shape_x, "y"),
#         #     get_padd_flattened_tensor(data, max_shape_x, "edge_index"),
#         #     get_padd_flattened_tensor(data, max_shape_x, "edge_attr"))
#         # )
#         get_padd_flattened_tensor(data, max_shape_x, "x")
#         for data in dataset
#     ]
    
#     stacked = torch.stack(padded_tensors).numpy()
#     mean = np.mean(stacked, axis=0)
#     cov = np.cov(stacked, rowvar=False)

#     pdf = multivariate_normal.pdf(stacked, mean=mean, cov=cov, allow_singular=True)
#     tholds = np.linspace(pdf.min(), pdf.max(), 10000)[1:]
    
#     F1 = 0
#     thold = 0

#     # print(f"{pdf.min(), pdf.max()}")
#     max_acc = 0
#     max_thold = 0
#     max_pred = 0
#     max_idx = 0
#     max_f1 = 0
    
#     idx = 1
#     ground_truth = [0] * true_len + [1] * (len(dataset)-true_len)
#     print(ground_truth)
#     for t in tholds:
#         pred = (pdf < t).astype(int)
        
#         tp = np.sum((pred == 1) & (pred == ground_truth))
#         fp = np.sum((pred == 1) & (pred != ground_truth))
#         fn = np.sum((pred == 0) & (pred != ground_truth))

#         if tp+fp == 0:
#             continue
#         if tp+fn == 0:
#             continue

#         prec = tp / (tp+fp)
#         rec = tp / (tp+fn)

#         f1_ = 2*prec*rec/(prec+rec)
        
#         acc = np.sum(pred==ground_truth)/len(dataset)
#         # if (acc > max_acc):
#         if (f1_ > max_f1):
#             max_acc = acc
#             max_thold = t
#             max_pred = pred
#             max_idx = idx
#             max_f1 = f1_
        
        
        
        
#         # if (acc > max_acc):
#         #     max_acc = acc
#         #     max_thold = t
#         #     max_pred = pred
#         #     max_idx = idx
#         idx+=1
        
#     print(f"{max_idx}, {max_acc}, {max_thold}, {max_f1}")
#     print(max_pred)

#     # print(f"best F1:{round(F1,3)}, threashold:{round(thold,3)}")

#     # pdf = multivariate_normal.pdf(Train_data, mean=mean, cov=var)
#     # pred = (pdf < thold).astype(int)
#     # tp = np.sum((pred == 1) & (pred == Val_label))
#     # print(f"number of outliers: {tp}")
    
    
    
#     return 0

from sklearn.ensemble import IsolationForest
def filter_outliers(dataset, true_len):
    
    
    features = []
    for data in dataset:
        # Use node features (data.x) for the Isolation Forest model
        node_features = data.x.numpy()  # Convert tensor to NumPy array
        # print(node_features.shape)
        graph_features = np.mean(node_features, axis=0)  # Example: Compute graph-level feature (mean of node features)
        features.append(graph_features)
        
    features_array = np.array(features)
    
    # nest = np.linspace(0, 1000, 20)[1:]
    # for n in nest:
    #     n = int(n) 
    
    best_f1 = 0
    best_c = 0
    best_outlier = 0
    
    cont = np.linspace(0.1, 0.5, 100)
    for c in cont:
        # model = IsolationForest(contamination=1-true_len/len(dataset))  # Set contamination rate (fraction of outliers)
        model = IsolationForest(n_estimators=100, contamination=c)
        # model = IsolationForest(n_estimators=int(n), contamination=1-true_len/len(dataset))
        outliers = model.fit_predict(features_array)
        outliers = (1-outliers)/2
        # print(outliers)
        
        ground_truth = [0] * true_len + [1] * (len(dataset)-true_len)
        tp = np.sum((outliers == 1) & (outliers == ground_truth))
        fp = np.sum((outliers == 1) & (outliers != ground_truth))
        fn = np.sum((outliers == 0) & (outliers != ground_truth))
        
        prec = tp / (tp+fp)
        rec = tp / (tp+fn)

        f1_ = 2*prec*rec/(prec+rec)
        
        if (f1_ > best_f1):
            best_f1 = f1_
            best_c = c
            best_outlier = outliers
        
        outlier_indices = np.where(outliers == 1)[0]
        # print("Outlier graphs indices:", outlier_indices)
        # print(f"c:{round(c,2)} true positive: {tp}, false positive: {fp}, false negative: {fn}, f1 score: {round(f1_,2)}")
        # print(f"n:{n} true positive: {tp}, false positive: {fp}, false negative: {fn}")
    print(f"best c:{round(best_c,2)}, f1 score: {round(best_f1,2)}")
    return [dataset[i] for i in np.where(best_outlier == 0)[0]]