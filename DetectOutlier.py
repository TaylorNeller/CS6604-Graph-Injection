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

PAD_VAL = 1e10

def get_max_shape(dataset, attr):
    return [max(dim) for dim in zip(*[getattr(data, attr).shape for data in dataset])]

def get_padd_flattened_tensor(data, max_shape, attr):
    return torch.flatten(pad(getattr(data, attr), sum([[0, max_dim - curr_dim] for curr_dim, max_dim in zip(getattr(data, attr).shape[::-1], max_shape[::-1])], []), 'constant', PAD_VAL))

def filter_outliers(dataset, true_len):
    max_shape_x = get_max_shape(dataset, "x")
    max_shape_y = get_max_shape(dataset, "y")
    max_shape_ei = get_max_shape(dataset, "edge_index")
    max_shape_ea = get_max_shape(dataset, "edge_attr")
    
    padded_tensors = [
        torch.cat(
            (get_padd_flattened_tensor(data, max_shape_x, "x"),
            get_padd_flattened_tensor(data, max_shape_x, "y"),
            get_padd_flattened_tensor(data, max_shape_x, "edge_index"),
            get_padd_flattened_tensor(data, max_shape_x, "edge_attr"))
        )
        
        for data in dataset
    ]
    
    stacked = torch.stack(padded_tensors).numpy()
    mean = np.mean(stacked, axis=0)
    cov = np.cov(stacked, rowvar=False)

    pdf = multivariate_normal.pdf(stacked, mean=mean, cov=cov, allow_singular=True)
    tholds = np.linspace(pdf.min(), pdf.max(), 10000)[1:]
    
    F1 = 0
    thold = 0

    # print(f"{pdf.min(), pdf.max()}")
    max_acc = 0
    max_thold = 0
    max_pred = 0
    max_idx = 0
    max_f1 = 0
    
    idx = 1
    ground_truth = [0] * true_len + [1] * (len(dataset)-true_len)
    print(ground_truth)
    for t in tholds:
        pred = (pdf < t).astype(int)
        
        tp = np.sum((pred == 1) & (pred == ground_truth))
        fp = np.sum((pred == 1) & (pred != ground_truth))
        fn = np.sum((pred == 0) & (pred != ground_truth))

        if tp+fp == 0:
            continue
        if tp+fn == 0:
            continue

        prec = tp / (tp+fp)
        rec = tp / (tp+fn)

        f1_ = 2*prec*rec/(prec+rec)
        
        acc = np.sum(pred==ground_truth)/len(dataset)
        if (acc > max_acc):
        # if (f1_ > max_f1):
            max_acc = acc
            max_thold = t
            max_pred = pred
            max_idx = idx
            max_f1 = f1_
        
        
        
        
        # if (acc > max_acc):
        #     max_acc = acc
        #     max_thold = t
        #     max_pred = pred
        #     max_idx = idx
        idx+=1
        
    print(f"{max_idx}, {max_acc}, {max_thold}, {max_f1}")
    print(max_pred)

    # print(f"best F1:{round(F1,3)}, threashold:{round(thold,3)}")

    # pdf = multivariate_normal.pdf(Train_data, mean=mean, cov=var)
    # pred = (pdf < thold).astype(int)
    # tp = np.sum((pred == 1) & (pred == Val_label))
    # print(f"number of outliers: {tp}")
    
    
    
    return 0
        