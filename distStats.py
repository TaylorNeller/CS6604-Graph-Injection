import concurrent.futures
from datetime import datetime
from functools import partial
import numpy as np
import networkx as nx
import os
import pickle as pkl
import subprocess as sp
import time
from sklearn.metrics import pairwise_kernels
from torch_geometric.utils import to_networkx
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
import torch
import torch_geometric


import mmd as mmd

PRINT_TIME = False

def degree_worker(G):
    return np.array(nx.degree_histogram(G))

def add_tensor(x,y):
    support_size = max(len(x), len(y))
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))
    return x+y

def degree_stats(graph_ref_list, graph_pred_list, is_parallel=False):
    ''' Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
      graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    '''
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_ref_list):
                sample_ref.append(deg_hist)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_pred_list_remove_empty):
                sample_pred.append(deg_hist)

    else:
        for i in range(len(graph_ref_list)):
            degree_temp = np.array(nx.degree_histogram(graph_ref_list[i]))
            sample_ref.append(degree_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            degree_temp = np.array(nx.degree_histogram(graph_pred_list_remove_empty[i]))
            sample_pred.append(degree_temp)
    print(len(sample_ref),len(sample_pred))
    mmd_dist = mmd.compute_mmd(sample_ref, sample_pred, kernel=mmd.gaussian_emd) #gausian kernal
    #mmd_dist = mmd.compute_mmd(sample_ref, sample_pred, kernel=mmd.rbf_kernel) #RBF kernal

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing degree mmd: ', elapsed)
    return mmd_dist

def clustering_worker(param):
    G, bins = param
    clustering_coeffs_list = list(nx.clustering(G).values())
    hist, _ = np.histogram(
            clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
    return hist

def clustering_stats(graph_ref_list, graph_pred_list, bins=100, is_parallel=True):
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for clustering_hist in executor.map(clustering_worker, 
                    [(G, bins) for G in graph_ref_list]):
                sample_ref.append(clustering_hist)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for clustering_hist in executor.map(clustering_worker, 
                    [(G, bins) for G in graph_pred_list_remove_empty]):
                sample_pred.append(clustering_hist)

    else:
        for i in range(len(graph_ref_list)):
            clustering_coeffs_list = list(nx.clustering(graph_ref_list[i]).values())
            hist, _ = np.histogram(
                    clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_ref.append(hist)

        for i in range(len(graph_pred_list_remove_empty)):
            clustering_coeffs_list = list(nx.clustering(graph_pred_list_remove_empty[i]).values())
            hist, _ = np.histogram(
                    clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_pred.append(hist)
    
    mmd_dist = mmd.compute_mmd(sample_ref, sample_pred, kernel=mmd.gaussian_emd,sigma=1.0/10, distance_scaling=bins)
    #mmd_dist = mmd.compute_mmd(sample_ref, sample_pred, kernel=mmd.rbf_kernel) # RBF Kernal

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing clustering mmd: ', elapsed)
    return mmd_dist




def myMMD(X, Y, kernel='rbf'):
    """
    Compute MMD between two sets of samples X and Y using the specified kernel.
    X and Y are numpy arrays.
    """
    max_len = max([len(hist) for hist in X + Y])  # Find the maximum histogram length
    # Padding function to make all histograms the same length
    def pad_histogram(hist, max_len):
        return np.pad(hist, (0, max_len - len(hist)), 'constant', constant_values=0)
    
    # Pad all histograms to the same length
    X_padded = [pad_histogram(hist, max_len) for hist in X]
    Y_padded = [pad_histogram(hist, max_len) for hist in Y]

    # Now X_padded and Y_padded are arrays of equal-length histograms
    K_XX = pairwise_kernels(X_padded, X_padded, metric=kernel)
    K_YY = pairwise_kernels(Y_padded, Y_padded, metric=kernel)
    K_XY = pairwise_kernels(X_padded, Y_padded, metric=kernel)
    
    mmd = np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)
    return mmd


def extract_node_degreesList(graph_list):
    """
    Extracts node degree features from a list of NetworkX graphs.
    """
    degree_features = []
    for G in graph_list:
        G2 = to_networkx(G, to_undirected=True)  # Ensure undirected graph
        degree_histogram = np.array(nx.degree_histogram(G2))
        degree_features.append(degree_histogram)
    return (degree_features)



def calculate_novelty_score(real_graphs, generated_graphs, kernel='rbf'):
    # Convert the graphs to feature vectors (e.g., degree distributions)
    real_features = extract_features(real_graphs)
    generated_features = extract_features(generated_graphs)

    # Calculate pairwise distances using a kernel (e.g., RBF kernel)
    distance_matrix = pairwise_kernels(real_features, generated_features, metric=kernel)

    # For each generated graph, find the minimum distance to any real graph
    novelty_scores = np.min(distance_matrix, axis=0)

    # Higher novelty score means the graph is more novel
    return np.mean(novelty_scores)


def calculate_uniqueness_score(generated_graphs, kernel='rbf'):
    # Convert the graphs to feature vectors (e.g., degree distributions)
    generated_features = extract_features(generated_graphs)

    # Calculate pairwise distances using a kernel (e.g., RBF kernel)
    distance_matrix = pairwise_distances(generated_features, metric=kernel)

    # Calculate the average pairwise distance to measure uniqueness (higher is better)
    avg_pairwise_distance = np.mean(distance_matrix)
    
    # Return the inverse of the pairwise distance to make it a "uniqueness" score
    return avg_pairwise_distance

def get_degree_distribution(graph):
    """Extracts the degree distribution of the graph."""
    degree_hist = np.array(nx.degree_histogram(graph))
    return degree_hist

def get_clustering_distribution(graph, bins=100):
    """Extracts the clustering coefficient distribution of the graph."""
    clustering_coeffs = list(nx.clustering(graph).values())
    hist, _ = np.histogram(clustering_coeffs, bins=bins, range=(0, 1), density=True)
    return hist

def extract_features(graphs, bins=100):
    """Extracts features from a list of graphs, such as degree and clustering distributions."""
    features = []
    
    for graph in graphs:
    
        degree_hist = np.array(nx.degree_histogram(graph))
        
        # Ensure the degree histograms are of the same length (pad with zeros if needed)
        # Here we pad all histograms to the same length (e.g., max degree + 1)
        max_degree = max([max(degree_hist) for degree_hist in degree_hist])
        degree_hist = np.pad(degree_hist, (0, max_degree - len(degree_hist) + 1), mode='constant', constant_values=0)
        
        features.append(degree_hist)
    features = np.array(features, dtype=object)    
    return (features)


def extract_node_degrees2(graph_list):
    """
    Extracts node degree features from a list of PyTorch Geometric Data graphs.
    """
    degree_features = []
    for G in graph_list:
        # Ensure that G is a Data object (with 'edge_index' at least)
        if isinstance(G, np.ndarray):
            # If the graph is an adjacency matrix (numpy.ndarray), convert it to edge_index
            # Convert the adjacency matrix to edge_index (this assumes the graph is undirected)
            edge_index = np.array(np.nonzero(G), dtype=np.long)
            edge_index = torch.tensor(edge_index) 
            G = torch_geometric.data.Data(edge_index=edge_index)
        # Convert the graph to networkx format (undirected)

        G2 = to_networkx(G, to_undirected=True)  # Ensure undirected graph
        
        # Compute the degree histogram for the graph
        degree_histogram = np.array(nx.degree_histogram(G2))
        degree_features.append(degree_histogram)
    
    return degree_features

def compute_uniqueness2(generated_graph_list, kernel='rbf'):
    """
    Compute the uniqueness of the generated graph set by calculating MMD between all pairs of graphs.
    """
    # Extract degree histograms from the generated graphs
    generated_degrees = extract_node_degrees2(generated_graph_list)
    
    # Compute pairwise MMD between all generated graphs
    mmd_scores = []
    for i in range(len(generated_degrees)):
        for j in range(i+1, len(generated_degrees)):
            mmd_score = myMMD([generated_degrees[i]], [generated_degrees[j]], kernel)
            mmd_scores.append(mmd_score)
    
    # Average MMD score between all pairs of generated graphs
    uniqueness_score = np.mean(mmd_scores)
    return uniqueness_score