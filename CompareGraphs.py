import torch
from distStats import degree_stats, clustering_stats, myMMD, extract_node_degreesList, calculate_novelty_score, compute_uniqueness2  # Replace with the actual file name
import networkx as nx
import numpy as np
from torch_geometric.utils import to_networkx
from sklearn.model_selection import train_test_split

# Load the datasets from disk using torch.load

original_data_list = torch.load('model/original_data_list.pt')
generated_data_list = torch.load('model/generated_data_list.pt')
combined_dataset = torch.load('model/combined_dataset.pt')

#original_data_list = torch.load('model/PROTEINS_original_data_list.pt')
#generated_data_list = torch.load('model/PROTEINS_generated_data_list.pt')
#combined_dataset = torch.load('model/PROTEINS_combined_dataset.pt')

# Print out a few details for verification
print("Datasets loaded successfully!")
print(f"Original dataset size: {len(original_data_list)}")
print(f"Generated dataset size: {len(generated_data_list)}")
print(f"Combined dataset size: {len(combined_dataset)}")


print(f"----------------------------------")

same1, same2 = train_test_split(original_data_list, test_size=0.4, random_state=12)
Same1_Gs = [nx.Graph(list(zip(data.edge_index[0].numpy(), data.edge_index[1].numpy()))) for data in same1]
Same2_Gs = [nx.Graph(list(zip(data.edge_index[0].numpy(), data.edge_index[1].numpy()))) for data in same2]
nodeDegrees1 = extract_node_degreesList(same1)
nodeDegrees2 = extract_node_degreesList(same2)


degree_mmdSame = degree_stats(Same1_Gs, Same2_Gs, is_parallel=False)
clustering_mmdSame = clustering_stats(Same1_Gs, Same2_Gs, bins=100, is_parallel=False)
print(f"Degree Distribution MMD of Train Data Split: {degree_mmdSame}")
print(f"Clustering Coefficient MMD of Train Data Split: {clustering_mmdSame}")

myMMDsame = myMMD(nodeDegrees1, nodeDegrees2)
print(f"Degree Distribution MMD of Train Data Split (RBF Kernel): {myMMDsame}")

print(f"----------------------------------")

# Prepare your original and generated graphs
original_Gs = [nx.Graph(list(zip(data.edge_index[0].numpy(), data.edge_index[1].numpy()))) for data in original_data_list]
generated_Gs = [nx.Graph(list(zip(data.edge_index[0].numpy(), data.edge_index[1].numpy()))) for data in generated_data_list]

original_data_features = extract_node_degreesList(original_data_list)
generated_data_features = extract_node_degreesList(generated_data_list)




# Run the MMD comparison on degree distribution and clustering coefficient
degree_mmd = degree_stats(original_Gs, generated_Gs, is_parallel=False)
clustering_mmd = clustering_stats(original_Gs, generated_Gs, bins=1000, is_parallel=False)
myMMDresults = myMMD(original_data_features, generated_data_features)

print(f"Degree Distribution MMD Generated and Original: {degree_mmd}")
print(f"Clustering Coefficient MMD Generated and Original: {clustering_mmd}")
print(f"MMD: {myMMDresults}")


# Sanity Check
myMMDresultsIdentical = myMMD(original_data_features, original_data_features)
degree_mmdDupe = degree_stats(original_Gs, original_Gs, is_parallel=False)
clustering_mmdDupe = clustering_stats(original_Gs, original_Gs, bins=100, is_parallel=False)
print(f"Degree Distribution MMD of Identical Graph : {degree_mmdDupe}")
print(f"Clustering Coefficient MMD of Identical Graph : {clustering_mmdDupe}")
print(f"Degree Distribution (RBF) MMD of Identical Graph : {myMMDresultsIdentical}")


print(f"----------------------------------")
# Expect to take ~5 minutes
Uniqueness_score = compute_uniqueness2(original_data_list, kernel='rbf')
Uniqueness_score2 = compute_uniqueness2(generated_data_list, kernel='rbf')

print(f"Uniqueness Score of Training Graphs: {Uniqueness_score}")
print(f"Uniqueness Score of Generated Adversarial Graphs: {Uniqueness_score2}")

print(f"----------------------------------")
