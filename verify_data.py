def verify_data_formats(real_data_list, generated_data_list):
    # Check if both lists are non-empty
    if not real_data_list:
        print("Warning: Real data list is empty.")
    if not generated_data_list:
        print("Warning: Generated data list is empty.")

    # Check a few real samples
    for idx, d in enumerate(real_data_list[:5]):
        assert isinstance(d, Data), f"Real sample {idx} is not a PyG Data object."
        assert d.x is not None, f"Real sample {idx} missing node features (x)."
        assert d.edge_index is not None, f"Real sample {idx} missing edge_index."
        assert d.x.dim() == 2, f"Real sample {idx} node features must be 2D, got {d.x.dim()}D."
        assert d.edge_index.dim() == 2, f"Real sample {idx} edge_index must be 2D."
        
        # Check if edge_index format is correct
        num_nodes = d.x.size(0)
        max_node_id = d.edge_index.max().item()
        assert max_node_id < num_nodes, f"Real sample {idx} edge_index contains invalid node id {max_node_id}."
        
        # Optionally, check if the graph is undirected (common in many datasets, but not mandatory)
        if not is_undirected(d.edge_index):
            print(f"Note: Real sample {idx} graph is not undirected.")

    # Check a few generated samples
    for idx, d in enumerate(generated_data_list[:5]):
        assert isinstance(d, Data), f"Generated sample {idx} is not a PyG Data object."
        assert d.x is not None, f"Generated sample {idx} missing node features (x)."
        assert d.edge_index is not None, f"Generated sample {idx} missing edge_index."
        assert d.x.dim() == 2, f"Generated sample {idx} node features must be 2D, got {d.x.dim()}D."
        assert d.edge_index.dim() == 2, f"Generated sample {idx} edge_index must be 2D."
        
        # Check node indexing
        num_nodes = d.x.size(0)
        max_node_id = d.edge_index.max().item()
        assert max_node_id < num_nodes, f"Generated sample {idx} edge_index contains invalid node id {max_node_id}."

        # Check if graphs are undirected (optional)
        if not is_undirected(d.edge_index):
            print(f"Note: Generated sample {idx} graph is not undirected.")
    
    # Print a summary
    print("All checked samples appear to have the correct PyG Data format for both real and generated graphs.")

# Run verification
verify_data_formats(original_data_list, generated_data_list)
