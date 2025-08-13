import pandas as pd
from script import farmer_data,subsidy_df,n_officials,n_institutions
import networkx as nx
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
# Convert to DataFrame and prepare features
farmer_df = pd.DataFrame(farmer_data)

# Merge farmer and subsidy data
merged_df = subsidy_df.merge(farmer_df, on='farmer_id')

# Create additional derived features for better fraud detection
merged_df['amount_ratio'] = merged_df['amount_approved'] / merged_df['amount_requested']
merged_df['amount_per_acre'] = merged_df['amount_requested'] / merged_df['farm_size_acres']
merged_df['subsidy_intensity'] = merged_df['amount_requested'] / merged_df['income_declared']

# Feature engineering for graph neural network
print("Preparing graph structure...")

# Create nodes and edges for the graph
# Node types: farmers, officials, institutions, applications
# Edges: farmer-application, official-application, institution-application, farmer-farmer (same region)

# Create graph
G = nx.Graph()

# Add farmer nodes
for _, farmer in farmer_df.iterrows():
    G.add_node(farmer['farmer_id'], 
               node_type='farmer',
               farm_size=farmer['farm_size_acres'],
               years_farming=farmer['years_farming'],
               crop_type=farmer['crop_type'],
               region=farmer['region'],
               income=farmer['income_declared'],
               equipment_value=farmer['equipment_value'])

# Add official nodes
for i in range(n_officials):
    official_id = f'OFF{i:03d}'
    G.add_node(official_id, node_type='official')

# Add institution nodes
for i in range(n_institutions):
    inst_id = f'INST{i:03d}'
    G.add_node(inst_id, node_type='institution')

# Add application nodes
for _, app in subsidy_df.iterrows():
    G.add_node(app['application_id'],
               node_type='application',
               subsidy_type=app['subsidy_type'],
               amount_requested=app['amount_requested'],
               amount_approved=app['amount_approved'],
               processing_time=app['processing_time_days'],
               documents_submitted=app['documents_submitted'],
               inspection_conducted=app['inspection_conducted'],
               is_fraudulent=app['is_fraudulent'])

print(f"Created graph with {G.number_of_nodes()} nodes")

# Add edges
edge_count = 0

# Farmer-Application edges
for _, app in subsidy_df.iterrows():
    G.add_edge(app['farmer_id'], app['application_id'], edge_type='applies')
    edge_count += 1

# Official-Application edges
for _, app in subsidy_df.iterrows():
    G.add_edge(app['processing_official'], app['application_id'], edge_type='processes')
    edge_count += 1

# Institution-Application edges
for _, app in subsidy_df.iterrows():
    G.add_edge(app['processing_institution'], app['application_id'], edge_type='handles')
    edge_count += 1

# Farmer-Farmer edges (same region connections)
farmer_by_region = farmer_df.groupby('region')['farmer_id'].apply(list).to_dict()
for region, farmers in farmer_by_region.items():
    if len(farmers) > 1:
        # Connect farmers in same region (limited connections to avoid too dense graph)
        for i, farmer1 in enumerate(farmers[:20]):  # Limit to first 20 farmers per region
            for farmer2 in farmers[i+1:min(i+6, len(farmers))]:  # Connect to next 5 farmers
                G.add_edge(farmer1, farmer2, edge_type='same_region')
                edge_count += 1

print(f"Added {edge_count} edges to the graph")
print(f"Graph density: {nx.density(G):.6f}")

# Create node feature matrix
print("Creating node feature matrices...")

# Get all nodes and create mapping
node_list = list(G.nodes())
node_to_idx = {node: idx for idx, node in enumerate(node_list)}

# Create node feature matrix
node_features = []
node_labels = []

for node in node_list:
    node_data = G.nodes[node]
    node_type = node_data['node_type']
    
    if node_type == 'farmer':
        features = [
            1, 0, 0, 0,  # one-hot for node type
            node_data.get('farm_size', 0),
            node_data.get('years_farming', 0),
            node_data.get('income', 0),
            node_data.get('equipment_value', 0),
            1 if node_data.get('crop_type') == 'wheat' else 0,
            1 if node_data.get('crop_type') == 'corn' else 0,
            1 if node_data.get('crop_type') == 'soybean' else 0,
            1 if node_data.get('crop_type') == 'rice' else 0,
            1 if node_data.get('crop_type') == 'cotton' else 0,
        ]
        label = 0  # Farmers are not directly labeled
        
    elif node_type == 'official':
        features = [0, 1, 0, 0] + [0] * 9  # one-hot + padding
        label = 0
        
    elif node_type == 'institution':
        features = [0, 0, 1, 0] + [0] * 9  # one-hot + padding
        label = 0
        
    elif node_type == 'application':
        features = [
            0, 0, 0, 1,  # one-hot for node type
            node_data.get('amount_requested', 0),
            node_data.get('amount_approved', 0),
            node_data.get('processing_time', 0),
            node_data.get('documents_submitted', 0),
            node_data.get('inspection_conducted', 0),
            1 if node_data.get('subsidy_type') == 'crop_insurance' else 0,
            1 if node_data.get('subsidy_type') == 'equipment_purchase' else 0,
            1 if node_data.get('subsidy_type') == 'land_improvement' else 0,
            1 if node_data.get('subsidy_type') == 'disaster_relief' else 0,
        ]
        label = node_data.get('is_fraudulent', 0)
    
    node_features.append(features)
    node_labels.append(label)

node_features = np.array(node_features, dtype=np.float32)
node_labels = np.array(node_labels)

# Normalize numerical features
scaler = StandardScaler()
node_features[:, 4:8] = scaler.fit_transform(node_features[:, 4:8])  # Normalize numerical columns

print(f"Node feature matrix shape: {node_features.shape}")
print(f"Node labels shape: {node_labels.shape}")
print(f"Fraudulent applications in graph: {np.sum(node_labels)}")

# Create adjacency matrix
adj_matrix = nx.adjacency_matrix(G, nodelist=node_list)
print(f"Adjacency matrix shape: {adj_matrix.shape}")
print(f"Number of edges: {adj_matrix.nnz // 2}")  # Undirected graph