# Create dataset files and save complete implementation
print("Creating complete dataset files and implementation...")

# Save the complete dataset
dataset_summary = {
    'dataset_name': 'Agricultural Subsidy Fraud Detection Dataset',
    'total_applications': len(subsidy_df),
    'total_farmers': len(farmer_df),
    'total_officials': n_officials,
    'total_institutions': n_institutions,
    'fraud_rate': f"{subsidy_df['is_fraudulent'].mean()*100:.1f}%",
    'graph_nodes': G.number_of_nodes(),
    'graph_edges': G.number_of_edges(),
    'graph_density': f"{nx.density(G):.6f}"
}

# Save farmers dataset
farmer_df.to_csv('farmers_data.csv', index=False)
print("Saved farmers_data.csv")

# Save subsidy applications dataset
subsidy_df.to_csv('subsidy_applications.csv', index=False)
print("Saved subsidy_applications.csv")

# Save merged dataset with additional features
merged_df.to_csv('merged_agricultural_data.csv', index=False)
print("Saved merged_agricultural_data.csv")

# Save model performance results
performance_df = pd.DataFrame(comparison_data)
performance_df.to_csv('model_performance_comparison.csv', index=False)
print("Saved model_performance_comparison.csv")

# Create a detailed fraud pattern analysis
print("\nDetailed Fraud Pattern Analysis:")
print("="*50)

# Analyze fraud patterns by different categories
fraud_apps = subsidy_df[subsidy_df['is_fraudulent'] == 1]
legitimate_apps = subsidy_df[subsidy_df['is_fraudulent'] == 0]

print(f"\nFraud Analysis by Subsidy Type:")
fraud_by_type = fraud_apps['subsidy_type'].value_counts()
total_by_type = subsidy_df['subsidy_type'].value_counts()
for subsidy_type in total_by_type.index:
    fraud_count = fraud_by_type.get(subsidy_type, 0)
    total_count = total_by_type[subsidy_type]
    fraud_rate = fraud_count / total_count * 100
    print(f"  {subsidy_type}: {fraud_count}/{total_count} ({fraud_rate:.1f}%)")

print(f"\nFraud Analysis by Processing Time:")
print(f"  Average processing time (Fraud): {fraud_apps['processing_time_days'].mean():.1f} days")
print(f"  Average processing time (Legitimate): {legitimate_apps['processing_time_days'].mean():.1f} days")

print(f"\nFraud Analysis by Amount:")
print(f"  Average amount requested (Fraud): ${fraud_apps['amount_requested'].mean():,.0f}")
print(f"  Average amount requested (Legitimate): ${legitimate_apps['amount_requested'].mean():,.0f}")
print(f"  Average amount approved (Fraud): ${fraud_apps['amount_approved'].mean():,.0f}")
print(f"  Average amount approved (Legitimate): ${legitimate_apps['amount_approved'].mean():,.0f}")

print(f"\nFraud Analysis by Documents & Inspection:")
print(f"  Average documents submitted (Fraud): {fraud_apps['documents_submitted'].mean():.1f}")
print(f"  Average documents submitted (Legitimate): {legitimate_apps['documents_submitted'].mean():.1f}")
print(f"  Inspection rate (Fraud): {fraud_apps['inspection_conducted'].mean()*100:.1f}%")
print(f"  Inspection rate (Legitimate): {legitimate_apps['inspection_conducted'].mean()*100:.1f}%")

# Graph analysis
print(f"\nGraph Structure Analysis:")
print(f"  Total nodes: {G.number_of_nodes():,}")
print(f"  Total edges: {G.number_of_edges():,}")
print(f"  Average degree: {2*G.number_of_edges()/G.number_of_nodes():.2f}")
print(f"  Graph density: {nx.density(G):.6f}")

# Node type distribution
node_types = {}
for node in G.nodes():
    node_type = G.nodes[node]['node_type']
    node_types[node_type] = node_types.get(node_type, 0) + 1

print(f"\nNode Type Distribution:")
for node_type, count in node_types.items():
    print(f"  {node_type}: {count:,} ({count/G.number_of_nodes()*100:.1f}%)")

# Save graph analysis
graph_analysis = {
    'total_nodes': G.number_of_nodes(),
    'total_edges': G.number_of_edges(),
    'average_degree': 2*G.number_of_edges()/G.number_of_nodes(),
    'graph_density': nx.density(G),
    'node_types': node_types
}

# Create summary report
summary_report = f"""
AGRICULTURAL SUBSIDY FRAUD DETECTION - GRAPHML MODEL
====================================================

Dataset Overview:
- Total Farmers: {len(farmer_df):,}
- Total Subsidy Applications: {len(subsidy_df):,}
- Total Processing Officials: {n_officials}
- Total Institutions: {n_institutions}
- Fraud Rate: {subsidy_df['is_fraudulent'].mean()*100:.1f}%

Graph Structure:
- Total Nodes: {G.number_of_nodes():,}
- Total Edges: {G.number_of_edges():,}
- Graph Density: {nx.density(G):.6f}
- Average Degree: {2*G.number_of_edges()/G.number_of_nodes():.2f}

Best Performing Models:
- Best F1-Score: {best_f1_model[0]} ({best_f1_model[1]['f1']:.4f})
- Best AUC: {best_auc_model[0]} ({best_auc_model[1]['auc']:.4f})

GraphML Models Performance:
- GraphSAGE: AUC={results['GraphSAGE']['test_results']['auc']:.4f}, F1={results['GraphSAGE']['test_results']['f1']:.4f}
- GCN: AUC={results['GCN']['test_results']['auc']:.4f}, F1={results['GCN']['test_results']['f1']:.4f}
- GAT: AUC={results['GAT']['test_results']['auc']:.4f}, F1={results['GAT']['test_results']['f1']:.4f}

Traditional ML Models Performance:
- Gradient Boosting: AUC={ml_results['Gradient Boosting']['auc']:.4f}, F1={ml_results['Gradient Boosting']['f1']:.4f}
- Logistic Regression: AUC={ml_results['Logistic Regression']['auc']:.4f}, F1={ml_results['Logistic Regression']['f1']:.4f}
- Random Forest: AUC={ml_results['Random Forest']['auc']:.4f}, F1={ml_results['Random Forest']['f1']:.4f}
- SVM: AUC={ml_results['SVM']['auc']:.4f}, F1={ml_results['SVM']['f1']:.4f}

Key Findings:
1. Traditional ML models generally outperform GNNs on this specific dataset
2. Gradient Boosting achieved the highest F1-score (0.7568)
3. Logistic Regression achieved the highest AUC (0.9750)
4. GraphSAGE was the best performing GNN model
5. The graph structure provides additional context but feature engineering is critical

Files Generated:
- farmers_data.csv: Individual farmer information
- subsidy_applications.csv: Subsidy application details
- merged_agricultural_data.csv: Combined dataset with derived features
- model_performance_comparison.csv: Model performance metrics
"""

with open('project_summary.txt', 'w') as f:
    f.write(summary_report)

print("\nFiles saved successfully!")
print("Generated files:")
print("- farmers_data.csv")
print("- subsidy_applications.csv") 
print("- merged_agricultural_data.csv")
print("- model_performance_comparison.csv")
print("- project_summary.txt")

print(f"\nProject Summary:")
print(summary_report)

import pickle
trained_model = {'model_name': 'Gnn', 'accuracy': auc}

filename = 'AgricSubsidy.pkl'

with open(filename, 'wb') as file:
    pickle.dump(trained_model, file)

print(f"Object successfully saved to {filename}")