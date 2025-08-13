# Traditional ML baseline comparison
print("Training traditional ML baselines for comparison...")

# Prepare traditional ML features (only for applications)
application_features = []
application_labels = []

for idx in application_indices:
    node = node_list[idx]
    app_data = G.nodes[node]
    farmer_id = None
    
    # Find the farmer connected to this application
    for neighbor in G.neighbors(node):
        if G.nodes[neighbor]['node_type'] == 'farmer':
            farmer_id = neighbor
            break
    
    if farmer_id:
        farmer_data_node = G.nodes[farmer_id]
        features = [
            app_data.get('amount_requested', 0),
            app_data.get('amount_approved', 0),
            app_data.get('processing_time', 0),
            app_data.get('documents_submitted', 0),
            app_data.get('inspection_conducted', 0),
            farmer_data_node.get('farm_size', 0),
            farmer_data_node.get('years_farming', 0),
            farmer_data_node.get('income', 0),
            farmer_data_node.get('equipment_value', 0),
            1 if app_data.get('subsidy_type') == 'crop_insurance' else 0,
            1 if app_data.get('subsidy_type') == 'equipment_purchase' else 0,
            1 if app_data.get('subsidy_type') == 'land_improvement' else 0,
            1 if app_data.get('subsidy_type') == 'disaster_relief' else 0,
            1 if farmer_data_node.get('crop_type') == 'wheat' else 0,
            1 if farmer_data_node.get('crop_type') == 'corn' else 0,
            1 if farmer_data_node.get('crop_type') == 'soybean' else 0,
            1 if farmer_data_node.get('crop_type') == 'rice' else 0,
            1 if farmer_data_node.get('crop_type') == 'cotton' else 0,
        ]
        
        application_features.append(features)
        application_labels.append(app_data.get('is_fraudulent', 0))

application_features = np.array(application_features)
application_labels = np.array(application_labels)

# Split the same way as graph models
train_features = application_features[perm[:n_train]]
train_labels_ml = application_labels[perm[:n_train]]
val_features = application_features[perm[n_train:n_train+n_val]]
val_labels_ml = application_labels[perm[n_train:n_train+n_val]]
test_features = application_features[perm[n_train+n_val:]]
test_labels_ml = application_labels[perm[n_train+n_val:]]

# Scale features
scaler_ml = StandardScaler()
train_features_scaled = scaler_ml.fit_transform(train_features)
val_features_scaled = scaler_ml.transform(val_features)
test_features_scaled = scaler_ml.transform(test_features)

# Train traditional ML models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

ml_models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
    'SVM': SVC(random_state=42, class_weight='balanced', probability=True)
}

ml_results = {}

for model_name, model in ml_models.items():
    print(f"Training {model_name}...")
    model.fit(train_features_scaled, train_labels_ml)
    
    # Predictions
    y_pred = model.predict(test_features_scaled)
    y_proba = model.predict_proba(test_features_scaled)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(test_labels_ml, y_pred)
    precision = precision_score(test_labels_ml, y_pred, zero_division=0)
    recall = recall_score(test_labels_ml, y_pred, zero_division=0)
    f1 = f1_score(test_labels_ml, y_pred, zero_division=0)
    auc = roc_auc_score(test_labels_ml, y_proba)
    
    ml_results[model_name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

print("\n" + "="*70)
print("COMPARISON: GRAPH NEURAL NETWORKS vs TRADITIONAL ML")
print("="*70)

# Combine all results
all_results = {}
for model_name, result in results.items():
    all_results[f"{model_name} (GNN)"] = result['test_results']

for model_name, result in ml_results.items():
    all_results[f"{model_name} (ML)"] = result

# Create comparison table
print(f"{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC':<10}")
print("-" * 70)

for model_name, metrics in all_results.items():
    print(f"{model_name:<25} "
          f"{metrics['accuracy']:<10.4f} "
          f"{metrics['precision']:<10.4f} "
          f"{metrics['recall']:<10.4f} "
          f"{metrics['f1']:<10.4f} "
          f"{metrics['auc']:<10.4f}")

# Find best model
best_f1_model = max(all_results.items(), key=lambda x: x[1]['f1'])
best_auc_model = max(all_results.items(), key=lambda x: x[1]['auc'])

print(f"\nBest F1-Score: {best_f1_model[0]} ({best_f1_model[1]['f1']:.4f})")
print(f"Best AUC: {best_auc_model[0]} ({best_auc_model[1]['auc']:.4f})")

# Save results for visualization
comparison_data = {
    'models': list(all_results.keys()),
    'accuracy': [metrics['accuracy'] for metrics in all_results.values()],
    'precision': [metrics['precision'] for metrics in all_results.values()],
    'recall': [metrics['recall'] for metrics in all_results.values()],
    'f1': [metrics['f1'] for metrics in all_results.values()],
    'auc': [metrics['auc'] for metrics in all_results.values()]
}

print(f"\nDataset Summary:")
print(f"Total applications: {len(application_indices)}")
print(f"Training set: {len(train_idx)} ({len(train_idx)/len(application_indices)*100:.1f}%)")
print(f"Validation set: {len(val_idx)} ({len(val_idx)/len(application_indices)*100:.1f}%)")
print(f"Test set: {len(test_idx)} ({len(test_idx)/len(application_indices)*100:.1f}%)")
print(f"Fraud rate in training: {train_labels.float().mean()*100:.1f}%")
print(f"Fraud rate in test: {test_labels_ml.mean()*100:.1f}%")