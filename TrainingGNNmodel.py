# Training function
def train_model(model, data, train_idx, val_idx, epochs=200, lr=0.01, weight_decay=5e-4):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Handle class imbalance with weighted loss
    train_labels = data.y[train_idx]
    class_counts = torch.bincount(train_labels)
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    best_val_acc = 0
    best_model_state = None
    patience = 20
    patience_counter = 0
    
    train_losses = []
    val_accs = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index)
        loss = criterion(out[train_idx], data.y[train_idx])
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(data.x, data.edge_index)
            val_pred = val_out[val_idx].argmax(dim=1)
            val_acc = (val_pred == data.y[val_idx]).float().mean().item()
        
        train_losses.append(loss.item())
        val_accs.append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}')
        
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model, train_losses, val_accs, best_val_acc

# Evaluation function
def evaluate_model(model, data, test_idx):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        test_pred = out[test_idx].argmax(dim=1)
        test_proba = F.softmax(out[test_idx], dim=1)[:, 1]  # Probability of fraud class
        
        y_true = data.y[test_idx].cpu().numpy()
        y_pred = test_pred.cpu().numpy()
        y_proba = test_proba.cpu().numpy()
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_proba)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_proba': y_proba
        }

# Train all models
print("Training Graph Neural Network models...")
results = {}

for model_name, model in models.items():
    print(f"\n--- Training {model_name} ---")
    trained_model, train_losses, val_accs, best_val_acc = train_model(
        model, data, train_idx, val_idx, epochs=100
    )
    
    # Evaluate on test set
    test_results = evaluate_model(trained_model, data, test_idx)
    
    results[model_name] = {
        'model': trained_model,
        'train_losses': train_losses,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc,
        'test_results': test_results
    }
    
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Test accuracy: {test_results['accuracy']:.4f}")
    print(f"Test F1-score: {test_results['f1']:.4f}")
    print(f"Test AUC: {test_results['auc']:.4f}")

print("\n" + "="*50)
print("FINAL RESULTS SUMMARY")
print("="*50)

for model_name, result in results.items():
    test_res = result['test_results']
    print(f"\n{model_name}:")
    print(f"  Accuracy:  {test_res['accuracy']:.4f}")
    print(f"  Precision: {test_res['precision']:.4f}")
    print(f"  Recall:    {test_res['recall']:.4f}")
    print(f"  F1-Score:  {test_res['f1']:.4f}")
    print(f"  AUC:       {test_res['auc']:.4f}")