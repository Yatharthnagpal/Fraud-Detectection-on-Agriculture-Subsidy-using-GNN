import plotly.graph_objects as go
import json

# Data with abbreviated model names to fit 15 character limit
data = {
    "models": ["GCN (GNN)", "GraphSAGE (GNN)", "GAT (GNN)", "RF (ML)", "GB (ML)", "LR (ML)", "SVM (ML)"],
    "accuracy": [0.8361, 0.8727, 0.8702, 0.9734, 0.9775, 0.9043, 0.9376],
    "precision": [0.1836, 0.2623, 0.2222, 1.0, 0.9333, 0.3533, 0.4516],
    "recall": [0.5758, 0.7273, 0.5455, 0.5152, 0.6364, 0.8939, 0.6364],
    "f1": [0.2784, 0.3855, 0.3158, 0.68, 0.7568, 0.5064, 0.5283],
    "auc": [0.7839, 0.9091, 0.7872, 0.9036, 0.9625, 0.975, 0.9046]
}

# Brand colors in order
colors = ['#1FB8CD', '#DB4545', '#2E8B57', '#5D878F', '#D2BA4C']

fig = go.Figure()

# Add bars for each metric
metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']

for i, (metric, name) in enumerate(zip(metrics, metric_names)):
    fig.add_trace(go.Bar(
        x=data['models'],
        y=data[metric],
        name=name,
        marker_color=colors[i]
    ))

# Update layout
fig.update_layout(
    title='Agric Subsidy Fraud Detection Models',
    xaxis_title='Models',
    yaxis_title='Performance',
    barmode='group',
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)

# Save the chart
fig.write_image('fraud_detection_performance.png')