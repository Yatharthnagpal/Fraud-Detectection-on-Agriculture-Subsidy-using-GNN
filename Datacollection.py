import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("Creating synthetic agricultural subsidy fraud detection dataset...")

# Generate synthetic agricultural subsidy data
n_farmers = 2000
n_officials = 50
n_institutions = 20

# Create farmer data
farmer_data = {
    'farmer_id': [f'F{i:04d}' for i in range(n_farmers)],
    'farm_size_acres': np.random.normal(150, 50, n_farmers).clip(5, 500),
    'years_farming': np.random.normal(15, 8, n_farmers).clip(1, 50),
    'crop_type': np.random.choice(['wheat', 'corn', 'soybean', 'rice', 'cotton'], n_farmers, 
                                 p=[0.3, 0.25, 0.2, 0.15, 0.1]),
    'region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], n_farmers),
    'income_declared': np.random.normal(50000, 20000, n_farmers).clip(10000, 200000),
    'previous_subsidies_count': np.random.poisson(3, n_farmers),
    'equipment_value': np.random.normal(80000, 30000, n_farmers).clip(10000, 300000),
    'land_ownership_type': np.random.choice(['owned', 'leased', 'mixed'], n_farmers, p=[0.6, 0.3, 0.1])
}

# Create subsidy applications
subsidy_data = []
for i in range(n_farmers):
    # Each farmer can have multiple subsidy applications
    n_applications = np.random.poisson(2) + 1
    for j in range(n_applications):
        subsidy_data.append({
            'application_id': f'APP{i:04d}_{j:02d}',
            'farmer_id': farmer_data['farmer_id'][i],
            'subsidy_type': np.random.choice(['crop_insurance', 'equipment_purchase', 'land_improvement', 
                                            'disaster_relief', 'conservation', 'organic_certification']),
            'amount_requested': np.random.normal(15000, 8000, 1)[0].clip(1000, 100000),
            'amount_approved': 0,  # Will be calculated
            'processing_official': f'OFF{np.random.randint(0, n_officials):03d}',
            'processing_institution': f'INST{np.random.randint(0, n_institutions):03d}',
            'application_date': pd.date_range('2022-01-01', '2023-12-31', periods=1)[0] + 
                               pd.Timedelta(days=np.random.randint(0, 730)),
            'processing_time_days': np.random.normal(30, 15, 1)[0].clip(1, 180),
            'documents_submitted': np.random.randint(3, 12),
            'inspection_conducted': np.random.choice([0, 1], p=[0.3, 0.7]),
            'is_fraudulent': 0  # Will be determined based on patterns
        })

subsidy_df = pd.DataFrame(subsidy_data)

# Create fraud patterns
print("Injecting realistic fraud patterns...")

# Pattern 1: Ghost farming - fictional farms with unrealistic productivity
ghost_farm_indices = np.random.choice(len(subsidy_df), size=int(0.03 * len(subsidy_df)), replace=False)
for idx in ghost_farm_indices:
    farmer_id = subsidy_df.iloc[idx]['farmer_id']
    farmer_idx = farmer_data['farmer_id'].index(farmer_id)
    
    # Unrealistic patterns for ghost farms
    farmer_data['farm_size_acres'][farmer_idx] *= 3  # Inflated farm size
    subsidy_df.at[idx, 'amount_requested'] *= 2.5  # Excessive subsidy requests
    subsidy_df.at[idx, 'documents_submitted'] = np.random.randint(3, 6)  # Fewer documents
    subsidy_df.at[idx, 'inspection_conducted'] = 0  # Avoid inspections
    subsidy_df.at[idx, 'is_fraudulent'] = 1

# Pattern 2: Subsidy farming - same officials approving multiple high-value claims
corrupt_officials = np.random.choice([f'OFF{i:03d}' for i in range(n_officials)], size=5, replace=False)
for official in corrupt_officials:
    official_applications = subsidy_df[subsidy_df['processing_official'] == official].index
    if len(official_applications) > 0:
        fraud_apps = np.random.choice(official_applications, 
                                     size=min(len(official_applications), np.random.randint(2, 8)), 
                                     replace=False)
        for idx in fraud_apps:
            subsidy_df.at[idx, 'amount_requested'] *= 1.8
            subsidy_df.at[idx, 'processing_time_days'] = np.random.uniform(5, 15)  # Fast processing
            subsidy_df.at[idx, 'is_fraudulent'] = 1

# Pattern 3: Identity fraud - multiple applications from same location/equipment
for _ in range(20):
    # Create clusters of suspicious applications
    base_farmer_idx = np.random.randint(0, n_farmers)
    cluster_size = np.random.randint(3, 8)
    
    for i in range(cluster_size):
        if len(subsidy_df) > 0:
            # Find applications from farmers in same region
            same_region_apps = subsidy_df[
                subsidy_df['farmer_id'].map(lambda x: farmer_data['region'][farmer_data['farmer_id'].index(x)] 
                                          == farmer_data['region'][base_farmer_idx])
            ].index
            
            if len(same_region_apps) > 0:
                fraud_idx = np.random.choice(same_region_apps)
                farmer_id = subsidy_df.iloc[fraud_idx]['farmer_id']
                farmer_idx = farmer_data['farmer_id'].index(farmer_id)
                
                # Similar equipment values (suggesting same person)
                farmer_data['equipment_value'][farmer_idx] = farmer_data['equipment_value'][base_farmer_idx] + np.random.normal(0, 1000)
                subsidy_df.at[fraud_idx, 'is_fraudulent'] = 1

# Calculate approved amounts based on patterns
for idx, row in subsidy_df.iterrows():
    if row['is_fraudulent'] == 1:
        # Fraudulent claims often get approved for high amounts
        approval_rate = np.random.uniform(0.8, 1.0)
    else:
        # Legitimate claims have more variable approval rates
        approval_rate = np.random.uniform(0.3, 0.9)
    
    subsidy_df.at[idx, 'amount_approved'] = row['amount_requested'] * approval_rate

print(f"Dataset created with {len(subsidy_df)} subsidy applications")
print(f"Fraudulent applications: {subsidy_df['is_fraudulent'].sum()} ({subsidy_df['is_fraudulent'].mean()*100:.1f}%)")
print(f"Total farmers: {n_farmers}")
print(f"Total processing officials: {n_officials}")
print(f"Total institutions: {n_institutions}")