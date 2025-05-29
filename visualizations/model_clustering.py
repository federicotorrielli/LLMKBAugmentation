import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from adjustText import adjust_text
import os
import re

# Load the aggregated scores
csv_path = 'evaluations/automatic_evaluation/aggregated_scores_data.csv'
df = pd.read_csv(csv_path)

# Extract model and KB from 'Parent Folder'
def extract_model_kb(parent_folder):
    parts = parent_folder.split('/')
    model_name = parts[-1]
    kb_name = parts[0]
    if len(parts) > 2: # e.g., framenet/gloss/Model
        kb_name = f"{parts[0]}_{parts[1]}"
    return model_name, kb_name

df[['Model', 'KB']] = df['Parent Folder'].apply(lambda x: pd.Series(extract_model_kb(x)))

# --- Model Size Extraction and Grouping ---
def get_model_size_group(model_name):
    model_name_lower = model_name.lower()
    match_b = re.search(r'(\d+)b', model_name_lower)
    if match_b:
        size = int(match_b.group(1))
        if size < 10: return 'Small (<10B)'
        if size < 40: return 'Medium (10B-40B)'
        return 'Large (>40B)'
    
    if 'jamba' in model_name_lower: 
        return 'Large (>40B)' 
    if 'phi-3-medium' in model_name_lower:
        return 'Medium (10B-40B)'
    if 'phi-3-small' in model_name_lower or 'phi-3-mini' in model_name_lower:
        return 'Small (<10B)'
    if 'c4ai-command-r-plus' in model_name_lower:
        return 'Large (>40B)'
    
    return 'Unknown' # Default for un-parsable sizes

df['ModelSizeGroup'] = df['Model'].apply(get_model_size_group)

# --- Performance Metric & Z-scoring ---
PERFORMANCE_METRIC = 'P@5'

# Z-score performance metric within each model size group
df['ZScoredPerformance'] = df.groupby('ModelSizeGroup')[PERFORMANCE_METRIC].transform(
    lambda x: (x - x.mean()) / (x.std() if x.std() > 0 else 1) # Avoid division by zero for single-entry groups
)
df['ZScoredPerformance'] = df['ZScoredPerformance'].fillna(0)

# Pivot table for PCA: Models as rows, KBs as columns, Z-scored performance as values
pivot_df = df.pivot_table(index='Model', columns='KB', values='ZScoredPerformance')

# Fill NaN KBs for a model using the model's mean z-score across other KBs.
# Then, fill any remaining NaNs (e.g., a model with data for only 1 KB) with 0.
pivot_df = pivot_df.apply(lambda row: row.fillna(row.mean()), axis=1)
pivot_df = pivot_df.fillna(0)

# --- PCA and Clustering ---
# Standardize the pivoted data before PCA (KBs are features)
X = pivot_df.values
X_scaled = StandardScaler().fit_transform(X)

# Apply PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# KMeans Clustering
N_CLUSTERS = 3
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init='auto')
cluster_labels = kmeans.fit_predict(X_pca)

# Create a DataFrame for PCA results, including cluster and size group
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'], index=pivot_df.index)
pca_df['Cluster'] = cluster_labels
pca_df['ModelSizeGroup'] = pca_df.index.map(lambda model_name: df[df['Model'] == model_name]['ModelSizeGroup'].iloc[0])

# --- Plotting ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(18, 14))

# Define markers for size groups for visual distinction
size_groups_ordered = ['Small (<10B)', 'Medium (10B-40B)', 'Large (>40B)', 'Unknown']
markers_list = ['o', 's', '^', 'X']  # Circle, Square, Triangle up, X
marker_map = {group: marker for group, marker in zip(size_groups_ordered, markers_list)}

texts = []
plotted_size_groups = set() # To ensure each size group is added to legend only once

for model_name in pca_df.index:
    cluster = pca_df.loc[model_name, 'Cluster']
    size_group = pca_df.loc[model_name, 'ModelSizeGroup']
    marker = marker_map.get(size_group, 'D') # Default to Diamond if group not in map
    
    # Determine if this size_group is being plotted for the first time for legend purposes
    label_for_legend = None
    if size_group not in plotted_size_groups:
        label_for_legend = size_group
        plotted_size_groups.add(size_group)

    ax.scatter(pca_df.loc[model_name, 'PC1'], pca_df.loc[model_name, 'PC2'], 
               c=[plt.cm.viridis(cluster / (N_CLUSTERS -1 if N_CLUSTERS > 1 else 1))], 
               marker=marker,
               s=250, alpha=0.8, edgecolors='k', linewidth=0.5, label=label_for_legend)

    # Clean model names for labels
    clean_model_name = model_name.replace('-FP8', '').replace('-AWQ', '')
    texts.append(ax.text(pca_df.loc[model_name, 'PC1'], pca_df.loc[model_name, 'PC2'], 
                         clean_model_name, fontsize=12))

adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

ax.set_xlabel(f'Principal Component 1 (Performance relative to size group, {pca.explained_variance_ratio_[0]*100:.2f}% variance)', fontsize=16)
ax.set_ylabel(f'Principal Component 2 (Performance relative to size group, {pca.explained_variance_ratio_[1]*100:.2f}% variance)', fontsize=16)
ax.grid(True, linestyle='--', alpha=0.6)

# --- Legends ---
# Performance Cluster legend
cluster_legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label=f'Perf. Cluster {i+1}',
               markerfacecolor=plt.cm.viridis(i / (N_CLUSTERS -1 if N_CLUSTERS > 1 else 1)), 
               markersize=10)
    for i in range(N_CLUSTERS)
]
legend1 = ax.legend(handles=cluster_legend_elements, title='Performance Clusters', loc='lower left', fontsize=12, title_fontsize=14)
ax.add_artist(legend1)

# Model Size Group legend (collect handles and labels from the scatter plot directly)
handles, labels = ax.get_legend_handles_labels()
# Create a new list of handles and labels, ensuring size_groups_ordered is respected for legend order
# and only existing groups are included.
unique_labels_in_plot = list(dict.fromkeys(labels)) # Get unique labels in order of appearance
ordered_handles = []
ordered_labels = []
for group in size_groups_ordered:
    if group in unique_labels_in_plot:
        idx = unique_labels_in_plot.index(group)
        ordered_handles.append(handles[labels.index(group)]) # Find first handle for this label
        ordered_labels.append(group)

if ordered_handles:
    ax.legend(handles=ordered_handles, labels=ordered_labels, title='Model Size Groups', loc='lower right', fontsize=12, title_fontsize=14)

plt.tight_layout()

# --- Save Plot ---
output_dir = 'visualizations'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'model_clusters_zscored_pca_kmeans.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')

print(f"Z-scored scatter plot saved to {output_path}")
print("Model Size Groups found:", df[['Model', 'ModelSizeGroup']].drop_duplicates().set_index('Model').to_dict()['ModelSizeGroup'])
print(f"Explained variance by PC1: {pca.explained_variance_ratio_[0]*100:.2f}%")
print(f"Explained variance by PC2: {pca.explained_variance_ratio_[1]*100:.2f}%")
print(f"Total variance explained by first two PCs: {sum(pca.explained_variance_ratio_)*100:.2f}%")
