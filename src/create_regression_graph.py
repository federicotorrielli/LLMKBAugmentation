import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Reload the dataset and re-apply all transformations
file_path = './aggregated_data.csv'
data = pd.read_csv(file_path)

# Define KB mappings: Aggregating RelatedTo and UsedFor into ConceptNet, and gloss/nogloss into FrameNet
def map_kb(folder_name):
    if 'RelatedTo' in folder_name or 'UsedFor' in folder_name:
        return 'ConceptNet'
    elif 'gloss' in folder_name or 'nogloss' in folder_name:
        return 'FrameNet'
    elif 'multialignet' in folder_name:
        return 'MultiAlignet'
    elif 'semagram' in folder_name:
        return 'Semagram'
    return folder_name

# Apply the KB mapping and extract model names
data['KB'] = data['Parent Folder'].apply(map_kb)
data['Model'] = data['Parent Folder'].apply(lambda x: x.split('/')[1])

# Filter out problematic results where R@1 > 1.0 in the Jamba model
data_filtered = data.copy()
data_filtered.loc[(data_filtered['Model'] == 'Jamba-v0.1') & (data_filtered['R@1'] > 1.0), 'R@1'] = None

# Aggregating filtered data by KB and Model, then calculating the mean of the performance metrics
aggregated_filtered_data = data_filtered.groupby(['KB', 'Model']).mean(numeric_only=True).reset_index()

# Define the model sizes provided
model_sizes = {
    "Meta-Llama-3-70B-AWQ": 70,
    "Meta-Llama-3-70B-Instruct-FP8": 70,
    "L3-8B-Stheno-v3.2": 8,
    "Phi-3-medium-4k-instruct": 14,
    "Mistral-7B-Instruct-v0.3": 7,
    "c4ai-command-r-plus": 104,
    "Jamba-v0.1": 52,
    "gemma-2-27b-it": 27
}

# Add model sizes to the aggregated data
aggregated_filtered_data['Model_Size'] = aggregated_filtered_data['Model'].map(model_sizes)

# Perform regression analysis between model size and performance for each KB
kb_results = {}

for kb in aggregated_filtered_data['KB'].unique():
    kb_data = aggregated_filtered_data[aggregated_filtered_data['KB'] == kb]
    
    X_kb_size = kb_data[['Model_Size']]
    y_kb_performance = kb_data[['P@1', 'P@5', 'R@1', 'R@5', 'ACC@2', 'ACC@5', 'MRR']].mean(axis=1)
    
    reg_kb = LinearRegression()
    reg_kb.fit(X_kb_size, y_kb_performance)
    
    performance_predictions_kb = reg_kb.predict(X_kb_size)
    
    kb_results[kb] = {
        'coef': reg_kb.coef_[0],
        'intercept': reg_kb.intercept_,
        'r_squared': reg_kb.score(X_kb_size, y_kb_performance)
    }
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X_kb_size, y_kb_performance, color='blue', label='Actual Performance')
    plt.plot(X_kb_size, performance_predictions_kb, color='red', label='Regression Line')
    plt.title(f'Relationship Between Model Size and Performance on {kb}')
    plt.xlabel('Model Size (Billion Parameters)')
    plt.ylabel('Overall Performance')
    plt.legend()
    plt.tight_layout()
    file_name = f'{kb}_regression_graph.png'
    plt.savefig(file_name, bbox_inches='tight')

# Display the regression results for each KB
print(kb_results)
