import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
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

# Aggregating filtered data by KB and Model, excluding non-numerical columns
aggregated_filtered_data = data_filtered.groupby(['KB', 'Model']).mean(numeric_only=True).reset_index()

# Defining metrics to include in the combined plot
selected_metrics = ['P@5', 'P@10', 'R@5', 'R@10', 'ACC@5', 'ACC@10', 'MRR']

# Function to create improved bar plots, save them, and display the plots
def plot_improved_kb_performance_with_title_and_colors(kb, aggregated_data, metrics):
    kb_data = aggregated_data[aggregated_data['KB'] == kb]
    
    # Create a long-form dataframe for easier plotting with seaborn
    long_data = kb_data.melt(id_vars='Model', value_vars=metrics, var_name='Metric', value_name='Score')

    # Using a more distinguishable and pleasant color palette
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Metric', y='Score', hue='Model', data=long_data, palette='bright')
    
    plt.title(f'Performance Comparison of Models on {kb} Across Multiple Metrics', fontsize=14)
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Scores', fontsize=12)
    plt.xticks(rotation=45)
    # plt.ylim(0, 0.35)  # Set y-axis limit to [0, 0.35]
    plt.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()

    # Save the plot as an image file
    file_name = f'{kb}_performance_comparison.png'
    plt.savefig(file_name, bbox_inches='tight')
    
    # Show the plot
    plt.show()

# Create improved plots for each KB with the new palette and more informative title, and save them
for kb in ['ConceptNet', 'FrameNet', 'MultiAlignet', 'Semagram']:
    plot_improved_kb_performance_with_title_and_colors(kb, aggregated_filtered_data, selected_metrics)
