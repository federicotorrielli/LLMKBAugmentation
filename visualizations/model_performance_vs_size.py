import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Canonical model name mapping
MODEL_INFO_MAP = {
    "Jamba-v0.1": (52, "Jamba-v0.1"),
    "c4ai-command-r-plus": (104, "Command R+"),
    "Phi-3-medium-4k-instruct": (14, "Phi-3-medium"),
    "Meta-Llama-3-70B-Instruct-FP8": (70, "Llama-3 70B Instruct"),
    "Meta-Llama-3-70B-AWQ": (70, "Llama-3 70B"),
    "Mistral-7B-Instruct-v0.3": (7, "Mistral 7B Instruct"),
    "L3-8B-Stheno-v3.2": (8, "L3 Stheno 8B"),
    "gemma-2-27b-it": (27, "Gemma-2 27B IT"),
}

def extract_model_info(folder_name):
    for model_key, (size, name) in MODEL_INFO_MAP.items():
        if model_key in folder_name:
            return size, name
    # Fallback: try to extract size and a simple name
    size_match = re.search(r"(\d+)(B|b)", folder_name)
    numeric_size = int(size_match.group(1)) if size_match else None
    # Remove AWQ, FP8, etc. from name
    name = folder_name.split('/')[-1]
    name = re.sub(r'[-_](AWQ|FP8|AWQFP8|AWQFP16|AWQINT8|AWQINT4|AWQINT2|AWQINT1|AWQINT|FP16|FP32|INT8|INT4|INT2|INT1|INT|AWQ8|AWQ4|AWQ2|AWQ1|AWQ|Q8|Q4|Q2|Q1|Q|AWQFP|FP|AWQINT|INT|AWQFP16|FP16|AWQFP8|FP8|AWQFP32|FP32|AWQFP|FP|AWQINT8|INT8|AWQINT4|INT4|AWQINT2|INT2|AWQINT1|INT1|AWQINT|INT|AWQ8|Q8|AWQ4|Q4|AWQ2|Q2|AWQ1|Q1|AWQ|Q)$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'[-_]+$', '', name) # Remove trailing dashes/underscores
    name = name.replace('-', ' ').replace('_', ' ')
    name = re.sub(r'\s+', ' ', name).strip()
    return numeric_size, name

def main():
    csv_file = "./evaluations/automatic_evaluation/aggregated_scores_data.csv"
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: The file {csv_file} was not found.")
        print("Please ensure the path is correct and the file exists.")
        return

    if "Parent Folder" not in df.columns:
        print("Error: 'Parent Folder' column not found in the CSV.")
        return
    if "P@5" not in df.columns:
        print("Error: 'P@5' column not found in the CSV.")
        return

    df[['Parameter Count', 'Model Name']] = df['Parent Folder'].apply(
        lambda x: pd.Series(extract_model_info(x))
    )

    df_filtered = df.dropna(subset=['Parameter Count', 'Model Name', 'P@5'])
    if df_filtered.empty:
        print("No data to plot after filtering. Check model name patterns and data.")
        return

    df_filtered['Parameter Count'] = pd.to_numeric(df_filtered['Parameter Count'])
    df_filtered['P@5'] = pd.to_numeric(df_filtered['P@5'])

    # Group by model name and parameter count
    df_agg = df_filtered.groupby(['Model Name', 'Parameter Count'])['P@5'].mean().reset_index()

    plt.figure(figsize=(18, 12))
    sns.set_theme(style="whitegrid")

    # Use unique parameter counts for x-ticks
    param_counts = sorted(df_agg['Parameter Count'].unique())
    
    scatter_plot = sns.scatterplot(
        data=df_agg,
        x='Parameter Count',
        y='P@5',
        hue='Model Name',
        size='Parameter Count',
        sizes=(200, 1200),
        palette="tab20",
        legend=False
    )

    # Add labels for each point
    for i in range(df_agg.shape[0]):
        plt.text(
            x=df_agg['Parameter Count'].iloc[i] + 0.01 * max(param_counts),
            y=df_agg['P@5'].iloc[i],
            s=df_agg['Model Name'].iloc[i],
            fontdict=dict(color='black', size=18),
            ha='left', va='center',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.2')
        )

    plt.xlabel("Parameter Count (Billions)", fontsize=28)
    plt.ylabel("P@5 Score", fontsize=28)
    plt.xscale('log')
    plt.xticks(param_counts, [str(x) for x in param_counts], fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout(pad=2.0)

    output_image_path = "visualizations/model_performance_vs_size_p5.png"
    plt.savefig(output_image_path, dpi=300, bbox_inches="tight")
    print(f"Graph saved to {output_image_path}")

if __name__ == "__main__":
    main() 