import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os 

# Constants
VMAF_WEIGHT = 0.13
SWITCH_WEIGHT = 0.75
REBUFFER_WEIGHT = 0.2

# File paths
FILE_PATHS = [
    './pq_test/LOL_3D_performance_comparison.csv',
]

VIDEO_TYPE_MAPPING = {
    'BBB_360p24': 'Animation',
    'TOS_360p24': 'Movies',
    'LOL_3D': 'Movies',
    'underwater': 'Documentary',
    'video_game': 'Animation',
    'sport_long_take': 'Sports',
    'sport_highlight': 'Sports'
}

# Helper functions
def filter_percentile(df, lower=5, upper=90):
    return df[(df >= df.quantile(lower / 100)) & (df <= df.quantile(upper / 100))]

def categorize_trace(file_name):
    file_name = file_name.lower() 
    if 'oboe' in file_name:
        return 'Slow'
    elif 'fcc18' in file_name or 'hsr' in file_name:
        return 'Medium'
    elif 'ghent' in file_name or 'lab' in file_name:
        return 'Fast'
    #elif 'lumos' in file_name:
        #return 'Lumos 5G'
    elif 'lumos' in file_name:
        return 'Fast'
    elif 'HSDPA' in file_name:
        return 'HSDPA'
    return 'Unknown'

def calculate_improvement(qaocs_value, baseline_value, metric):
    if metric == 'Stall Ratio(%)':
        return (baseline_value - qaocs_value) / baseline_value * 100
    elif metric == 'Switch Ratio(%)':
         return (baseline_value - qaocs_value) / baseline_value * 100
    else:
        return (qaocs_value - baseline_value) / baseline_value * 100

# Data loading and preprocessing
def load_and_preprocess_data(file_paths):
    dataframes = [pd.read_csv(file) for file in file_paths]
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    combined_df["Stall Ratio(%)"] = filter_percentile(combined_df["Stall Ratio(%)"])
    #combined_df["Switch Ratio(%)"] = filter_percentile(combined_df["Switch Ratio(%)"])
    combined_df = combined_df.dropna()
    
    combined_df['Network Type'] = combined_df['File'].apply(categorize_trace)
    combined_df['VMAF Change'] = combined_df['Average VMAF Smoothness']
    combined_df['QoE'] = (combined_df['Average VMAF Score'] * VMAF_WEIGHT - 
                          (combined_df['Stall Ratio(%)'] * REBUFFER_WEIGHT) - 
                          (combined_df['Switch Ratio(%)'] * SWITCH_WEIGHT))
    
    return combined_df


def plot_scatter_plot(combined_df):
    plt.figure(figsize=(14, 10))
    markers = {"Slow": "o", "Medium": "s", "Fast": "^", "HSDPA": "D", "Lumos 5G": "p"}

    sns.scatterplot(data=combined_df, 
                    x='QoE', 
                    y='Stall Ratio(%)', 
                    hue='Method', 
                    style='Network Type',
                    #markers=markers,
                    #size='Network Type',
                    sizes=(89, 90),
                    palette='deep')

    plt.title('QoE vs Stall Ratio(%) by Method', fontweight='bold', fontsize=18)
    plt.xlabel('QoE', fontsize=20)
    plt.ylabel('Stall Ratio(%)', fontsize=20)
    plt.legend(title='Method and Network Type', bbox_to_anchor=(.65, .8), loc='upper left', fontsize=24, markerscale=1.8)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tight_layout()
    plt.savefig('./pq_test/scatter_plot.pdf', bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    combined_df = load_and_preprocess_data(FILE_PATHS)
    plot_scatter_plot(combined_df)