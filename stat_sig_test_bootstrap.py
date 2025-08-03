import os
import glob
import pandas as pd
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns

# run this script after running bootstrap_over13.py with the saving of each metric lists enabled/uncommented (see bootstrap_over13.py)

def plot_all_boot_distributions(boot_metrics):
    # os.makedirs(output_dir, exist_ok=True)

    for metric, model_dict in boot_metrics.items():
        for model_name, samples in model_dict.items():
            plt.figure(figsize=(6, 4))
            sns.histplot(samples, kde=True, bins=30, color='steelblue', edgecolor='black')
            plt.title(f'{model_name} - {metric}')
            plt.xlabel(metric)
            plt.ylabel('Frequency')
            plt.tight_layout()

            fname = f"{metric}_{model_name}.png".replace(" ", "_")
            save_path = os.path.join('logs_MH_subset/NEW_figs_stats/', fname)
            plt.savefig(save_path)
            plt.close()


def load_boot_metrics_from_dir(directory):
    boot_metrics = {}

    # Grab all CSVs starting with NEW_statsig_
    csv_files = glob.glob(os.path.join(directory, 'NEWstatsig_*.csv'))
    # csv_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    print(csv_files)

    for file_path in csv_files:
        # Extract model name from file name
        model_name = os.path.basename(file_path).replace('NEWstatsig_', '').replace('.csv', '')
        print(model_name)

        # Read CSV
        df = pd.read_csv(file_path)

        # For each metric column in the CSV
        for metric in df.columns:
            if metric not in boot_metrics:
                boot_metrics[metric] = {}
            boot_metrics[metric][model_name] = df[metric].values

    return boot_metrics





def bootstrap_p_value(boot_model_a, boot_model_b):
    assert len(boot_model_a) == len(boot_model_b), "Bootstrap samples must be same length"
    diff = boot_model_a - boot_model_b
    mean_diff = np.mean(diff)
    p = 2 * min(np.mean(diff > 0), np.mean(diff < 0))
    return p, mean_diff


def run_all_pairwise_tests(boot_metrics):
    results = []

    for metric, model_dict in boot_metrics.items():
        model_names = list(model_dict.keys())
        for model_a, model_b in combinations(model_names, 2):
            boot_a = model_dict[model_a]
            boot_b = model_dict[model_b]
            p, diff = bootstrap_p_value(boot_a, boot_b)

            results.append({
                "metric": metric,
                "model_a": model_a,
                "model_b": model_b,
                "mean_diff_a_minus_b": diff,
                "p_value": p
            })

    return pd.DataFrame(results)


directory = "./logs_MH_subset/NEW_statsig/"
boot_metrics = load_boot_metrics_from_dir(directory)
# print(boot_metrics)
print()

plot_all_boot_distributions(boot_metrics)

# Run pairwise tests
df_results = run_all_pairwise_tests(boot_metrics)
print(df_results)
print()

df_results.to_csv('NEW_stat_sig_output.csv', index=False)


