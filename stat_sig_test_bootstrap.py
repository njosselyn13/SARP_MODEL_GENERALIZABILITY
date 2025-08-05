import os
import glob
import pandas as pd
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
# from pingouin import ttest

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

    # csv_files = sorted(glob.glob(os.path.join(directory, 'NEWstatsig_*.csv')), reverse=True)

    # csv_files = sorted(
    #     glob.glob(os.path.join(directory, 'NEWstatsig_*.csv')),
    #     key=lambda x: os.path.basename(x).lower()
    # )

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





def load_boot_metrics_from_dir_ordered(directory, reference_csv_path):
    boot_metrics = {}

    # Read the reference order of models
    ref_df = pd.read_csv(reference_csv_path)
    model_order = list(pd.unique(ref_df[['model_a', 'model_b']].values.ravel()))

    # Map to file paths
    all_files = glob.glob(os.path.join(directory, 'NEWstatsig_*.csv'))
    file_map = {
        os.path.basename(f).replace('NEWstatsig_', '').replace('.csv', ''): f
        for f in all_files
    }

    # Order files based on model_order
    ordered_files = [file_map[model] for model in model_order if model in file_map]

    for file_path in ordered_files:
        model_name = os.path.basename(file_path).replace('NEWstatsig_', '').replace('.csv', '')
        df = pd.read_csv(file_path)

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


def bootstrap_p_value_options(boot_model_a, boot_model_b, alternative='two-sided'):
    assert len(boot_model_a) == len(boot_model_b), "Bootstrap samples must be same length"
    diff = boot_model_a - boot_model_b
    mean_diff = np.mean(diff)

    if alternative == 'less':         # Null: Model A >= Model B, alt: Model A < Model B
        p = np.mean(diff < 0)
    elif alternative == 'greater':    # Model A > Model B
        p = np.mean(diff > 0)
    elif alternative == 'two-sided':
        p = 2 * min(np.mean(diff > 0), np.mean(diff < 0))
    else:
        raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")

    return p, mean_diff




def bootstrap_t_p_value_from_samples(boot_model_a, boot_model_b):
    """
    Compute a bootstrap-t test p-value given bootstrap samples of two models.
    Assumes paired bootstrap samples (same index = same resample).
    """
    assert len(boot_model_a) == len(boot_model_b), "Bootstrap samples must be the same length"

    # Difference in bootstrap samples
    diffs = boot_model_a - boot_model_b

    # Observed mean difference
    observed_diff = np.mean(diffs)

    # Standard error of bootstrap differences
    se_diff = np.std(diffs, ddof=1)

    # Compute t statistics
    t_stats = (diffs - observed_diff) / se_diff

    # Two-sided p-value from empirical distribution of t
    p = 2 * min(np.mean(t_stats > 0), np.mean(t_stats < 0))

    return p, observed_diff


def run_all_pairwise_tests(boot_metrics, alt_side = 'two-sided', ttest = False):
    results = []

    for metric, model_dict in boot_metrics.items():
        model_names = list(model_dict.keys())
        for model_a, model_b in combinations(model_names, 2):
            boot_a = model_dict[model_a]
            boot_b = model_dict[model_b]

            if ttest == False:
                # p, diff = bootstrap_p_value(boot_a, boot_b)


                p, diff = bootstrap_p_value_options(boot_a, boot_b, alternative=alt_side)

                # p, diff = bootstrap_t_p_value_from_samples(boot_a, boot_b)

                results.append({
                    "metric": metric,
                    "model_a": model_a,
                    "model_b": model_b,
                    "mean_diff_a_minus_b": diff,
                    "p_value_" + alt_side: p
                })

            elif ttest == True:
                t_stat, p = stats.ttest_rel(a=boot_a, b=boot_b, alternative=alt_side)
                # t_stat, p = pingouin.ttest(a=boot_a, b=boot_b, paired=rtalternative=alt_side)

                results.append({
                    "metric": metric,
                    "model_a": model_a,
                    "model_b": model_b,
                    "test_stat": t_stat,
                    "p_value_" + alt_side: p
                })

    return pd.DataFrame(results)


directory = "./logs_MH_subset/NEW_statsig/"
# directory = "NEW_statsig/"

boot_metrics = load_boot_metrics_from_dir(directory) # if dont care about mathcin gth emodel output pairing from turing
# boot_metrics = load_boot_metrics_from_dir_ordered(directory, 'NEW_stat_sig_output.csv') # match the model order pairing from turing output



# print(boot_metrics)
print()

# plot bootstrap distributions of each metric
plot_all_boot_distributions(boot_metrics)

# Run pairwise tests

# uncomment this to run bootstrap hypothesis test (simple--in paper)
# df_results_twosided = run_all_pairwise_tests(boot_metrics, alt_side = 'two-sided')


# run this for paired t-tests -- in paper
df_results_twosided = run_all_pairwise_tests(boot_metrics, alt_side = 'two-sided', ttest=True)
df_results_less = run_all_pairwise_tests(boot_metrics, alt_side = 'less', ttest=True)
df_results_greater = run_all_pairwise_tests(boot_metrics, alt_side = 'greater', ttest=True)
# print(df_results)
# print()

# change file name to save depending on which test youre running to whatever you want
df_results_twosided.to_csv('ttest_twosided_stat_sig_output.csv', index=False)
df_results_less.to_csv('ttest_less_stat_sig_output.csv', index=False)
df_results_greater.to_csv('ttest_greater_stat_sig_output.csv', index=False)

# df_results_twosided.to_csv('x_twosided_stat_sig_output_linux_sort.csv', index=False)
# df_results_less.to_csv('x_less_stat_sig_output_linux_sort.csv', index=False)
# df_results_greater.to_csv('x_greater_stat_sig_output_linux_sort.csv', index=False)

print()
print('DONE')

