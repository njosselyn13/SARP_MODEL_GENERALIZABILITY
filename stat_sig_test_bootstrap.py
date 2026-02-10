import os
import glob
import pandas as pd
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import wilcoxon
# from pingouin import ttest
import math

def plot_all_boot_distributions(boot_metrics, save_path = 'logs_MH_subset/all_models_new/NEW_figs_stats/'):
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
            save_path1 = os.path.join(save_path, fname)
            plt.savefig(save_path1)
            plt.close()


def plot_all_boot_distributions_onefig(boot_metrics, save_path='logs_MH_subset/all_models_new/NEW_figs_stats/all_boot_distributions.png'):
    # Count total number of plots
    total_plots = sum(len(model_dict) for model_dict in boot_metrics.values())

    # Choose grid size automatically
    n_cols = 24  # number of columns you want (you can change this)
    n_rows = math.ceil(total_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*4))
    axes = axes.flatten()  # make it 1D for easy indexing

    idx = 0
    for metric, model_dict in boot_metrics.items():
        for model_name, samples in model_dict.items():
            ax = axes[idx]
            sns.histplot(samples, kde=True, bins=30, color='steelblue', edgecolor='black', ax=ax)
            ax.set_title(f'{model_name} - {metric}', fontsize=10)
            ax.set_xlabel(metric)
            ax.set_ylabel('Frequency')
            idx += 1

    # Hide any unused subplots
    for j in range(idx, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    # plt.show()




def plot_all_boot_distributions_onefig_4metrics(boot_metrics, save_path='logs_MH_subset/all_models_new/NEW_figs_stats/all_boot_distributions_4metrics.png'):
    # metrics you want combined into one figure
    combined_metrics = ['sensitivities', 'specificities', 'roc_aucs', 'precisions']
    combined_items = []

    # collect all (metric, model_name, samples) for combined metrics
    for metric, model_dict in boot_metrics.items():
        if metric in combined_metrics:
            for model_name, samples in model_dict.items():
                combined_items.append((metric, model_name, samples))

    # --- plot the combined metrics together ---
    if combined_items:
        total_plots = len(combined_items)
        n_cols = 20
        n_rows = math.ceil(total_plots / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        axes = axes.flatten()

        for idx, (metric, model_name, samples) in enumerate(combined_items):
            ax = axes[idx]
            sns.histplot(samples, kde=True, bins=30, color='steelblue', edgecolor='black', ax=ax)
            ax.set_title(f'{model_name} - {metric}', fontsize=10)
            ax.set_xlabel(metric)
            ax.set_ylabel('Frequency')

        # hide unused subplots
        for j in range(len(combined_items), len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        os.makedirs('logs_MH_subset/all_models_new/NEW_figs_stats/', exist_ok=True)
        plt.savefig(save_path, dpi=300)
        # plt.show()


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
        p = np.mean(diff < 0) # proportion of replicates that a less than b
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



def koehn_paired_bootstrap_test_from_scores(
    boot_a,
    boot_b,
    alternative="greater",):

    """
    Paired bootstrap significance test (Koehn, 2004),
    assuming paired bootstrap metric scores are precomputed.

    Parameters
    ----------
    boot_a : array-like
        Bootstrap scores for system A
    boot_b : array-like
        Bootstrap scores for system B
    alternative : {'greater', 'less', 'two-sided'}
        - 'greater': A > B
        - 'less':    A < B
        - 'two-sided': A != B

    Returns
    -------
    p_value : float
        Bootstrap p-value
    stats : dict
        Win-rate and summary statistics
    """

    boot_a = np.asarray(boot_a)
    boot_b = np.asarray(boot_b)

    assert len(boot_a) == len(boot_b), "Bootstrap samples must be paired"

    B = len(boot_a)

    wins_a = np.sum(boot_a > boot_b)
    wins_b = np.sum(boot_a < boot_b)
    ties = np.sum(boot_a == boot_b)

    p_a = wins_a / B
    p_b = wins_b / B
    p_tie = ties / B

    if alternative == "greater":      # H1: A > B
        p_value = 1 - p_a
    elif alternative == "less":       # H1: A < B
        p_value = 1 - p_b
    elif alternative == "two-sided":
        p_value = 2 * min(1 - p_a, 1 - p_b)
        p_value = min(p_value, 1.0)
    else:
        raise ValueError("alternative must be 'greater', 'less', or 'two-sided'")

    stats = {
        "win_rate_a": p_a,
        "win_rate_b": p_b,
        "tie_rate": p_tie,
        "mean_a": boot_a.mean(),
        "mean_b": boot_b.mean(),
        "mean_diff": (boot_a - boot_b).mean(),
        "ci95_a": np.percentile(boot_a, [2.5, 97.5]),
        "ci95_b": np.percentile(boot_b, [2.5, 97.5]),
        "p-value": p_value,
    }

    return p_value, stats



def run_all_pairwise_tests(boot_metrics, alt_side = 'two-sided', ttest = False, wilcoxon = False, koen_bootstrap = True):
    results = []

    for metric, model_dict in boot_metrics.items():
        model_names = list(model_dict.keys())
        for model_a, model_b in combinations(model_names, 2):
            boot_a = model_dict[model_a]
            boot_b = model_dict[model_b]

            if koen_bootstrap == True:
                p_value, stats = koehn_paired_bootstrap_test_from_scores(boot_a=boot_a, boot_b=boot_b, alternative=alt_side)

                results.append({
                    "metric": metric,
                    "model_a": model_a,
                    "model_b": model_b,
                    **stats
                })


            elif koen_bootstrap == False:

                if ttest == False:
                    # p, diff = bootstrap_p_value(boot_a, boot_b)


                    if wilcoxon == True:
                        stat, p = wilcoxon(x=boot_a, y=boot_b, alternative=alt_side, zero_method="pratt")

                        diffs = boot_a - boot_b  # shape (1000,)
                        mean_diff = np.mean(diff)

                        ci_lower = np.percentile(diffs, 2.5)
                        ci_upper = np.percentile(diffs, 97.5)


                        results.append({
                            "metric": metric,
                            "model_a": model_a,
                            "model_b": model_b,
                            "mean_diff_a_minus_b": mean_diff,
                            "p_value_" + alt_side: p,
                            "statistic": stat,
                            "Bootstrap CI lower diffs": ci_lower,
                            "Bootstrap CI upper diffs": ci_upper
                        })

                    elif wilcoxon == False:
                        p, diff = bootstrap_p_value_options(boot_a, boot_b, alternative=alt_side)

                        diffs = boot_a - boot_b  # shape (1000,)

                        ci_lower = np.percentile(diffs, 2.5)
                        ci_upper = np.percentile(diffs, 97.5)

                        # p, diff = bootstrap_t_p_value_from_samples(boot_a, boot_b)

                        results.append({
                            "metric": metric,
                            "model_a": model_a,
                            "model_b": model_b,
                            "mean_diff_a_minus_b": diff,
                            "p_value_" + alt_side: p,
                            "Bootstrap CI lower diffs": ci_lower,
                            "Bootstrap CI upper diffs": ci_upper
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


# directory = "./logs_MH_subset/all_models_1000_bootstraps/"
directory = "./logs_MH_subset/mlp_mh2pc_ppv_bootsraps_corrected/"
# directory = "NEW_statsig/"

boot_metrics = load_boot_metrics_from_dir(directory) # if dont care about mathcin gth emodel output pairing from turing
# boot_metrics = load_boot_metrics_from_dir_ordered(directory, 'NEW_stat_sig_output.csv') # match the model order pairing from turing output



# print(boot_metrics)
print()

# plot bootstrap distributions of each metric
# plot_all_boot_distributions(boot_metrics, save_path='logs_MH_subset/all_models_new/NEW_figs_stats/')
# plot_all_boot_distributions_onefig(boot_metrics, save_path='logs_MH_subset/all_models_new/NEW_figs_stats/all_boot_distributions.png')



# plot_all_boot_distributions_onefig_4metrics(boot_metrics, save_path='logs_MH_subset/all_models_new/NEW_figs_stats/all_boot_distributions_4metrics.png')


# Run pairwise tests

# df_results_twosided = run_all_pairwise_tests(boot_metrics, alt_side = 'two-sided', ttest=False, wilcoxon = True)
# df_results_less = run_all_pairwise_tests(boot_metrics, alt_side = 'less', ttest=False, wilcoxon = True)
# df_results_greater = run_all_pairwise_tests(boot_metrics, alt_side = 'greater', ttest=False, wilcoxon = True)
#
# df_results_twosided.to_csv('stat_sig_test_bootstrap/wilcoxon_twosided_stat_sig_output.csv', index=False)
# df_results_less.to_csv('stat_sig_test_bootstrap/wilcoxon_less_stat_sig_output.csv', index=False)
# df_results_greater.to_csv('stat_sig_test_bootstrap/wilcoxon_greater_stat_sig_output.csv', index=False)

# # uncomment this to run bootstrap hypothesis test (simple--in paper)
# # df_results_twosided = run_all_pairwise_tests(boot_metrics, alt_side = 'two-sided')
# df_results_twosided = run_all_pairwise_tests(boot_metrics, alt_side = 'two-sided', ttest=False)
# df_results_less = run_all_pairwise_tests(boot_metrics, alt_side = 'less', ttest=False)
# df_results_greater = run_all_pairwise_tests(boot_metrics, alt_side = 'greater', ttest=False)
#
# df_results_twosided.to_csv('stat_sig_test_bootstrap/new_bootstrap_diff_mean_twosided_stat_sig_output.csv', index=False)
# df_results_less.to_csv('stat_sig_test_bootstrap/new_bootstrap_diff_mean_less_stat_sig_output.csv', index=False)
# df_results_greater.to_csv('stat_sig_test_bootstrap/new_bootstrap_diff_mean_greater_stat_sig_output.csv', index=False)



df_results_twosided = run_all_pairwise_tests(boot_metrics, alt_side = 'two-sided', ttest=False, wilcoxon=False, koen_bootstrap=True)
df_results_less = run_all_pairwise_tests(boot_metrics, alt_side = 'less', ttest=False, wilcoxon=False, koen_bootstrap=True)
df_results_greater = run_all_pairwise_tests(boot_metrics, alt_side = 'greater', ttest=False, wilcoxon=False, koen_bootstrap=True)

df_results_twosided.to_csv('stat_sig_test_bootstrap/koen_bootstrap_twosided_stat_sig_output.csv', index=False)
df_results_less.to_csv('stat_sig_test_bootstrap/koen_bootstrap_less_stat_sig_output.csv', index=False)
df_results_greater.to_csv('stat_sig_test_bootstrap/koen_bootstrap_greater_stat_sig_output.csv', index=False)


# # run this for paired t-tests -- in paper
# df_results_twosided = run_all_pairwise_tests(boot_metrics, alt_side = 'two-sided', ttest=True)
# df_results_less = run_all_pairwise_tests(boot_metrics, alt_side = 'less', ttest=True)
# df_results_greater = run_all_pairwise_tests(boot_metrics, alt_side = 'greater', ttest=True)
# # print(df_results)
# print()

# change file name to save depending on which test youre running to whatever you want
# df_results_twosided.to_csv('stat_sig_test_bootstrap/ttest_twosided_stat_sig_output_mlp_mh2pc_ppv.csv', index=False)
# df_results_less.to_csv('stat_sig_test_bootstrap/ttest_less_stat_sig_output_mlp_mh2pc_ppv.csv', index=False)
# df_results_greater.to_csv('stat_sig_test_bootstrap/ttest_greater_stat_sig_output_mlp_mh2pc_ppv.csv', index=False)

# df_results_twosided.to_csv('x_twosided_stat_sig_output_linux_sort.csv', index=False)
# df_results_less.to_csv('x_less_stat_sig_output_linux_sort.csv', index=False)
# df_results_greater.to_csv('x_greater_stat_sig_output_linux_sort.csv', index=False)

print()
print('DONE')
