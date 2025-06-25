import numpy as np
# import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

auc = {}
ppv = {}
spec = {}
sens = {}

# ALL, White, Black, Other, Female, Male, Hispanic, Not Hispanic

auc['pc2pc'] = [0.73, 0.62, 0.61, 0.66, 0.66, 0.57, 0.60, 0.64]
auc['mh2mh'] = [0.97, 0.65, 0.52, 0.56, 0.66, 0.63, 0.68, 0.64]
auc['mh2pc'] = [0.77, 0.70, 0.80, 0.90, 0.76, 0.68, 0.80, 0.73]
auc['pc2mh'] = [0.80, 0.69, 0.70, 0.79, 0.66, 0.72, 0.55, 0.71]

ppv['pc2pc'] = [0.01, 0.003, 0.0, 0.01, 0.003, 0.003, 0.003, 0.003]
ppv['mh2mh'] = [0.14, 0.01, 0.003, 0.002, 0.005, 0.01, 0.01, 0.006]
ppv['mh2pc'] = [0.03, 0.03, 0.0, 0.01, 0.04, 0.02, 0.0, 0.03]
ppv['pc2mh'] = [0.01, 0.003, 0.003, 0.001, 0.003, 0.004, 0.002, 0.003]

spec['pc2pc'] = [0.94, 0.84, 0.92, 0.86, 0.82, 0.90, 0.88, 0.84]
spec['mh2mh'] = [0.99, 0.89, 0.89, 0.94, 0.88, 0.92, 0.94, 0.89]
spec['mh2pc'] = [0.98, 0.99, 0.99, 0.99, 0.99, 0.98, 0.99, 0.99]
spec['pc2mh'] = [0.85, 0.74, 0.75, 0.74, 0.74, 0.74, 0.73, 0.74]

sens['pc2pc'] = [0.44, 0.36, 0.0, 0.36, 0.40, 0.26, 0.31, 0.36]
sens['mh2mh'] = [0.91, 0.42, 0.17, 0.20, 0.41, 0.40, 0.50, 0.39]
sens['mh2pc'] = [0.35, 0.24, 0.0, 0.03, 0.17, 0.25, 0.0, 0.21]
sens['pc2mh'] = [0.57, 0.49, 0.41, 0.71, 0.53, 0.45, 0.35, 0.51]



# auc['pc2pc'] = [0.82, 0.77, 0.67, 0.90, 0.84, 0.79]
# auc['nonpc2nonpc'] = [0.83, 0.83, 0.87, 0.48, 0.82, 0.82]
# auc['nonpc2pc'] = [0.76, 0.71, 0.50, 0.95, 0.83, 0.67]
# auc['pc2nonpc'] = [0.67, 0.68, 0.50, 0.82, 0.64, 0.72]
# # auc['pc+nonpc2pc'] = [0.87, 0.84, 0.61, 0.95, 0.92, 0.78]
# # auc['pc+nonpc2nonpc'] = [0.84, 0.81, 0.92, 0.99, 0.81, 0.87]
# # auc['both2pc'] = [0.87, 0.84, 0.61, 0.95, 0.92, 0.78]
# # auc['both2nonpc'] = [0.84, 0.81, 0.92, 0.99, 0.81, 0.87]
#
# ppv['pc2pc'] = [0.02, 0.01, 0.003, 0.04, 0.01, 0.01]
# ppv['nonpc2nonpc'] = [0.02, 0.03, 0.03, 0.002, 0.02, 0.03]
# ppv['nonpc2pc'] = [0.01, 0.01, 0.0, 0.05, 0.01, 0.01]
# ppv['pc2nonpc'] = [0.004, 0.005, 0.004, 0.001, 0.004, 0.004]
# # ppv['pc+nonpc2pc'] = [0.01, 0.01, 0.002, 0.02, 0.01, 0.01]
# # ppv['pc+nonpc2nonpc'] = [0.07, 0.08, 0.05, 0.02, 0.07, 0.09]
# # ppv['both2pc'] = [0.01, 0.01, 0.002, 0.02, 0.01, 0.01]
# # ppv['both2nonpc'] = [0.07, 0.08, 0.05, 0.02, 0.07, 0.09]
#
# spec['pc2pc'] = [0.95, 0.95, 0.87, 0.93, 0.93, 0.96]
# spec['nonpc2nonpc'] = [0.97, 0.96, 0.97, 0.97, 0.96, 0.98]
# spec['nonpc2pc'] = [0.90, 0.90, 0.94, 0.94, 0.88, 0.93]
# spec['pc2nonpc'] = [0.81, 0.82, 0.81, 0.72, 0.77, 0.87]
# # spec['pc+nonpc2pc'] = [0.86, 0.86, 0.78, 0.84, 0.84, 0.89]
# # spec['pc+nonpc2nonpc'] = [0.99, 0.99, 0.97, 0.99, 0.99, 0.99]
# # spec['both2pc'] = [0.86, 0.86, 0.78, 0.84, 0.84, 0.89]
# # spec['both2nonpc'] = [0.99, 0.99, 0.97, 0.99, 0.99, 0.99]
#
# sens['pc2pc'] = [0.64, 0.51, 0.40, 0.87, 0.73, 0.41]
# sens['nonpc2nonpc'] = [0.55, 0.56, 0.65, 0.23, 0.54, 0.61]
# sens['nonpc2pc'] = [0.59, 0.50, 0.0, 0.83, 0.71, 0.38]
# sens['pc2nonpc'] = [0.50, 0.49, 0.47, 0.65, 0.52, 0.47]
# # sens['pc+nonpc2pc'] = [0.81, 0.77, 0.40, 0.90, 0.90, 0.66]
# # sens['pc+nonpc2nonpc'] = [0.61, 0.58, 0.88, 0.40, 0.57, 0.70]
# # sens['both2pc'] = [0.81, 0.77, 0.40, 0.90, 0.90, 0.66]
# # sens['both2nonpc'] = [0.61, 0.58, 0.88, 0.40, 0.57, 0.70]



bar_order_names = ['All', 'White', 'Black', 'Other', 'Female', 'Male', 'Hispanic', 'Not Hispanic']
metrics = [auc, ppv, spec, sens]
metric_names = ['AUC', 'PPV', 'Specificity', 'Sensitivity']
# colors = ['b', 'g', 'r', 'c', 'm', 'y']
# colors = ['black', 'red', 'orange', 'yellow', 'purple', 'blue']
# colors = ['black', '#DC143C', '#CD5C5C', '#FF7F7F', '#4682B4', '#483D8B']
# colors = ['black', '#DC143C', '#B22222', '#CD5C5C', '#4682B4', '#4169E1']
colors = ['black', '#ff6666', '#dc143c', '#8b0000', '#87CEEB', '#000080', '#BA55D3', '#4B0082']


# V4
# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

# Font sizes
title_font_size = 18
label_font_size = 14
legend_font_size = 13.5

# Generate plots
for i, (metric, ax) in enumerate(zip(metrics, axes)):
    keys = list(metric.keys())
    values = np.array([metric[key] for key in keys])

    # Create bar plots for each group
    for j in range(len(bar_order_names)):

        if bar_order_names[j] in ['All']:
            offset = j * 0.10 - 0.09  # Adding extra space after black (All) and yellow (Other)
        elif bar_order_names[j] in ['White', 'Black', 'Other']:
            offset = j * 0.10 - 0.05
        elif bar_order_names[j] in ['Hispanic', 'Not Hispanic']:
            offset = j * 0.10 + 0.05
        else:
            offset = j * 0.10

        ax.bar(np.arange(len(keys)) + offset, values[:, j], width=0.10, label=bar_order_names[j], color=colors[j])

        # ax.bar(np.arange(len(keys)) + j * 0.1, values[:, j], width=0.1, label=bar_order_names[j], color=colors[j])

    ax.set_title(metric_names[i], fontsize=title_font_size)

    if i >= 2:  # Only show x-axis labels for the bottom two subplots
        ax.set_xticks(np.arange(len(keys)) + 0.35) # + 0.3)
        ax.set_xticklabels(keys, rotation=45, fontsize=label_font_size)
    else:
        ax.set_xticks([])

    # Set custom y-axis limits
    if i == 1:
        ax.set_ylim(0, 0.15)  # ppv subplot
    else:
        ax.set_ylim(0, 1)

    if i == 1:  # Add legend only to the PPV plot
        ax.legend(fontsize=legend_font_size)

    ax.grid(axis='y')
    ax.tick_params(axis='y', labelsize=label_font_size)

plt.tight_layout()
plt.show()







# # V3
# # Create subplots
# fig, axes = plt.subplots(2, 2, figsize=(15, 10))
# axes = axes.flatten()
#
# # Generate plots
# for i, (metric, ax) in enumerate(zip(metrics, axes)):
#     keys = list(metric.keys())
#     values = np.array([metric[key] for key in keys])
#
#     # Create bar plots for each group
#     for j in range(len(bar_order_names)):
#         ax.bar(np.arange(len(keys)) + j * 0.15, values[:, j], width=0.15, label=bar_order_names[j], color=colors[j])
#
#     ax.set_title(metric_names[i])
#
#     if i >= 2:  # Only show x-axis labels for the bottom two subplots
#         ax.set_xticks(np.arange(len(keys)) + 0.3)
#         ax.set_xticklabels(keys, rotation=45)
#     else:
#         ax.set_xticks([])
#
#     if i == 1:  # Add legend only to the PPV plot
#         ax.legend()
#
#     ax.grid(axis='y')
#
# plt.tight_layout()
# plt.show()






# # V2
#
# # Create subplots
# fig, axes = plt.subplots(2, 2, figsize=(15, 10))
# axes = axes.flatten()
#
# # Generate plots
# for i, (metric, ax) in enumerate(zip(metrics, axes)):
#     keys = list(metric.keys())
#     values = np.array([metric[key] for key in keys])
#
#     # Create bar plots for each group
#     for j in range(len(bar_order_names)):
#         ax.bar(np.arange(len(keys)) + j * 0.15, values[:, j], width=0.15, label=bar_order_names[j], color=colors[j])
#
#     ax.set_title(metric_names[i])
#
#     if i >= 2:  # Only show x-axis labels for the bottom two subplots
#         ax.set_xticks(np.arange(len(keys)) + 0.3)
#         ax.set_xticklabels(keys, rotation=45)
#     else:
#         ax.set_xticks([])
#
#     ax.legend()
#     ax.grid(axis='y')
#
# plt.tight_layout()
# plt.show()





# # V1
# # Create subplots
# fig, axes = plt.subplots(2, 2, figsize=(20, 10))
# axes = axes.flatten()
#
# # Generate plots
# for i, (metric, ax) in enumerate(zip(metrics, axes)):
#     keys = list(metric.keys())
#     values = np.array([metric[key] for key in keys])
#
#     # Create bar plots for each group
#     for j in range(len(bar_order_names)):
#         ax.bar(np.arange(len(keys)) + j * 0.15, values[:, j], width=0.15, label=bar_order_names[j], color=colors[j])
#
#     ax.set_title(metric_names[i])
#     ax.set_xticks(np.arange(len(keys)) + 0.3)
#     ax.set_xticklabels(keys, rotation=45)
#     ax.legend()
#     ax.grid(axis='y')
#
# plt.tight_layout()
# plt.show()
