# In[1]:
import numpy as np
#import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
import warnings
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from pytorch_tabnet.pretraining import TabNetPretrainer
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score
#from deepforest import CascadeForestClassifier
import optuna
import shap
#import lime
#import lime.lime_tabular
from time import time
#from alibi.explainers import Counterfactual
#import dice_ml
#from dice_ml.utils import helpers # helper functions
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
#from anchor import utils
#from anchor import anchor_tabular
#from alibi.explainers import AnchorTabular
import joblib
import time

import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

start_time = time.time()


# data1 = pd.read_csv('MHRN4_with_zip_income_college_binarized_new.csv')
data1 = pd.read_csv('combined_pc_mh_data.csv')

data = data1[data1['age'] >= 13]
min_value = data['age'].min()
print()
print('-----------------------')
print('MIN VALUE:', min_value)
print('-----------------------')
print()

columns_to_drop_idx = ['antidep_rx_pre3m_idx', 'antidep_rx_pre1y_cumulative_idx', 'antidep_rx_pre5y_cumulative_idx', 'benzo_rx_pre3m_idx', 'benzo_rx_pre1y_cumulative_idx', 'benzo_rx_pre5y_cumulative_idx', 'hypno_rx_pre3m_idx', 'hypno_rx_pre1y_cumulative_idx', 'hypno_rx_pre5y_cumulative_idx', 'sga_rx_pre3m_idx', 'sga_rx_pre1y_cumulative_idx', 'sga_rx_pre5y_cumulative_idx', 'mh_ip_pre3m_idx', 'mh_ip_pre1y_cumulative_idx', 'mh_ip_pre5y_cumulative_idx', 'mh_op_pre3m_idx', 'mh_op_pre1y_cumulative_idx', 'mh_op_pre5y_cumulative_idx', 'mh_ed_pre3m_idx', 'mh_ed_pre1y_cumulative_idx', 'mh_ed_pre5y_cumulative_idx', 'any_sui_att_pre3m_idx', 'any_sui_att_pre1y_cumulative_idx', 'any_sui_att_pre5y_cumulative_idx', 'any_sui_att_pre5y_cumulative_idx_a', 'any_sui_att_pre5y_cumulative_idx_f', 'lvi_sui_att_pre3m_idx', 'lvi_sui_att_pre1y_cumulative_idx', 'lvi_sui_att_pre5y_cumulative_idx', 'ovi_sui_att_pre3m_idx', 'ovi_sui_att_pre1y_cumulative_idx', 'ovi_sui_att_pre5y_cumulative_idx', 'any_inj_poi_pre3m_idx', 'any_inj_poi_pre1y_cumulative_idx', 'any_inj_poi_pre5y_cumulative_idx']
data = data.drop(columns=columns_to_drop_idx)


# In[3]:
# print(data)

# In[4]:
# print(data.columns[316:329])


# In[5]:
# print(sum(data["PRIMARY_CARE_VISIT"] == 1))


# In[6]:
# print(sum(data["PRIMARY_CARE_VISIT"] == 0))


# In[7]:
data["event90"] = data["event90"].fillna(value=0)


# In[8]:
# print(sum(data["event90"].isna()))


# In[9]:
y = data["event90"]


# # ## Scaling data
#
# # In[10]:
# columns_to_scale = ["age","days_since_prev","charlson_score","charlson_a","dep_dx_pre5y_cumulative_a","anx_dx_pre5y_cumulative_a","bip_dx_pre5y_cumulative_a","sch_dx_pre5y_cumulative_a","phqnumber90","phqnumber183","phqnumber365","phq8_index_score_calc_f","raceAsian_8","raceIN_8","hispanic_8","age_8","q9_0_a","q9_1_8","q9_2_8","q9_3_8","q9_1_c","q9_2_c","q9_3_c","any_sui_att_pre5y_cumulative_a","any_sui_att_pre5y_cumulative_8","any_sui_att_pre5y_cumulative_c"]
#
#
# # In[11]:
# count = 0
# for i in columns_to_scale:
#     if i in data.columns:
#         count += 1
#     else:
#         print(i)
# print(count)
#
#
# # In[12]:
# data[columns_to_scale] = scale(data[columns_to_scale])


# In[13]:
# print(data)


# In[14]:
data = data.drop(columns=["person_id","event30","death30","death90","visit_mh"])
# print(data)


data = data.drop(columns=["income", "college", "hhld_inc_It40k", "coll_deg_It25p"])

data.rename(columns={'hhld_inc_lt40k_NJ': 'hhld_inc_It40k', 'coll_deg_lt25p_NJ': 'coll_deg_It25p'}, inplace=True)




# In[15]:
# print(data.columns)


# ## Scaling data

# In[10]:
# columns_to_scale = ["age","days_since_prev","charlson_score","charlson_a","dep_dx_pre5y_cumulative_a","anx_dx_pre5y_cumulative_a","bip_dx_pre5y_cumulative_a","sch_dx_pre5y_cumulative_a","phqnumber90","phqnumber183","phqnumber365","phq8_index_score_calc_f","raceAsian_8","raceIN_8","hispanic_8","age_8","q9_0_a","q9_1_8","q9_2_8","q9_3_8","q9_1_c","q9_2_c","q9_3_c","any_sui_att_pre5y_cumulative_a","any_sui_att_pre5y_cumulative_8","any_sui_att_pre5y_cumulative_c"]

columns_to_scale = ['age', 'days_since_prev', 'dep_dx_pre5y_noi_cumulative', 'anx_dx_pre5y_noi_cumulative', 'bip_dx_pre5y_noi_cumulative', 'sch_dx_pre5y_noi_cumulative', 'oth_dx_pre5y_noi_cumulative', 'dem_dx_pre5y_noi_cumulative', 'add_dx_pre5y_noi_cumulative', 'asd_dx_pre5y_noi_cumulative', 'per_dx_pre5y_noi_cumulative', 'alc_dx_pre5y_noi_cumulative', 'pts_dx_pre5y_noi_cumulative', 'eat_dx_pre5y_noi_cumulative', 'tbi_dx_pre5y_noi_cumulative', 'dru_dx_pre5y_noi_cumulative', 'antidep_rx_pre1y_cumulative', 'antidep_rx_pre5y_cumulative', 'benzo_rx_pre1y_cumulative', 'benzo_rx_pre5y_cumulative', 'hypno_rx_pre1y_cumulative', 'hypno_rx_pre5y_cumulative', 'sga_rx_pre1y_cumulative', 'sga_rx_pre5y_cumulative', 'mh_ip_pre1y_cumulative', 'mh_ip_pre5y_cumulative', 'mh_op_pre1y_cumulative', 'mh_op_pre5y_cumulative', 'mh_ed_pre1y_cumulative', 'mh_ed_pre5y_cumulative', 'any_sui_att_pre1y_cumulative', 'any_sui_att_pre5y_cumulative', 'lvi_sui_att_pre1y_cumulative', 'lvi_sui_att_pre5y_cumulative', 'ovi_sui_att_pre1y_cumulative', 'ovi_sui_att_pre5y_cumulative', 'any_inj_poi_pre1y_cumulative', 'any_inj_poi_pre5y_cumulative', 'any_sui_att_pre5y_cumulative_f', 'any_sui_att_pre5y_cumulative_a', 'charlson_score', 'charlson_a', 'phqnumber90', 'phqnumber183', 'phqnumber365', 'dep_dx_pre5y_cumulative', 'dep_dx_pre5y_cumulative_f', 'dep_dx_pre5y_cumulative_a', 'anx_dx_pre5y_cumulative', 'anx_dx_pre5y_cumulative_f', 'anx_dx_pre5y_cumulative_a', 'bip_dx_pre5y_cumulative', 'bip_dx_pre5y_cumulative_f', 'bip_dx_pre5y_cumulative_a', 'sch_dx_pre5y_cumulative', 'sch_dx_pre5y_cumulative_f', 'sch_dx_pre5y_cumulative_a', 'oth_dx_pre5y_cumulative', 'dem_dx_pre5y_cumulative', 'add_dx_pre5y_cumulative', 'asd_dx_pre5y_cumulative', 'per_dx_pre5y_cumulative', 'alc_dx_pre5y_cumulative', 'dru_dx_pre5y_cumulative', 'pts_dx_pre5y_cumulative', 'eat_dx_pre5y_cumulative', 'tbi_dx_pre5y_cumulative', 'phq8_index_score_calc', 'phq8_index_score_calc_f', 'raceAsian_8', 'raceBlack_8', 'raceHP_8', 'raceIN_8', 'raceMUOT_8', 'raceUN_8', 'hispanic_8', 'age_8', 'q9_0_a', 'q9_1_a', 'q9_2_a', 'q9_3_a', 'q9_0_8', 'q9_1_8', 'q9_2_8', 'q9_3_8', 'q9_0_c', 'q9_1_c', 'q9_2_c', 'q9_3_c', 'any_sui_att_pre5y_cumulative_8', 'any_sui_att_pre5y_cumulative_c', 'any_sui_att_pre5y_cumulative_de', 'any_sui_att_pre5y_cumulative_an', 'any_sui_att_pre5y_cumulative_bi', 'any_sui_att_pre5y_cumulative_sc', 'any_sui_att_pre5y_cumulative_al', 'any_sui_att_pre5y_cumulative_dr', 'any_sui_att_pre5y_cumulative_pe']
# these columns to scale are same as all the numeric (num_cols) columns

# In[11]:
count = 0
for i in columns_to_scale:
    if i in data.columns:
        count += 1
    else:
        print(i)
print(count)


# In[12]:
data[columns_to_scale] = scale(data[columns_to_scale])




# ### Dealing with missing values

# In[16]:
missing_columns = data.columns[data.isnull().any()]


# In[17]:
# print(missing_columns)


# In[18]:
plt.figure(figsize=(10,6))
sns.heatmap(data[missing_columns].isna().transpose(),
            cmap="YlGnBu",
            cbar_kws={'label': 'Missing Data'})
plt.show()


# In[19]:
for i in missing_columns:
    if sum(data[i].isnull()) == data.shape[0]: #446893:
        print(i)
        print(sum(data[i].isnull()))
    else:
        print(i)
        print(sum(data[i].isnull()))


# # In[20]:
# for i in missing_columns:
#     if sum(data[i].isnull()) == 412045: #446893:
#         data[i].fillna(value=0,inplace=True)
#         # print(sum(data[i].isnull()))
#     else:
#         mean_value=data[i].mean()
#         data[i].fillna(value=mean_value,inplace=True)
#         # print(sum(data[i].isnull()))



bin_cols = ['event30', 'event90', 'death30', 'death90', 'visit_mh', 'ac1', 'ac2', 'ac3', 'ac4', 'ac5', 'ac1f', 'ac3f', 'ac4f', 'ac5f', 'Enrolled', 'medicaid', 'commercial', 'privatepay', 'statesubsidized', 'selffunded', 'medicare', 'highdedectible', 'other', 'first_visit', 'female', 'dep_dx_pre5y', 'anx_dx_pre5y', 'bip_dx_pre5y', 'sch_dx_pre5y', 'oth_dx_pre5y', 'dem_dx_pre5y', 'add_dx_pre5y', 'asd_dx_pre5y', 'per_dx_pre5y', 'alc_dx_pre5y', 'pts_dx_pre5y', 'eat_dx_pre5y', 'tbi_dx_pre5y', 'dru_dx_pre5y', 'antidep_rx_pre3m', 'benzo_rx_pre3m', 'hypno_rx_pre3m', 'sga_rx_pre3m', 'mh_ip_pre3m', 'mh_op_pre3m', 'mh_ed_pre3m', 'any_sui_att_pre3m', 'lvi_sui_att_pre3m', 'ovi_sui_att_pre3m', 'any_inj_poi_pre3m', 'current_pregnancy', 'del_pre_1_90', 'del_pre_1_180', 'del_pre_1_365', 'charlson_mi', 'charlson_chd', 'charlson_pvd', 'charlson_cvd', 'charlson_dem', 'charlson_cpd', 'charlson_rhd', 'charlson_pud', 'charlson_mlivd', 'charlson_diab', 'charlson_diabc', 'charlson_plegia', 'charlson_ren', 'charlson_malign', 'charlson_slivd', 'charlson_mst', 'charlson_aids', 'raceAsian_asa', 'raceBlack_asa', 'raceHP_asa', 'raceIN_asa', 'raceMUOT_asa', 'raceUN_asa', 'hispanic_asa', 'raceAsian', 'raceBlack', 'raceHP', 'raceIN', 'raceMUOT', 'raceUN', 'raceWH', 'hispanic', 'raceAsian_f', 'raceBlack_f', 'raceHP_f', 'raceIN_f', 'raceMUOT_f', 'raceUN_f', 'hispanic_f', 'census_missing', 'hhld_inc_It40k', 'coll_deg_It25p', 'phqmode90_0', 'phqmode90_1', 'phqmode90_2', 'phqmax90_0', 'phqmax90_1', 'phqmax90_2', 'phqmax90_3', 'phqmode183_0', 'phqmode183_1', 'phqmode183_2', 'phqmax183_0', 'phqmax183_1', 'phqmax183_2', 'phqmax183_3', 'phqmode365_0', 'phqmode365_1', 'phqmode365_2', 'phqmax365_0', 'phqmax365_1', 'phqmax365_2', 'phqmax365_3', 'raceAsian_de', 'raceBlack_de', 'raceHP_de', 'raceIN_de', 'raceMUOT_de', 'raceUN_de', 'hispanic_de', 'raceAsian_an', 'raceBlack_an', 'raceHP_an', 'raceIN_an', 'raceMUOT_an', 'raceUN_an', 'hispanic_an', 'raceAsian_bi', 'raceBlack_bi', 'raceHP_bi', 'raceIN_bi', 'raceMUOT_bi', 'raceUN_bi', 'hispanic_bi', 'raceAsian_sc', 'raceBlack_sc', 'raceHP_sc', 'raceIN_sc', 'raceMUOT_sc', 'raceUN_sc', 'hispanic_sc', 'phq8_missing', 'phq8_missing_f', 'q9_0', 'q9_1', 'q9_2', 'q9_3', 'q9_0_f', 'q9_1_f', 'q9_2_f', 'q9_3_f', 'raceAsian_q90', 'raceBlack_q90', 'raceHP_q90', 'raceIN_q90', 'raceMUOT_q90', 'raceUN_q90', 'hispanic_q90', 'raceAsian_q91', 'raceBlack_q91', 'raceHP_q91', 'raceIN_q91', 'raceMUOT_q91', 'raceUN_q91', 'hispanic_q91', 'raceAsian_q92', 'raceBlack_q92', 'raceHP_q92', 'raceIN_q92', 'raceMUOT_q92', 'raceUN_q92', 'hispanic_q92', 'raceAsian_q93', 'raceBlack_q93', 'raceHP_q93', 'raceIN_q93', 'raceMUOT_q93', 'raceUN_q93', 'hispanic_q93', 'q9_0_de', 'q9_1_de', 'q9_2_de', 'q9_3_de', 'q9_0_an', 'q9_1_an', 'q9_2_an', 'q9_3_an', 'q9_0_bi', 'q9_1_bi', 'q9_2_bi', 'q9_3_bi', 'q9_0_sc', 'q9_1_sc', 'q9_2_sc', 'q9_3_sc', 'q9_0_al', 'q9_1_al', 'q9_2_al', 'q9_3_al', 'q9_0_dr', 'q9_1_dr', 'q9_2_dr', 'q9_3_dr', 'q9_0_pe', 'q9_1_pe', 'q9_2_pe', 'q9_3_pe', 'phqMax90_0_q90', 'phqMax90_1_q90', 'phqMax90_2_q90', 'phqMax90_3_q90', 'phqMax90_0_q91', 'phqMax90_1_q91', 'phqMax90_2_q91', 'phqMax90_3_q91', 'phqMax90_0_q92', 'phqMax90_1_q92', 'phqMax90_2_q92', 'phqMax90_3_q92', 'phqMax90_0_q93', 'phqMax90_1_q93', 'phqMax90_2_q93', 'phqMax90_3_q93']

num_cols = ['age', 'days_since_prev', 'dep_dx_pre5y_noi_cumulative', 'anx_dx_pre5y_noi_cumulative', 'bip_dx_pre5y_noi_cumulative', 'sch_dx_pre5y_noi_cumulative', 'oth_dx_pre5y_noi_cumulative', 'dem_dx_pre5y_noi_cumulative', 'add_dx_pre5y_noi_cumulative', 'asd_dx_pre5y_noi_cumulative', 'per_dx_pre5y_noi_cumulative', 'alc_dx_pre5y_noi_cumulative', 'pts_dx_pre5y_noi_cumulative', 'eat_dx_pre5y_noi_cumulative', 'tbi_dx_pre5y_noi_cumulative', 'dru_dx_pre5y_noi_cumulative', 'antidep_rx_pre1y_cumulative', 'antidep_rx_pre5y_cumulative', 'benzo_rx_pre1y_cumulative', 'benzo_rx_pre5y_cumulative', 'hypno_rx_pre1y_cumulative', 'hypno_rx_pre5y_cumulative', 'sga_rx_pre1y_cumulative', 'sga_rx_pre5y_cumulative', 'mh_ip_pre1y_cumulative', 'mh_ip_pre5y_cumulative', 'mh_op_pre1y_cumulative', 'mh_op_pre5y_cumulative', 'mh_ed_pre1y_cumulative', 'mh_ed_pre5y_cumulative', 'any_sui_att_pre1y_cumulative', 'any_sui_att_pre5y_cumulative', 'lvi_sui_att_pre1y_cumulative', 'lvi_sui_att_pre5y_cumulative', 'ovi_sui_att_pre1y_cumulative', 'ovi_sui_att_pre5y_cumulative', 'any_inj_poi_pre1y_cumulative', 'any_inj_poi_pre5y_cumulative', 'any_sui_att_pre5y_cumulative_f', 'any_sui_att_pre5y_cumulative_a', 'charlson_score', 'charlson_a', 'phqnumber90', 'phqnumber183', 'phqnumber365', 'dep_dx_pre5y_cumulative', 'dep_dx_pre5y_cumulative_f', 'dep_dx_pre5y_cumulative_a', 'anx_dx_pre5y_cumulative', 'anx_dx_pre5y_cumulative_f', 'anx_dx_pre5y_cumulative_a', 'bip_dx_pre5y_cumulative', 'bip_dx_pre5y_cumulative_f', 'bip_dx_pre5y_cumulative_a', 'sch_dx_pre5y_cumulative', 'sch_dx_pre5y_cumulative_f', 'sch_dx_pre5y_cumulative_a', 'oth_dx_pre5y_cumulative', 'dem_dx_pre5y_cumulative', 'add_dx_pre5y_cumulative', 'asd_dx_pre5y_cumulative', 'per_dx_pre5y_cumulative', 'alc_dx_pre5y_cumulative', 'dru_dx_pre5y_cumulative', 'pts_dx_pre5y_cumulative', 'eat_dx_pre5y_cumulative', 'tbi_dx_pre5y_cumulative', 'phq8_index_score_calc', 'phq8_index_score_calc_f', 'raceAsian_8', 'raceBlack_8', 'raceHP_8', 'raceIN_8', 'raceMUOT_8', 'raceUN_8', 'hispanic_8', 'age_8', 'q9_0_a', 'q9_1_a', 'q9_2_a', 'q9_3_a', 'q9_0_8', 'q9_1_8', 'q9_2_8', 'q9_3_8', 'q9_0_c', 'q9_1_c', 'q9_2_c', 'q9_3_c', 'any_sui_att_pre5y_cumulative_8', 'any_sui_att_pre5y_cumulative_c', 'any_sui_att_pre5y_cumulative_de', 'any_sui_att_pre5y_cumulative_an', 'any_sui_att_pre5y_cumulative_bi', 'any_sui_att_pre5y_cumulative_sc', 'any_sui_att_pre5y_cumulative_al', 'any_sui_att_pre5y_cumulative_dr', 'any_sui_att_pre5y_cumulative_pe']

# for i in missing_columns:
#     if sum(data[i].isnull()) == data.shape[0]: #446893:
#         # data[i].fillna(value=0,inplace=True)
#         data = data.drop(columns=[i])
#         # print(sum(data[i].isnull()))
#     else:
#         if i in bin_cols:
#             data[i].fillna(value=-1, inplace=True)
#         elif i in num_cols:
#             mean_value=data[i].mean()
#             data[i].fillna(value=mean_value,inplace=True)
#         else:
#             print('Column not binary or numeric', i)
#         # print(sum(data[i].isnull()))

for i in missing_columns:
    if sum(data[i].isnull()) == data.shape[0]: #412045: #446893:
        # data[i].fillna(value=0,inplace=True)
        data = data.drop(columns=[i]) # if all rows are empty for a column, drop that column
        # print(sum(data[i].isnull()))
        print("Empty column:", i)
    else:
        data[i].fillna(value=-1, inplace=True) # # if only some of rows have missing value for a column, then fill that with -1, for num or bin cols






# ## Seperating Primary Care and Non Primary Care data

# In[21]:
pc_data = data[data["PRIMARY_CARE_VISIT"] == 1]
# print(pc_data)


# In[22]:
non_pc_data = data[data["PRIMARY_CARE_VISIT"] == 0]
# print(non_pc_data)


# In[23]:
y_pc = pc_data["event90"]


# In[24]:
y_non_pc = non_pc_data["event90"]


# In[25]:
# print(pc_data)


# In[26]:
# print(non_pc_data)


# In[27]:
non_pc_data = non_pc_data.drop(columns=["PRIMARY_CARE_VISIT","event90"])
# print(non_pc_data)


# In[28]:
pc_data = pc_data.drop(columns=["PRIMARY_CARE_VISIT","event90"])
# print(pc_data)


# ### Splitting data into train and validation

# In[29]:
train_X_pc, X_test_pc, train_y_pc, y_test_pc = train_test_split(pc_data, y_pc, test_size=0.35, random_state=42, stratify = y_pc)


# In[30]:
X_train_pc, X_val_pc, y_train_pc, y_val_pc = train_test_split(train_X_pc, train_y_pc, test_size=0.35, random_state=42, stratify = train_y_pc)


# In[31]:
train_X_non_pc, X_test_non_pc, train_y_non_pc, y_test_non_pc = train_test_split(non_pc_data, y_non_pc, test_size=0.35, random_state=42, stratify = y_non_pc)


# In[32]:
X_train_non_pc, X_val_non_pc, y_train_non_pc, y_val_non_pc = train_test_split(train_X_non_pc, train_y_non_pc, test_size=0.35, random_state=42, stratify = train_y_non_pc)


# In[33]:
X_train = pd.concat([X_train_pc,X_train_non_pc])
# print(X_train)


# In[34]:
y_train = pd.concat([y_train_pc, y_train_non_pc])
# print(y_train)


# In[35]:
# print(X_test_non_pc)


# In[36]:
# print(y_test_non_pc)


# # TabNet
print()
print('--------------------')
print()
print('TabNet:')
print()

# ### Optimizing TabNet using Optuna for PC
print('------------------------------------------')
print()
print('Now using Optuna!!!')

print('Training TabNet, optimizing with Optuna, weights=1, Train PC+Non PC, Test PC')
print()

def objective_pretrain(trial):
    # parameters to tune
    mask_type = trial.suggest_categorical("mask_type", ["entmax", "sparsemax"])
    # n_d = trial.suggest_int("n_da", 8, 64, step=8)
    n_a = trial.suggest_int("n_a", 8, 64, step=8)
    n_steps = trial.suggest_int("n_steps", 1, 5, step=1)
    gamma = trial.suggest_float("gamma", 1.0, 2.0, step=0.2)
    n_shared = trial.suggest_int("n_shared", 1, 5)
    lambda_sparse = trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True)
    tabnet_params_pretrain = dict(n_d=n_a, n_a=n_a, n_steps=n_steps, gamma=gamma,
                         lambda_sparse=lambda_sparse, optimizer_fn=torch.optim.Adam,
                         optimizer_params=dict(lr=trial.suggest_categorical('learning_rate', [2e-2, 1e-1, 1e-3]), weight_decay=1e-5),
                         mask_type=mask_type, n_shared=n_shared,
                         scheduler_params=dict(mode="min",
                                               patience=trial.suggest_int("patience", low=3, high=10),
                                               # changing sheduler patience to be lower than early stopping patience
                                               min_lr=1e-5,
                                               factor=0.5, ),
                         scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                         verbose=0,
                         )

    unsupervised_model = TabNetPretrainer(**tabnet_params_pretrain) # **tabnet_params_pretrain

    unsupervised_model.fit(
        X_train=X_train.values,
        eval_set=[X_val_pc.values],
        pretraining_ratio=0.5,
    )

    # preds = clf.predict(X_test.values)

    # history = clf.history
    # acc = accuracy_score(y_test, preds) * 100
    # Make reconstruction from a dataset
    # reconstructed_X, embedded_X = unsupervised_model.predict(X_val_pc.values)
    # assert (reconstructed_X.shape == embedded_X.shape)

    # preds_valid = unsupervised_model.predict_proba(X_val.values)
    # valid_auc = roc_auc_score(y_score=preds_valid[:, 1], y_true=y_val)

    return unsupervised_model.best_cost


# In[47]:
print("TabNet Pre-Training Optuna Tuning")
optuna.logging.set_verbosity(optuna.logging.WARNING)
sampler = optuna.samplers.TPESampler(seed=1)
study = optuna.create_study(sampler=sampler, direction="minimize")
study.optimize(objective_pretrain, n_trials=100)

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

best_params_pretrain = study.best_params
patience_op = best_params_pretrain['patience']
del best_params_pretrain['patience']
print('patience_op:', patience_op)

lr_op = best_params_pretrain['learning_rate']
del best_params_pretrain['learning_rate']
print('lr_op', lr_op)


unsupervised_model = TabNetPretrainer(optimizer_params=dict(lr=lr_op,
                                             weight_decay=1e-5),
                                      scheduler_params=dict(mode="min",
                                               patience=patience_op,
                                               # changing sheduler patience to be lower than early stopping patience
                                               min_lr=1e-5,
                                               factor=0.5, ),**best_params_pretrain) # **best_params_pretrain

unsupervised_model.fit(
    X_train=X_train.values,
    eval_set=[X_val_pc.values],
    pretraining_ratio=0.5,
)


# In[45]:
def objective(trial):
    mask_type = trial.suggest_categorical("mask_type", ["entmax", "sparsemax"])
    # n_d = trial.suggest_int("n_da", 8, 64, step=8)
    n_a = trial.suggest_int("n_a", 8, 64, step=8)
    n_steps = trial.suggest_int("n_steps", 1, 5, step=1)
    gamma = trial.suggest_float("gamma", 1.0, 2.0, step=0.2)
    n_shared = trial.suggest_int("n_shared", 1, 5)
    lambda_sparse = trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True)
    tabnet_params = dict(n_d=n_a, n_a=n_a, n_steps=n_steps, gamma=gamma,
                         lambda_sparse=lambda_sparse, optimizer_fn=torch.optim.Adam,
                         optimizer_params=dict(lr=trial.suggest_categorical('learning_rate', [2e-2, 1e-1, 1e-3]), weight_decay=1e-5),
                         mask_type=mask_type, n_shared=n_shared,
                         scheduler_params=dict(mode="min",
                                               patience=trial.suggest_int("patience", low=3, high=10),
                                               # changing sheduler patience to be lower than early stopping patience
                                               min_lr=1e-5,
                                               factor=0.5, ),
                         scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                         verbose=0,
                         )


    clf = TabNetClassifier(**tabnet_params)
    print(clf)

    clf.fit(
        X_train.values, y_train, weights=1,
        eval_set=[(X_val_pc.values, y_val_pc)],
        eval_metric=['auc'],
        max_epochs=100, from_unsupervised=unsupervised_model
    )
    preds = clf.predict(X_test_pc.values)

    history = clf.history
    acc = accuracy_score(y_test_pc, preds) * 100
    preds_valid = clf.predict_proba(X_val_pc.values)
    valid_auc = roc_auc_score(y_score=preds_valid[:, 1], y_true=y_val_pc)

    return valid_auc

# In[47]:

print()
print("TabNet Classifier Optuna Tuning")
optuna.logging.set_verbosity(optuna.logging.WARNING)
sampler = optuna.samplers.TPESampler(seed=1)
study = optuna.create_study(sampler=sampler, direction="maximize")
study.optimize(objective, n_trials=100)

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

best_params = study.best_params
patience_op = best_params['patience']
del best_params['patience']
print('patience_op:', patience_op)

lr_op = best_params['learning_rate']
del best_params['learning_rate']
print('lr_op', lr_op)


# Use best params from optuna tuning -- for PreTraining and Classifier

clf = TabNetClassifier(optimizer_params=dict(lr=lr_op,
                                             weight_decay=1e-5),
                       scheduler_params=dict(mode="min",
                                               patience=patience_op,
                                               # changing sheduler patience to be lower than early stopping patience
                                               min_lr=1e-5,
                                               factor=0.5, ), **best_params)

# ### Wegiths 1
print()
print('Training TabNet, with optimized Optuna params, weights=1, Train PC+Non PC, Test PC')
print()


# In[51]:
clf.fit(
      X_train.values, y_train,weights=1,
      eval_set=[(X_val_pc.values, y_val_pc)],
               eval_metric=['auc'],
               max_epochs = 100, from_unsupervised=unsupervised_model
    )
preds_val = clf.predict(X_val_pc.values)
preds = clf.predict(X_test_pc.values)

history = clf.history
print(history)
print()

print('Results TabNet, with optimized Optuna params, weights=1, Train PC+Non PC, Test PC')
acc_val = accuracy_score(y_val_pc, preds_val) * 100
print("Validation Accuracy for Optuna Tuned Model: ", acc_val)
preds_val= clf.predict_proba(X_val_pc.values)
val_auc = roc_auc_score(y_score=preds_val[:,1], y_true=y_val_pc)
print("Validation AUC for Optuna Tuned Model: ", val_auc)

acc = accuracy_score(y_test_pc, preds) * 100
print("Test Accuracy for Optuna Tuned Model: ", acc)
preds_test= clf.predict_proba(X_test_pc.values)
test_auc = roc_auc_score(y_score=preds_test[:,1], y_true=y_test_pc)
print("Test AUC for Optuna Tuned Model: ", test_auc)



preds_y_val = clf.predict(X_val_pc.values)



roc_auc = metrics.roc_auc_score(y_test_pc, preds_test[:,1])
print("ROC AUC is: ", roc_auc)

f1 = metrics.f1_score(y_test_pc, preds)
print("F1 score is: ", f1)

precision = metrics.precision_score(y_test_pc, preds)
print("Precision score is: ", precision)

recall = metrics.recall_score(y_test_pc, preds)
print("Recall score is: ", recall)

print(metrics.classification_report(y_test_pc, preds))
print(metrics.confusion_matrix(y_test_pc, preds))

tn, fp, fn, tp = metrics.confusion_matrix(y_test_pc, preds).ravel()
specificity = tn / (tn+fp)
print("Specificity score is: ", specificity)
sensitivity = tp / (tp+fn)
print("Sensitivity score is: ", sensitivity)
print()
print()


# Calculate precision-recall pairs
prec_test, recall_test, _ = metrics.precision_recall_curve(y_test_pc, preds_test[:,1])
# Calculate AUC-PRC
auc_prc_test = metrics.auc(recall_test, prec_test)
print("TEST Precision-Recall AUC:", auc_prc_test)

prec_val, recall_val, _ = metrics.precision_recall_curve(y_val_pc, preds_val[:,1])
# Calculate AUC-PRC
auc_prc_val = metrics.auc(recall_val, prec_val)
print("VALID Precision-Recall AUC:", auc_prc_val)

# Plot Precision-Recall curve -- TEST data
plt.figure()
plt.plot(recall_test, prec_test, color='blue', lw=2, label='Precision-Recall curve (area = %0.2f)' % auc_prc_test)
plt.xlabel('Recall')
plt.ylabel('Precision')
# plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.savefig('logs_MH_subset/figures/TabNet_pre2_entire_data_testPC_prec_recall_curve_over13.png')
print()
print()

# ROC-AUC curve -- Test data
fpr_curve, tpr_curve, _ = metrics.roc_curve(y_test_pc, preds_test[:,1])

# Plot ROC curve
plt.figure()
plt.plot(fpr_curve, tpr_curve, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1 - Specificity') # False Positive Rate
plt.ylabel('Sensitivity') # True Positive Rate
# plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('logs_MH_subset/figures/TabNet_pre2_entire_data_testPC_ROC_curve_over13.png')
print('saved roc-auc curve')
print()

# save test on PC model
# clf.save_model('logs_MH_subset/saved_models/xgboost_entire_data_testPC.json')
# joblib.dump(clf, 'logs_MH_subset/saved_models/xgboost_entire_data_testPC.pkl')
clf.save_model('logs_MH_subset/saved_models/TabNet_pre2_entire_data_testPC_over13')
print("model saved")
print()

# saving metrics and outputs to csv
df_pc = pd.DataFrame()
df_pc['Predicted_probs_val'] = pd.Series(preds_val[:,1])
df_pc['Pred_y_val'] = pd.Series(preds_y_val)
df_pc['GT_y_val'] = pd.Series(y_val_pc.values)

df_pc['Predicted_probs_test'] = pd.Series(preds_test[:,1])
df_pc['Pred_y_test'] = pd.Series(preds)
df_pc['GT_y_test'] = pd.Series(y_test_pc.values)

df_pc['Val_ROC_AUC'] = pd.Series(val_auc)
df_pc['Val_PRC_AUC'] = pd.Series(auc_prc_val)

df_pc['FINAL_Test_AUC'] = pd.Series(test_auc)
df_pc['Test_ROC_AUC'] = pd.Series(roc_auc)
df_pc['Test_PRC_AUC'] = pd.Series(auc_prc_test)

df_pc['Test_F1'] = pd.Series(f1)
df_pc['Test_Precision'] = pd.Series(precision)
df_pc['Test_Recall'] = pd.Series(recall)

df_pc['Test_Specificity'] = pd.Series(specificity)
df_pc['Test_Sensitivity'] = pd.Series(sensitivity)

df_pc['True_Negatives'] = pd.Series(tn)
df_pc['True_Positives'] = pd.Series(tp)
df_pc['False_Negatives'] = pd.Series(fn)
df_pc['False_Positives'] = pd.Series(fp)

df_pc.to_csv('logs_MH_subset/saved_csvs/TabNet_pre2_entire_data_testPC_over13.csv')
print('metrics file saved')
print()


# # ### Weights 0
#
# print('Training TabNet, with optimized Optuna params, weights=0, Train PC+Non PC, Test PC')
# print("weights=0")
# print()
#
#
# # In[44]:
#
# clf.fit(
#       X_train.values, y_train,weights=0,
#       eval_set=[(X_val_pc.values, y_val_pc)],
#                eval_metric=['auc'],
#                max_epochs = 100, from_unsupervised=unsupervised_model
#     )
# preds_val = clf.predict(X_val_pc.values)
# preds = clf.predict(X_test_pc.values)
#
# history = clf.history
# print(history)
# print()
#
# print('Results TabNet, with optimized Optuna params, weights=0, Train PC+Non PC, Test PC')
# acc_val = accuracy_score(y_val_pc, preds_val) * 100
# print("Validation Accuracy for Optuna Tuned Model: ", acc_val)
# preds_val= clf.predict_proba(X_val_pc.values)
# val_auc = roc_auc_score(y_score=preds_val[:,1], y_true=y_val_pc)
# print("Validation AUC for Optuna Tuned Model: ", val_auc)
#
# acc = accuracy_score(y_test_pc, preds) * 100
# print("Test Accuracy for Optuna Tuned Model: ", acc)
# preds_test= clf.predict_proba(X_test_pc.values)
# test_auc = roc_auc_score(y_score=preds_test[:,1], y_true=y_test_pc)
# print("Test AUC for Optuna Tuned Model: ", test_auc)
#
# print("ROC AUC is: ",metrics.roc_auc_score(y_test_pc, preds_test[:,1]))
# print("F1 score is: ", metrics.f1_score(y_test_pc, preds))
# print("Preicision score is: ", metrics.precision_score(y_test_pc, preds))
# print("Recall score is: ", metrics.recall_score(y_test_pc, preds))
# print(metrics.classification_report(y_test_pc, preds))
# print(metrics.confusion_matrix(y_test_pc, preds))
# tn, fp, fn, tp = metrics.confusion_matrix(y_test_pc, preds).ravel()
# specificity = tn / (tn+fp)
# print("Specificity score is: ", specificity)
# sensitivity = tp / (tp+fn)
# print("Sensitivity score is: ", sensitivity)
# print()
# print()
# print('-----------------------------')
# print()



####################################################
####################################################
####################################################
####################################################



print()
print()

print('Training TabNet, optimizing with Optuna, weights=1, Train PC+Non PC, Test Non PC')
print()

def objective_pretrain(trial):
    # parameters to tune
    mask_type = trial.suggest_categorical("mask_type", ["entmax", "sparsemax"])
    # n_d = trial.suggest_int("n_da", 8, 64, step=8)
    n_a = trial.suggest_int("n_a", 8, 64, step=8)
    n_steps = trial.suggest_int("n_steps", 1, 5, step=1)
    gamma = trial.suggest_float("gamma", 1.0, 2.0, step=0.2)
    n_shared = trial.suggest_int("n_shared", 1, 5)
    lambda_sparse = trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True)
    tabnet_params_pretrain = dict(n_d=n_a, n_a=n_a, n_steps=n_steps, gamma=gamma,
                         lambda_sparse=lambda_sparse, optimizer_fn=torch.optim.Adam,
                         optimizer_params=dict(lr=trial.suggest_categorical('learning_rate', [2e-2, 1e-1, 1e-3]), weight_decay=1e-5),
                         mask_type=mask_type, n_shared=n_shared,
                         scheduler_params=dict(mode="min",
                                               patience=trial.suggest_int("patience", low=3, high=10),
                                               # changing sheduler patience to be lower than early stopping patience
                                               min_lr=1e-5,
                                               factor=0.5, ),
                         scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                         verbose=0,
                         )

    unsupervised_model = TabNetPretrainer(**tabnet_params_pretrain) # **tabnet_params_pretrain

    unsupervised_model.fit(
        X_train=X_train.values,
        eval_set=[X_val_non_pc.values],
        pretraining_ratio=0.5,
    )

    # preds = clf.predict(X_test.values)

    # history = clf.history
    # acc = accuracy_score(y_test, preds) * 100
    # Make reconstruction from a dataset
    # reconstructed_X, embedded_X = unsupervised_model.predict(X_val_non_pc.values)
    # assert (reconstructed_X.shape == embedded_X.shape)

    # preds_valid = unsupervised_model.predict_proba(X_val.values)
    # valid_auc = roc_auc_score(y_score=preds_valid[:, 1], y_true=y_val)

    return unsupervised_model.best_cost


# In[47]:
print("TabNet Pre-Training Optuna Tuning")
optuna.logging.set_verbosity(optuna.logging.WARNING)
sampler = optuna.samplers.TPESampler(seed=1)
study = optuna.create_study(sampler=sampler, direction="minimize")
study.optimize(objective_pretrain, n_trials=100)

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

best_params_pretrain = study.best_params
patience_op = best_params_pretrain['patience']
del best_params_pretrain['patience']
print('patience_op:', patience_op)

lr_op = best_params_pretrain['learning_rate']
del best_params_pretrain['learning_rate']
print('lr_op', lr_op)


unsupervised_model = TabNetPretrainer(optimizer_params=dict(lr=lr_op,
                                             weight_decay=1e-5),
                                      scheduler_params=dict(mode="min",
                                               patience=patience_op,
                                               # changing sheduler patience to be lower than early stopping patience
                                               min_lr=1e-5,
                                               factor=0.5, ),**best_params_pretrain) # **best_params_pretrain

unsupervised_model.fit(
    X_train=X_train.values,
    eval_set=[X_val_non_pc.values],
    pretraining_ratio=0.5,
)


# In[45]:
def objective(trial):
    mask_type = trial.suggest_categorical("mask_type", ["entmax", "sparsemax"])
    # n_d = trial.suggest_int("n_da", 8, 64, step=8)
    n_a = trial.suggest_int("n_a", 8, 64, step=8)
    n_steps = trial.suggest_int("n_steps", 1, 5, step=1)
    gamma = trial.suggest_float("gamma", 1.0, 2.0, step=0.2)
    n_shared = trial.suggest_int("n_shared", 1, 5)
    lambda_sparse = trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True)
    tabnet_params = dict(n_d=n_a, n_a=n_a, n_steps=n_steps, gamma=gamma,
                         lambda_sparse=lambda_sparse, optimizer_fn=torch.optim.Adam,
                         optimizer_params=dict(lr=trial.suggest_categorical('learning_rate', [2e-2, 1e-1, 1e-3]), weight_decay=1e-5),
                         mask_type=mask_type, n_shared=n_shared,
                         scheduler_params=dict(mode="min",
                                               patience=trial.suggest_int("patience", low=3, high=10),
                                               # changing sheduler patience to be lower than early stopping patience
                                               min_lr=1e-5,
                                               factor=0.5, ),
                         scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                         verbose=0,
                         )


    clf = TabNetClassifier(**tabnet_params)
    print(clf)

    clf.fit(
        X_train.values, y_train, weights=1,
        eval_set=[(X_val_non_pc.values, y_val_non_pc)],
        eval_metric=['auc'],
        max_epochs=100, from_unsupervised=unsupervised_model
    )
    preds = clf.predict(X_test_non_pc.values)

    history = clf.history
    acc = accuracy_score(y_test_non_pc, preds) * 100
    preds_valid = clf.predict_proba(X_val_non_pc.values)
    valid_auc = roc_auc_score(y_score=preds_valid[:, 1], y_true=y_val_non_pc)

    return valid_auc

# In[47]:

print()
print("TabNet Classifier Optuna Tuning")
optuna.logging.set_verbosity(optuna.logging.WARNING)
sampler = optuna.samplers.TPESampler(seed=1)
study = optuna.create_study(sampler=sampler, direction="maximize")
study.optimize(objective, n_trials=100)

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

best_params = study.best_params
patience_op = best_params['patience']
del best_params['patience']
print('patience_op:', patience_op)

lr_op = best_params['learning_rate']
del best_params['learning_rate']
print('lr_op', lr_op)


# Use best params from optuna tuning -- for PreTraining and Classifier

clf = TabNetClassifier(optimizer_params=dict(lr=lr_op,
                                             weight_decay=1e-5),
                       scheduler_params=dict(mode="min",
                                               patience=patience_op,
                                               # changing sheduler patience to be lower than early stopping patience
                                               min_lr=1e-5,
                                               factor=0.5, ), **best_params)

# ### Wegiths 1
print()
print('Training TabNet, with optimized Optuna params, weights=1, Train PC+Non PC, Test Non PC')
print()


# In[51]:
clf.fit(
      X_train.values, y_train,weights=1,
      eval_set=[(X_val_non_pc.values, y_val_non_pc)],
               eval_metric=['auc'],
               max_epochs = 100, from_unsupervised=unsupervised_model
    )
preds_val = clf.predict(X_val_non_pc.values)
preds = clf.predict(X_test_non_pc.values)

history = clf.history
print(history)
print()

print('Results TabNet, with optimized Optuna params, weights=1, Train PC+Non PC, Test Non PC')
acc_val = accuracy_score(y_val_non_pc, preds_val) * 100
print("Validation Accuracy for Optuna Tuned Model: ", acc_val)
preds_val= clf.predict_proba(X_val_non_pc.values)
val_auc = roc_auc_score(y_score=preds_val[:,1], y_true=y_val_non_pc)
print("Validation AUC for Optuna Tuned Model: ", val_auc)

acc = accuracy_score(y_test_non_pc, preds) * 100
print("Test Accuracy for Optuna Tuned Model: ", acc)
preds_test= clf.predict_proba(X_test_non_pc.values)
test_auc = roc_auc_score(y_score=preds_test[:,1], y_true=y_test_non_pc)
print("Test AUC for Optuna Tuned Model: ", test_auc)


preds_y_val = clf.predict(X_val_non_pc.values)



roc_auc = metrics.roc_auc_score(y_test_non_pc, preds_test[:,1])
print("ROC AUC is: ", roc_auc)

f1 = metrics.f1_score(y_test_non_pc, preds)
print("F1 score is: ", f1)

precision = metrics.precision_score(y_test_non_pc, preds)
print("Precision score is: ", precision)

recall = metrics.recall_score(y_test_non_pc, preds)
print("Recall score is: ", recall)

print(metrics.classification_report(y_test_non_pc, preds))
print(metrics.confusion_matrix(y_test_non_pc, preds))

tn, fp, fn, tp = metrics.confusion_matrix(y_test_non_pc, preds).ravel()
specificity = tn / (tn+fp)
print("Specificity score is: ", specificity)
sensitivity = tp / (tp+fn)
print("Sensitivity score is: ", sensitivity)
print()
print()



# Calculate precision-recall pairs
prec_test, recall_test, _ = metrics.precision_recall_curve(y_test_non_pc, preds_test[:,1])
# Calculate AUC-PRC
auc_prc_test = metrics.auc(recall_test, prec_test)
print("TEST Precision-Recall AUC:", auc_prc_test)

prec_val, recall_val, _ = metrics.precision_recall_curve(y_val_non_pc, preds_val[:,1])
# Calculate AUC-PRC
auc_prc_val = metrics.auc(recall_val, prec_val)
print("VALID Precision-Recall AUC:", auc_prc_val)

# Plot Precision-Recall curve -- TEST data
plt.figure()
plt.plot(recall_test, prec_test, color='blue', lw=2, label='Precision-Recall curve (area = %0.2f)' % auc_prc_test)
plt.xlabel('Recall')
plt.ylabel('Precision')
# plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.savefig('logs_MH_subset/figures/TabNet_pre2_entire_data_testNonPC_prec_recall_curve_over13.png')
print()
print()

# ROC-AUC curve -- Test data
fpr_curve, tpr_curve, _ = metrics.roc_curve(y_test_non_pc, preds_test[:,1])

# Plot ROC curve
plt.figure()
plt.plot(fpr_curve, tpr_curve, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1 - Specificity') # False Positive Rate
plt.ylabel('Sensitivity') # True Positive Rate
# plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('logs_MH_subset/figures/TabNet_pre2_entire_data_testNonPC_ROC_curve_over13.png')
print('saved roc-auc curve')
print()

# save test on PC model
# clf.save_model('logs_MH_subset/saved_models/xgboost_entire_data_testPC.json')
# joblib.dump(clf, 'logs_MH_subset/saved_models/xgboost_entire_data_testNonPC.pkl')
clf.save_model('logs_MH_subset/saved_models/TabNet_pre2_entire_data_testNonPC_over13')
print("model saved")
print()

# saving metrics and outputs to csv
df_pc = pd.DataFrame()
df_pc['Predicted_probs_val'] = pd.Series(preds_val[:,1])
df_pc['Pred_y_val'] = pd.Series(preds_y_val)
df_pc['GT_y_val'] = pd.Series(y_val_non_pc.values)

df_pc['Predicted_probs_test'] = pd.Series(preds_test[:,1])
df_pc['Pred_y_test'] = pd.Series(preds)
df_pc['GT_y_test'] = pd.Series(y_test_non_pc.values)

df_pc['Val_ROC_AUC'] = pd.Series(val_auc)
df_pc['Val_PRC_AUC'] = pd.Series(auc_prc_val)

df_pc['FINAL_Test_AUC'] = pd.Series(test_auc)
df_pc['Test_ROC_AUC'] = pd.Series(roc_auc)
df_pc['Test_PRC_AUC'] = pd.Series(auc_prc_test)

df_pc['Test_F1'] = pd.Series(f1)
df_pc['Test_Precision'] = pd.Series(precision)
df_pc['Test_Recall'] = pd.Series(recall)

df_pc['Test_Specificity'] = pd.Series(specificity)
df_pc['Test_Sensitivity'] = pd.Series(sensitivity)

df_pc['True_Negatives'] = pd.Series(tn)
df_pc['True_Positives'] = pd.Series(tp)
df_pc['False_Negatives'] = pd.Series(fn)
df_pc['False_Positives'] = pd.Series(fp)

df_pc.to_csv('logs_MH_subset/saved_csvs/TabNet_pre2_entire_data_testNonPC_over13.csv')
print('metrics file saved')
print()



# # ### Weights 0
#
# print('Training TabNet, with optimized Optuna params, weights=0, Train PC+Non PC, Test Non PC')
# print("weights=0")
# print()
#
#
# # In[44]:
#
# clf.fit(
#       X_train.values, y_train,weights=0,
#       eval_set=[(X_val_non_pc.values, y_val_non_pc)],
#                eval_metric=['auc'],
#                max_epochs = 100, from_unsupervised=unsupervised_model
#     )
# preds_val = clf.predict(X_val_non_pc.values)
# preds = clf.predict(X_test_non_pc.values)
#
# history = clf.history
# print(history)
# print()
#
# print('Results TabNet, with optimized Optuna params, weights=0, Train PC+Non PC, Test Non PC')
# acc_val = accuracy_score(y_val_non_pc, preds_val) * 100
# print("Validation Accuracy for Optuna Tuned Model: ", acc_val)
# preds_val= clf.predict_proba(X_val_non_pc.values)
# val_auc = roc_auc_score(y_score=preds_val[:,1], y_true=y_val_non_pc)
# print("Validation AUC for Optuna Tuned Model: ", val_auc)
#
# acc = accuracy_score(y_test_non_pc, preds) * 100
# print("Test Accuracy for Optuna Tuned Model: ", acc)
# preds_test = clf.predict_proba(X_test_non_pc.values)
# test_auc = roc_auc_score(y_score=preds_test[:,1], y_true=y_test_non_pc)
# print("Test AUC for Optuna Tuned Model: ", test_auc)
#
# print("ROC AUC is: ",metrics.roc_auc_score(y_test_non_pc, preds_test[:,1]))
# print("F1 score is: ", metrics.f1_score(y_test_non_pc, preds))
# print("Preicision score is: ", metrics.precision_score(y_test_non_pc, preds))
# print("Recall score is: ", metrics.recall_score(y_test_non_pc, preds))
# print(metrics.classification_report(y_test_non_pc, preds))
# print(metrics.confusion_matrix(y_test_non_pc, preds))
# tn, fp, fn, tp = metrics.confusion_matrix(y_test_non_pc, preds).ravel()
# specificity = tn / (tn+fp)
# print("Specificity score is: ", specificity)
# sensitivity = tp / (tp+fn)
# print("Sensitivity score is: ", sensitivity)
# print()
# print()
# print('-----------------------------')
# print()


print('DONE')

end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")



