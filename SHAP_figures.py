# In[1]:
import numpy as np
# import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
import warnings
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
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
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import argparse
import pickle
import time
import random



SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


start_time = time.time()

# In[2]:


# data1 = pd.read_csv('MHRN4_with_zip_income_college_binarized_new.csv')
data1 = pd.read_csv('combined_pc_mh_data.csv')

# filter for patients 13+  
data = data1[data1['age'] >= 13]
min_value = data['age'].min()
print()
print('-----------------------')
print('MIN VALUE:', min_value)
print('-----------------------')
print()

# drop cols not overlap mhrn 
columns_to_drop_idx = ['antidep_rx_pre3m_idx', 'antidep_rx_pre1y_cumulative_idx', 'antidep_rx_pre5y_cumulative_idx', 'benzo_rx_pre3m_idx', 'benzo_rx_pre1y_cumulative_idx', 'benzo_rx_pre5y_cumulative_idx', 'hypno_rx_pre3m_idx', 'hypno_rx_pre1y_cumulative_idx', 'hypno_rx_pre5y_cumulative_idx', 'sga_rx_pre3m_idx', 'sga_rx_pre1y_cumulative_idx', 'sga_rx_pre5y_cumulative_idx', 'mh_ip_pre3m_idx', 'mh_ip_pre1y_cumulative_idx', 'mh_ip_pre5y_cumulative_idx', 'mh_op_pre3m_idx', 'mh_op_pre1y_cumulative_idx', 'mh_op_pre5y_cumulative_idx', 'mh_ed_pre3m_idx', 'mh_ed_pre1y_cumulative_idx', 'mh_ed_pre5y_cumulative_idx', 'any_sui_att_pre3m_idx', 'any_sui_att_pre1y_cumulative_idx', 'any_sui_att_pre5y_cumulative_idx', 'any_sui_att_pre5y_cumulative_idx_a', 'any_sui_att_pre5y_cumulative_idx_f', 'lvi_sui_att_pre3m_idx', 'lvi_sui_att_pre1y_cumulative_idx', 'lvi_sui_att_pre5y_cumulative_idx', 'ovi_sui_att_pre3m_idx', 'ovi_sui_att_pre1y_cumulative_idx', 'ovi_sui_att_pre5y_cumulative_idx', 'any_inj_poi_pre3m_idx', 'any_inj_poi_pre1y_cumulative_idx', 'any_inj_poi_pre5y_cumulative_idx']
data = data.drop(columns=columns_to_drop_idx)


# In[7]:
data["event90"] = data["event90"].fillna(value=0)



data = data.drop(columns=["income", "college", "hhld_inc_It40k", "coll_deg_It25p"])

# fix dropped cols above
data.rename(columns={'hhld_inc_lt40k_NJ': 'hhld_inc_It40k', 'coll_deg_lt25p_NJ': 'coll_deg_It25p'}, inplace=True)



# ## Selecting Primary Care data where Flag(PRIMARY_CARE_VISIT) == 1

# In[9]:
pc_data = data[data["PRIMARY_CARE_VISIT"] == 1]
# print(pc_data)


# In[10]:
y = pc_data["event90"]


# ## Selecting Non Primary Care data where Flag(PRIMARY_CARE_VISIT) == 0

# In[12]:
non_pc_data = data[data["PRIMARY_CARE_VISIT"] == 0]
# print(non_pc_data)


# In[13]:
y_non_pc = non_pc_data["event90"]


# ## Scaling data


# In[20]:
pc_data = pc_data.drop(columns=["PRIMARY_CARE_VISIT","person_id","event30","event90","death30","death90","visit_mh"])
# print(pc_data)



# # In[22]:
# non_pc_columns_to_scale = ["age","days_since_prev","charlson_score","charlson_a","dep_dx_pre5y_cumulative_a","anx_dx_pre5y_cumulative_a","bip_dx_pre5y_cumulative_a","sch_dx_pre5y_cumulative_a","phqnumber90","phqnumber183","phqnumber365","phq8_index_score_calc_f","raceAsian_8","raceIN_8","hispanic_8","age_8","q9_0_a","q9_1_8","q9_2_8","q9_3_8","q9_1_c","q9_2_c","q9_3_c","any_sui_att_pre5y_cumulative_a","any_sui_att_pre5y_cumulative_8","any_sui_att_pre5y_cumulative_c"]



# In[25]:
non_pc_data = non_pc_data.drop(columns=["PRIMARY_CARE_VISIT","person_id","event30","event90","death30","death90","visit_mh"])
# print(non_pc_data)


# In[16]:
# pc_columns_to_scale = ["age","days_since_prev","charlson_score","charlson_a","dep_dx_pre5y_cumulative_a","anx_dx_pre5y_cumulative_a","bip_dx_pre5y_cumulative_a","sch_dx_pre5y_cumulative_a","phqnumber90","phqnumber183","phqnumber365","phq8_index_score_calc_f","raceAsian_8","raceIN_8","hispanic_8","age_8","q9_0_a","q9_1_8","q9_2_8","q9_3_8","q9_1_c","q9_2_c","q9_3_c","any_sui_att_pre5y_cumulative_a","any_sui_att_pre5y_cumulative_8","any_sui_att_pre5y_cumulative_c"]
pc_columns_to_scale = ['age', 'days_since_prev', 'dep_dx_pre5y_noi_cumulative', 'anx_dx_pre5y_noi_cumulative', 'bip_dx_pre5y_noi_cumulative', 'sch_dx_pre5y_noi_cumulative', 'oth_dx_pre5y_noi_cumulative', 'dem_dx_pre5y_noi_cumulative', 'add_dx_pre5y_noi_cumulative', 'asd_dx_pre5y_noi_cumulative', 'per_dx_pre5y_noi_cumulative', 'alc_dx_pre5y_noi_cumulative', 'pts_dx_pre5y_noi_cumulative', 'eat_dx_pre5y_noi_cumulative', 'tbi_dx_pre5y_noi_cumulative', 'dru_dx_pre5y_noi_cumulative', 'antidep_rx_pre1y_cumulative', 'antidep_rx_pre5y_cumulative', 'benzo_rx_pre1y_cumulative', 'benzo_rx_pre5y_cumulative', 'hypno_rx_pre1y_cumulative', 'hypno_rx_pre5y_cumulative', 'sga_rx_pre1y_cumulative', 'sga_rx_pre5y_cumulative', 'mh_ip_pre1y_cumulative', 'mh_ip_pre5y_cumulative', 'mh_op_pre1y_cumulative', 'mh_op_pre5y_cumulative', 'mh_ed_pre1y_cumulative', 'mh_ed_pre5y_cumulative', 'any_sui_att_pre1y_cumulative', 'any_sui_att_pre5y_cumulative', 'lvi_sui_att_pre1y_cumulative', 'lvi_sui_att_pre5y_cumulative', 'ovi_sui_att_pre1y_cumulative', 'ovi_sui_att_pre5y_cumulative', 'any_inj_poi_pre1y_cumulative', 'any_inj_poi_pre5y_cumulative', 'any_sui_att_pre5y_cumulative_f', 'any_sui_att_pre5y_cumulative_a', 'charlson_score', 'charlson_a', 'phqnumber90', 'phqnumber183', 'phqnumber365', 'dep_dx_pre5y_cumulative', 'dep_dx_pre5y_cumulative_f', 'dep_dx_pre5y_cumulative_a', 'anx_dx_pre5y_cumulative', 'anx_dx_pre5y_cumulative_f', 'anx_dx_pre5y_cumulative_a', 'bip_dx_pre5y_cumulative', 'bip_dx_pre5y_cumulative_f', 'bip_dx_pre5y_cumulative_a', 'sch_dx_pre5y_cumulative', 'sch_dx_pre5y_cumulative_f', 'sch_dx_pre5y_cumulative_a', 'oth_dx_pre5y_cumulative', 'dem_dx_pre5y_cumulative', 'add_dx_pre5y_cumulative', 'asd_dx_pre5y_cumulative', 'per_dx_pre5y_cumulative', 'alc_dx_pre5y_cumulative', 'dru_dx_pre5y_cumulative', 'pts_dx_pre5y_cumulative', 'eat_dx_pre5y_cumulative', 'tbi_dx_pre5y_cumulative', 'phq8_index_score_calc', 'phq8_index_score_calc_f', 'raceAsian_8', 'raceBlack_8', 'raceHP_8', 'raceIN_8', 'raceMUOT_8', 'raceUN_8', 'hispanic_8', 'age_8', 'q9_0_a', 'q9_1_a', 'q9_2_a', 'q9_3_a', 'q9_0_8', 'q9_1_8', 'q9_2_8', 'q9_3_8', 'q9_0_c', 'q9_1_c', 'q9_2_c', 'q9_3_c', 'any_sui_att_pre5y_cumulative_8', 'any_sui_att_pre5y_cumulative_c', 'any_sui_att_pre5y_cumulative_de', 'any_sui_att_pre5y_cumulative_an', 'any_sui_att_pre5y_cumulative_bi', 'any_sui_att_pre5y_cumulative_sc', 'any_sui_att_pre5y_cumulative_al', 'any_sui_att_pre5y_cumulative_dr', 'any_sui_att_pre5y_cumulative_pe']
# these columns to scale are same as all the numeric (num_cols) columns

# In[17]:
count = 0
for i in pc_columns_to_scale:
    if i in pc_data.columns:
        count += 1
    else:
        print(i)
print(count)


# In[18]:
pc_data[pc_columns_to_scale] = scale(pc_data[pc_columns_to_scale])


# ### Dealing with missing values

# In[27]:
missing_columns = pc_data.columns[pc_data.isnull().any()]



# # In[29]:
# plt.figure(figsize=(10,6))
# sns.heatmap(pc_data[missing_columns].isna().transpose(),
#             cmap="YlGnBu",
#             cbar_kws={'label': 'Missing Data'})
# plt.show()


# In[30]:
for i in missing_columns:
    if sum(pc_data[i].isnull()) == pc_data.shape[0]: #178460: #188894:
        print(i)
        print(sum(pc_data[i].isnull()))
    else:
        print(i)
        print(sum(pc_data[i].isnull()))



bin_cols = ['event30', 'event90', 'death30', 'death90', 'visit_mh', 'ac1', 'ac2', 'ac3', 'ac4', 'ac5', 'ac1f', 'ac3f', 'ac4f', 'ac5f', 'Enrolled', 'medicaid', 'commercial', 'privatepay', 'statesubsidized', 'selffunded', 'medicare', 'highdedectible', 'other', 'first_visit', 'female', 'dep_dx_pre5y', 'anx_dx_pre5y', 'bip_dx_pre5y', 'sch_dx_pre5y', 'oth_dx_pre5y', 'dem_dx_pre5y', 'add_dx_pre5y', 'asd_dx_pre5y', 'per_dx_pre5y', 'alc_dx_pre5y', 'pts_dx_pre5y', 'eat_dx_pre5y', 'tbi_dx_pre5y', 'dru_dx_pre5y', 'antidep_rx_pre3m', 'benzo_rx_pre3m', 'hypno_rx_pre3m', 'sga_rx_pre3m', 'mh_ip_pre3m', 'mh_op_pre3m', 'mh_ed_pre3m', 'any_sui_att_pre3m', 'lvi_sui_att_pre3m', 'ovi_sui_att_pre3m', 'any_inj_poi_pre3m', 'current_pregnancy', 'del_pre_1_90', 'del_pre_1_180', 'del_pre_1_365', 'charlson_mi', 'charlson_chd', 'charlson_pvd', 'charlson_cvd', 'charlson_dem', 'charlson_cpd', 'charlson_rhd', 'charlson_pud', 'charlson_mlivd', 'charlson_diab', 'charlson_diabc', 'charlson_plegia', 'charlson_ren', 'charlson_malign', 'charlson_slivd', 'charlson_mst', 'charlson_aids', 'raceAsian_asa', 'raceBlack_asa', 'raceHP_asa', 'raceIN_asa', 'raceMUOT_asa', 'raceUN_asa', 'hispanic_asa', 'raceAsian', 'raceBlack', 'raceHP', 'raceIN', 'raceMUOT', 'raceUN', 'raceWH', 'hispanic', 'raceAsian_f', 'raceBlack_f', 'raceHP_f', 'raceIN_f', 'raceMUOT_f', 'raceUN_f', 'hispanic_f', 'census_missing', 'hhld_inc_It40k', 'coll_deg_It25p', 'phqmode90_0', 'phqmode90_1', 'phqmode90_2', 'phqmax90_0', 'phqmax90_1', 'phqmax90_2', 'phqmax90_3', 'phqmode183_0', 'phqmode183_1', 'phqmode183_2', 'phqmax183_0', 'phqmax183_1', 'phqmax183_2', 'phqmax183_3', 'phqmode365_0', 'phqmode365_1', 'phqmode365_2', 'phqmax365_0', 'phqmax365_1', 'phqmax365_2', 'phqmax365_3', 'raceAsian_de', 'raceBlack_de', 'raceHP_de', 'raceIN_de', 'raceMUOT_de', 'raceUN_de', 'hispanic_de', 'raceAsian_an', 'raceBlack_an', 'raceHP_an', 'raceIN_an', 'raceMUOT_an', 'raceUN_an', 'hispanic_an', 'raceAsian_bi', 'raceBlack_bi', 'raceHP_bi', 'raceIN_bi', 'raceMUOT_bi', 'raceUN_bi', 'hispanic_bi', 'raceAsian_sc', 'raceBlack_sc', 'raceHP_sc', 'raceIN_sc', 'raceMUOT_sc', 'raceUN_sc', 'hispanic_sc', 'phq8_missing', 'phq8_missing_f', 'q9_0', 'q9_1', 'q9_2', 'q9_3', 'q9_0_f', 'q9_1_f', 'q9_2_f', 'q9_3_f', 'raceAsian_q90', 'raceBlack_q90', 'raceHP_q90', 'raceIN_q90', 'raceMUOT_q90', 'raceUN_q90', 'hispanic_q90', 'raceAsian_q91', 'raceBlack_q91', 'raceHP_q91', 'raceIN_q91', 'raceMUOT_q91', 'raceUN_q91', 'hispanic_q91', 'raceAsian_q92', 'raceBlack_q92', 'raceHP_q92', 'raceIN_q92', 'raceMUOT_q92', 'raceUN_q92', 'hispanic_q92', 'raceAsian_q93', 'raceBlack_q93', 'raceHP_q93', 'raceIN_q93', 'raceMUOT_q93', 'raceUN_q93', 'hispanic_q93', 'q9_0_de', 'q9_1_de', 'q9_2_de', 'q9_3_de', 'q9_0_an', 'q9_1_an', 'q9_2_an', 'q9_3_an', 'q9_0_bi', 'q9_1_bi', 'q9_2_bi', 'q9_3_bi', 'q9_0_sc', 'q9_1_sc', 'q9_2_sc', 'q9_3_sc', 'q9_0_al', 'q9_1_al', 'q9_2_al', 'q9_3_al', 'q9_0_dr', 'q9_1_dr', 'q9_2_dr', 'q9_3_dr', 'q9_0_pe', 'q9_1_pe', 'q9_2_pe', 'q9_3_pe', 'phqMax90_0_q90', 'phqMax90_1_q90', 'phqMax90_2_q90', 'phqMax90_3_q90', 'phqMax90_0_q91', 'phqMax90_1_q91', 'phqMax90_2_q91', 'phqMax90_3_q91', 'phqMax90_0_q92', 'phqMax90_1_q92', 'phqMax90_2_q92', 'phqMax90_3_q92', 'phqMax90_0_q93', 'phqMax90_1_q93', 'phqMax90_2_q93', 'phqMax90_3_q93']

num_cols = ['age', 'days_since_prev', 'dep_dx_pre5y_noi_cumulative', 'anx_dx_pre5y_noi_cumulative', 'bip_dx_pre5y_noi_cumulative', 'sch_dx_pre5y_noi_cumulative', 'oth_dx_pre5y_noi_cumulative', 'dem_dx_pre5y_noi_cumulative', 'add_dx_pre5y_noi_cumulative', 'asd_dx_pre5y_noi_cumulative', 'per_dx_pre5y_noi_cumulative', 'alc_dx_pre5y_noi_cumulative', 'pts_dx_pre5y_noi_cumulative', 'eat_dx_pre5y_noi_cumulative', 'tbi_dx_pre5y_noi_cumulative', 'dru_dx_pre5y_noi_cumulative', 'antidep_rx_pre1y_cumulative', 'antidep_rx_pre5y_cumulative', 'benzo_rx_pre1y_cumulative', 'benzo_rx_pre5y_cumulative', 'hypno_rx_pre1y_cumulative', 'hypno_rx_pre5y_cumulative', 'sga_rx_pre1y_cumulative', 'sga_rx_pre5y_cumulative', 'mh_ip_pre1y_cumulative', 'mh_ip_pre5y_cumulative', 'mh_op_pre1y_cumulative', 'mh_op_pre5y_cumulative', 'mh_ed_pre1y_cumulative', 'mh_ed_pre5y_cumulative', 'any_sui_att_pre1y_cumulative', 'any_sui_att_pre5y_cumulative', 'lvi_sui_att_pre1y_cumulative', 'lvi_sui_att_pre5y_cumulative', 'ovi_sui_att_pre1y_cumulative', 'ovi_sui_att_pre5y_cumulative', 'any_inj_poi_pre1y_cumulative', 'any_inj_poi_pre5y_cumulative', 'any_sui_att_pre5y_cumulative_f', 'any_sui_att_pre5y_cumulative_a', 'charlson_score', 'charlson_a', 'phqnumber90', 'phqnumber183', 'phqnumber365', 'dep_dx_pre5y_cumulative', 'dep_dx_pre5y_cumulative_f', 'dep_dx_pre5y_cumulative_a', 'anx_dx_pre5y_cumulative', 'anx_dx_pre5y_cumulative_f', 'anx_dx_pre5y_cumulative_a', 'bip_dx_pre5y_cumulative', 'bip_dx_pre5y_cumulative_f', 'bip_dx_pre5y_cumulative_a', 'sch_dx_pre5y_cumulative', 'sch_dx_pre5y_cumulative_f', 'sch_dx_pre5y_cumulative_a', 'oth_dx_pre5y_cumulative', 'dem_dx_pre5y_cumulative', 'add_dx_pre5y_cumulative', 'asd_dx_pre5y_cumulative', 'per_dx_pre5y_cumulative', 'alc_dx_pre5y_cumulative', 'dru_dx_pre5y_cumulative', 'pts_dx_pre5y_cumulative', 'eat_dx_pre5y_cumulative', 'tbi_dx_pre5y_cumulative', 'phq8_index_score_calc', 'phq8_index_score_calc_f', 'raceAsian_8', 'raceBlack_8', 'raceHP_8', 'raceIN_8', 'raceMUOT_8', 'raceUN_8', 'hispanic_8', 'age_8', 'q9_0_a', 'q9_1_a', 'q9_2_a', 'q9_3_a', 'q9_0_8', 'q9_1_8', 'q9_2_8', 'q9_3_8', 'q9_0_c', 'q9_1_c', 'q9_2_c', 'q9_3_c', 'any_sui_att_pre5y_cumulative_8', 'any_sui_att_pre5y_cumulative_c', 'any_sui_att_pre5y_cumulative_de', 'any_sui_att_pre5y_cumulative_an', 'any_sui_att_pre5y_cumulative_bi', 'any_sui_att_pre5y_cumulative_sc', 'any_sui_att_pre5y_cumulative_al', 'any_sui_att_pre5y_cumulative_dr', 'any_sui_att_pre5y_cumulative_pe']


for i in missing_columns:
    if sum(pc_data[i].isnull()) == pc_data.shape[0]: #412045: #446893:
        # data[i].fillna(value=0,inplace=True)
        pc_data = pc_data.drop(columns=[i]) # if all rows are empty for a column, drop that column
        # print(sum(data[i].isnull()))
        print("Empty column:", i)
    else:
        pc_data[i].fillna(value=-1, inplace=True) # # if only some of rows have missing value for a column, then fill that with -1, for num or bin cols



# In[22]:
# non_pc_columns_to_scale = ["age","days_since_prev","charlson_score","charlson_a","dep_dx_pre5y_cumulative_a","anx_dx_pre5y_cumulative_a","bip_dx_pre5y_cumulative_a","sch_dx_pre5y_cumulative_a","phqnumber90","phqnumber183","phqnumber365","phq8_index_score_calc_f","raceAsian_8","raceIN_8","hispanic_8","age_8","q9_0_a","q9_1_8","q9_2_8","q9_3_8","q9_1_c","q9_2_c","q9_3_c","any_sui_att_pre5y_cumulative_a","any_sui_att_pre5y_cumulative_8","any_sui_att_pre5y_cumulative_c"]
non_pc_columns_to_scale = ['age', 'days_since_prev', 'dep_dx_pre5y_noi_cumulative', 'anx_dx_pre5y_noi_cumulative', 'bip_dx_pre5y_noi_cumulative', 'sch_dx_pre5y_noi_cumulative', 'oth_dx_pre5y_noi_cumulative', 'dem_dx_pre5y_noi_cumulative', 'add_dx_pre5y_noi_cumulative', 'asd_dx_pre5y_noi_cumulative', 'per_dx_pre5y_noi_cumulative', 'alc_dx_pre5y_noi_cumulative', 'pts_dx_pre5y_noi_cumulative', 'eat_dx_pre5y_noi_cumulative', 'tbi_dx_pre5y_noi_cumulative', 'dru_dx_pre5y_noi_cumulative', 'antidep_rx_pre1y_cumulative', 'antidep_rx_pre5y_cumulative', 'benzo_rx_pre1y_cumulative', 'benzo_rx_pre5y_cumulative', 'hypno_rx_pre1y_cumulative', 'hypno_rx_pre5y_cumulative', 'sga_rx_pre1y_cumulative', 'sga_rx_pre5y_cumulative', 'mh_ip_pre1y_cumulative', 'mh_ip_pre5y_cumulative', 'mh_op_pre1y_cumulative', 'mh_op_pre5y_cumulative', 'mh_ed_pre1y_cumulative', 'mh_ed_pre5y_cumulative', 'any_sui_att_pre1y_cumulative', 'any_sui_att_pre5y_cumulative', 'lvi_sui_att_pre1y_cumulative', 'lvi_sui_att_pre5y_cumulative', 'ovi_sui_att_pre1y_cumulative', 'ovi_sui_att_pre5y_cumulative', 'any_inj_poi_pre1y_cumulative', 'any_inj_poi_pre5y_cumulative', 'any_sui_att_pre5y_cumulative_f', 'any_sui_att_pre5y_cumulative_a', 'charlson_score', 'charlson_a', 'phqnumber90', 'phqnumber183', 'phqnumber365', 'dep_dx_pre5y_cumulative', 'dep_dx_pre5y_cumulative_f', 'dep_dx_pre5y_cumulative_a', 'anx_dx_pre5y_cumulative', 'anx_dx_pre5y_cumulative_f', 'anx_dx_pre5y_cumulative_a', 'bip_dx_pre5y_cumulative', 'bip_dx_pre5y_cumulative_f', 'bip_dx_pre5y_cumulative_a', 'sch_dx_pre5y_cumulative', 'sch_dx_pre5y_cumulative_f', 'sch_dx_pre5y_cumulative_a', 'oth_dx_pre5y_cumulative', 'dem_dx_pre5y_cumulative', 'add_dx_pre5y_cumulative', 'asd_dx_pre5y_cumulative', 'per_dx_pre5y_cumulative', 'alc_dx_pre5y_cumulative', 'dru_dx_pre5y_cumulative', 'pts_dx_pre5y_cumulative', 'eat_dx_pre5y_cumulative', 'tbi_dx_pre5y_cumulative', 'phq8_index_score_calc', 'phq8_index_score_calc_f', 'raceAsian_8', 'raceBlack_8', 'raceHP_8', 'raceIN_8', 'raceMUOT_8', 'raceUN_8', 'hispanic_8', 'age_8', 'q9_0_a', 'q9_1_a', 'q9_2_a', 'q9_3_a', 'q9_0_8', 'q9_1_8', 'q9_2_8', 'q9_3_8', 'q9_0_c', 'q9_1_c', 'q9_2_c', 'q9_3_c', 'any_sui_att_pre5y_cumulative_8', 'any_sui_att_pre5y_cumulative_c', 'any_sui_att_pre5y_cumulative_de', 'any_sui_att_pre5y_cumulative_an', 'any_sui_att_pre5y_cumulative_bi', 'any_sui_att_pre5y_cumulative_sc', 'any_sui_att_pre5y_cumulative_al', 'any_sui_att_pre5y_cumulative_dr', 'any_sui_att_pre5y_cumulative_pe']
# these columns to scale are same as all the numeric (num_cols) columns

# In[23]:
count = 0
for i in non_pc_columns_to_scale:
    if i in non_pc_data.columns:
        count += 1
    else:
        print(i)
print(count)


# In[24]:
non_pc_data[non_pc_columns_to_scale] = scale(non_pc_data[non_pc_columns_to_scale])
# print(non_pc_data)


# In[32]:
missing_columns = non_pc_data.columns[non_pc_data.isnull().any()]
# print(missing_columns)


# # In[33]:
# plt.figure(figsize=(10,6))
# sns.heatmap(non_pc_data[missing_columns].isna().transpose(),
#             cmap="YlGnBu",
#             cbar_kws={'label': 'Missing Data'})
# plt.show()


# In[34]:
for i in missing_columns:
    if sum(non_pc_data[i].isnull()) == non_pc_data.shape[0]: #233585: #257999:
        print(i)
        print(sum(non_pc_data[i].isnull()))
    else:
        print(i)
        print(sum(non_pc_data[i].isnull()))



for i in missing_columns:
    if sum(non_pc_data[i].isnull()) == non_pc_data.shape[0]: #412045: #446893:
        # data[i].fillna(value=0,inplace=True)
        non_pc_data = non_pc_data.drop(columns=[i]) # if all rows are empty for a column, drop that column
        # print(sum(data[i].isnull()))
        print("Empty column:", i)
    else:
        non_pc_data[i].fillna(value=-1, inplace=True) # # if only some of rows have missing value for a column, then fill that with -1, for num or bin cols





# ### Splitting data into train and validation

# In[36]:
train_X, X_test, train_y, y_test = train_test_split(pc_data, y, test_size=0.35, random_state=42, stratify = y)


# In[37]:
X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, test_size=0.35, random_state=42, stratify = train_y)


# In[38]:
train_X_non_pc, X_test_non_pc, train_y_non_pc, y_test_non_pc = train_test_split(non_pc_data, y_non_pc, test_size=0.35, random_state=42, stratify = y_non_pc)


# In[39]:
X_train_non_pc, X_val_non_pc, y_train_non_pc, y_val_non_pc = train_test_split(train_X_non_pc, train_y_non_pc, test_size=0.35, random_state=42, stratify = train_y_non_pc)





print()
print('COLUMNS REMAINING CHECK')
data_cols_remaining = list(X_train.columns)
# print(data_cols_remaining)
print(len(data_cols_remaining))
print()


print()
print()
print('-----------------------------------')
print()
print()



parser = argparse.ArgumentParser(description='SHAP')
# parser.add_argument('--p', '--pretrain', default='no', type=str, help='Pre-train or not')
# parser.add_argument('--d', '--train', default='single', type=str, help='Data to train classifier on: single or both') # 'test', 'train'
parser.add_argument('--m', '--model', default='logs_MH_subset/saved_models/xgboost_entire_data_testPC.pkl', type=str, help='Model to evaluate')
parser.add_argument('--mt', '--model_type', default='ML', type=str, help='ML (pkl) or DL (pth) model to load')
parser.add_argument('--sv', '--save_file', default='xgboost_entire_data', type=str, help='File name for the plot')
parser.add_argument('--td', '--test_domain', default=0, type=int, help='0=PC, 1=Non-PC')
args = parser.parse_args()


# Load Model
model_type = args.mt
save_file = args.sv
model_l = args.m
test_data_idx = args.td


if model_type == 'ML':
    print('Model path:', model_l)
    model = joblib.load(model_l)
elif model_type == 'DL':
    print('Model path:', model_l)
    model = TabNetClassifier()
    model.load_model(model_l)

    # print('Model path:', args.m)
    # model = torch.load(args.m)
else:
    print('Model path not found:', model_type, model_l)



X_train_data_options = [X_train, X_train_non_pc]
y_train_data_options = [y_train, y_train_non_pc]

X_gt = X_train_data_options[test_data_idx]
y_gt = y_train_data_options[test_data_idx]

# X_gt = X_gt[0:5]
# y_gt = y_gt[0:5]



print()
# print('y_gt', y_gt.values)
print('X_gt shape', X_gt.shape)
print('y_gt shape', y_gt.shape)
print()



# SHAP

if model_type == 'ignore': #'ML':
    print('TreeExplainer')
    explainer = shap.TreeExplainer(model)

    plt.figure()
    # shap_values = explainer.shap_values(X_test_pc.values)
    shap_values = explainer.shap_values(X_gt.values)
    shap.summary_plot(shap_values, X_gt.values, feature_names=X_gt.columns, plot_type="bar", max_display=10)
    # shap.plots.bar(shap_values, max_display=20) #, show=False)
    # plt.savefig('logs/figures/xgboost_entire_data_testPC_SHAP_bar.png')
    # plt.savefig('xgboost_entire_data_testPC_SHAP_bar.png')
    # Change the axis titles
    ax = plt.gca()
    ax.set_xlabel('mean(|SHAP Value|)')
    # ax.set_ylabel('Feature Names')
    plt.savefig('logs_MH_subset/figures/' + save_file + '_bar.png', bbox_inches='tight')

    plt.figure()
    shap.summary_plot(shap_values, X_gt.values, feature_names=X_gt.columns, max_display=10)
    # plt.savefig('logs/figures/xgboost_entire_data_testPC_SHAP_summary.png')
    ax.set_xlabel('SHAP Value')
    # ax.set_ylabel('Feature Names')
    plt.savefig('logs_MH_subset/figures/' + save_file + '_summary.png', bbox_inches='tight')
    print('shap done and saved')
    print()

    print('Checking ML model:')
    # Check model predictions
    preds_check = model.predict(X_gt.values)
    print("Model predictions range:", preds_check.min(), preds_check.max())

    # Inspect SHAP values range
    print("SHAP values range for model:", shap_values.values.min(), shap_values.values.max())

# this one runs for SHAP 
elif model_type == 'DL' or model_type == 'ML':
    print('Explainer')
    # explainer = shap.DeepExplainer(model, X_gt.values)
    explainer = shap.Explainer(model.predict, X_gt.values)

    # print("X_gt shape:", X_gt.shape)
    # print("Model expected input dimension:", model.input_dim)

    plt.figure()
    # shap_values = explainer.shap_values(X_test_pc.values)
    shap_values = explainer(X_gt.values)

    # Manually set feature names if missing
    shap_values.feature_names = X_gt.columns.tolist()
    # shap_values = explainer.shap_values(X_gt.values)
    shap.summary_plot(shap_values, X_gt.values, feature_names=X_gt.columns, plot_type="bar", max_display=10)
    # shap.plots.bar(shap_values, max_display=20) #, show=False)
    # plt.savefig('logs/figures/xgboost_entire_data_testPC_SHAP_bar.png')
    # plt.savefig('xgboost_entire_data_testPC_SHAP_bar.png')
    # Change the axis titles
    ax = plt.gca()
    ax.set_xlabel('mean(|SHAP Value|)')
    # ax.set_ylabel('Feature Names')
    plt.savefig('logs_MH_subset/figures/' + save_file + '_bar.png', bbox_inches='tight')

    plt.figure()
    shap.summary_plot(shap_values, X_gt.values, feature_names=X_gt.columns, max_display=10)
    # plt.savefig('logs/figures/xgboost_entire_data_testPC_SHAP_summary.png')
    ax.set_xlabel('SHAP Value')
    # ax.set_ylabel('Feature Names')
    plt.savefig('logs_MH_subset/figures/' + save_file + '_summary.png', bbox_inches='tight')
    print('shap done and saved')
    print()

    print('Checking model:')
    # Check model predictions
    preds_check = model.predict(X_gt.values)
    print("Model predictions range:", preds_check.min(), preds_check.max())

    # Inspect SHAP values range
    print("SHAP values range for model:", shap_values.values.min(), shap_values.values.max())
    print()
    # print(shap_values.values)

    # Save SHAP values
    with open('logs_MH_subset/shap_values/' + save_file +'_shap_values.pkl', 'wb') as f:
        pickle.dump(shap_values, f)


else:
    print('Explainer failed')


print()

end_time = time.time()

print(f"Time taken to generate SHAP values and plots: {end_time - start_time} seconds")



