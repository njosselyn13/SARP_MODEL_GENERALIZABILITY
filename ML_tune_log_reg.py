import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
import pickle
import json
from sklearn.model_selection import ParameterGrid
import parfit.parfit as pf
import optuna
from sklearn.metrics import accuracy_score
from sklearn import linear_model
import joblib

import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
# torch.manual_seed(SEED)


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


# In[3]:
# print(data.columns[316:329])


# In[4]:
# print(sum(data["PRIMARY_CARE_VISIT"] == 1))


# In[5]:
data["event90"] = data["event90"].fillna(value=0)

data = data.drop(columns=["income", "college", "hhld_inc_It40k", "coll_deg_It25p"])

data.rename(columns={'hhld_inc_lt40k_NJ': 'hhld_inc_It40k', 'coll_deg_lt25p_NJ': 'coll_deg_It25p'}, inplace=True)




# In[6]:
# print(sum(data["event90"].isna()))


# In[7]:
#pc_column_list = ["age","ac1","ac3","ac4","ac1f","ac4f","medicaid","commercial","selffunded","highdedectible","days_since_prev","female","dep_dx_pre5y_noi_cumulative","dep_dx_pre5y","anx_dx_pre5y","bip_dx_pre5y","add_dx_pre5y","alc_dx_pre5y","dru_dx_pre5y_noi_cumulative","dru_dx_pre5y","antidep_rx_pre3m","antidep_rx_pre1y_cumulative","benzo_rx_pre3m","hypno_rx_pre3m","hypno_rx_pre5y_cumulative","sga_rx_pre1y_cumulative","sga_rx_pre5y_cumulative","mh_ip_pre3m","mh_ip_pre1y_cumulative","mh_ip_pre5y_cumulative","mh_op_pre3m","mh_op_pre1y_cumulative","mh_op_pre5y_cumulative","mh_ed_pre3m","mh_ed_pre1y_cumulative","mh_ed_pre5y_cumulative","any_sui_att_pre3m","any_sui_att_pre1y_cumulative","any_sui_att_pre5y_cumulative","lvi_sui_att_pre5y","any_inj_poi_pre3m","any_inj_poi_pre1y_cumulative","any_inj_poi_pre5y_cumulative","any_sui_att_pre5y_cumulative_a","charlson_score","charlson_a","charlson_cvd","charlson_diab","raceAsian","raceBlack","hispanic","coll_deg_It25p","phqnumber90","phqnumber183","phqnumber365","phqmode365_2","phqmax365_0","phqmax365_1","phqmax365_2","phqmax365_3","raceBlack_de","raceBlack_an","raceUN_bi","hispanic_bi","dep_dx_pre5y_cumulative","dep_dx_pre5y_cumulative_a","anx_dx_pre5y_cumulative","anx_dx_pre5y_cumulative_a","bip_dx_pre5y_cumulative","bip_dx_pre5y_cumulative_f","bip_dx_pre5y_cumulative_a","sch_dx_pre5y_cumulative_a","oth_dx_pre5y_cumulative","dem_dx_pre5y_cumulative","per_dx_pre5y_cumulative","alc_dx_pre5y_cumulative","dru_dx_pre5y_cumulative","pts_dx_pre5y_cumulative","eat_dx_pre5y_cumulative","tbi_dx_pre5y_cumulative","phq8_index_score_calc_f","raceAsian_8","raceIN_8","hispanic_8","age_8","q9_0_a","q9_1_8","q9_2_8","q9_3_8","q9_1_c","q9_2_c","q9_3_c","q9_0_de","any_sui_att_pre5y_cumulative_8","any_sui_att_pre5y_cumulative_c","any_sui_att_pre5y_cumulative_de","any_sui_att_pre5y_cumulative_an","any_sui_att_pre5y_cumulative_bi","any_sui_att_pre5y_cumulative_sc","any_sui_att_pre5y_cumulative_al","any_sui_att_pre5y_cumulative_dr","any_sui_att_pre5y_cumulative_pe"]


# In[8]:
#len(pc_column_list)


# ## Selecting Primary Care data where Flag(PRIMARY_CARE_VISIT) == 1

# In[9]:
pc_data = data[data["PRIMARY_CARE_VISIT"] == 1]
# print(pc_data)


# In[10]:
y = pc_data["event90"]


# In[11]:
#pc_data = pc_data[pc_column_list]


# In[12]:
# print(pc_data)


# # In[13]:
# pc_columns_to_scale = ["age","days_since_prev","charlson_score","charlson_a","dep_dx_pre5y_cumulative_a","anx_dx_pre5y_cumulative_a","bip_dx_pre5y_cumulative_a","sch_dx_pre5y_cumulative_a","phqnumber90","phqnumber183","phqnumber365","phq8_index_score_calc_f","raceAsian_8","raceIN_8","hispanic_8","age_8","q9_0_a","q9_1_8","q9_2_8","q9_3_8","q9_1_c","q9_2_c","q9_3_c","any_sui_att_pre5y_cumulative_a","any_sui_att_pre5y_cumulative_8","any_sui_att_pre5y_cumulative_c"]
#
#
# # In[14]:
# count = 0
# for i in pc_columns_to_scale:
#     if i in pc_data.columns:
#         count += 1
#     else:
#         print(i)
# print(count)
#
#
# # In[15]:
# pc_data[pc_columns_to_scale] = scale(pc_data[pc_columns_to_scale])


# In[16]:
# print(pc_data)


# In[17]:
# print(pc_data)


# In[18]:
pc_data = pc_data.drop(columns=["PRIMARY_CARE_VISIT","person_id","event30","event90","death30","death90","visit_mh"])
# print(pc_data)


# In[19]:
# print(pc_data.columns)


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

# In[20]:
missing_columns = pc_data.columns[pc_data.isnull().any()]


# In[21]:
# print(missing_columns)


# # In[22]:
# for i in missing_columns:
#     if sum(pc_data[i].isnull()) == 178460: #188894:
#         pc_data[i].fillna(value=0,inplace=True)
#         print(sum(pc_data[i].isnull()))
#     else:
#         mean_value=pc_data[i].mean()
#         pc_data[i].fillna(value=mean_value,inplace=True)
#         print(sum(pc_data[i].isnull()))



bin_cols = ['event30', 'event90', 'death30', 'death90', 'visit_mh', 'ac1', 'ac2', 'ac3', 'ac4', 'ac5', 'ac1f', 'ac3f', 'ac4f', 'ac5f', 'Enrolled', 'medicaid', 'commercial', 'privatepay', 'statesubsidized', 'selffunded', 'medicare', 'highdedectible', 'other', 'first_visit', 'female', 'dep_dx_pre5y', 'anx_dx_pre5y', 'bip_dx_pre5y', 'sch_dx_pre5y', 'oth_dx_pre5y', 'dem_dx_pre5y', 'add_dx_pre5y', 'asd_dx_pre5y', 'per_dx_pre5y', 'alc_dx_pre5y', 'pts_dx_pre5y', 'eat_dx_pre5y', 'tbi_dx_pre5y', 'dru_dx_pre5y', 'antidep_rx_pre3m', 'benzo_rx_pre3m', 'hypno_rx_pre3m', 'sga_rx_pre3m', 'mh_ip_pre3m', 'mh_op_pre3m', 'mh_ed_pre3m', 'any_sui_att_pre3m', 'lvi_sui_att_pre3m', 'ovi_sui_att_pre3m', 'any_inj_poi_pre3m', 'current_pregnancy', 'del_pre_1_90', 'del_pre_1_180', 'del_pre_1_365', 'charlson_mi', 'charlson_chd', 'charlson_pvd', 'charlson_cvd', 'charlson_dem', 'charlson_cpd', 'charlson_rhd', 'charlson_pud', 'charlson_mlivd', 'charlson_diab', 'charlson_diabc', 'charlson_plegia', 'charlson_ren', 'charlson_malign', 'charlson_slivd', 'charlson_mst', 'charlson_aids', 'raceAsian_asa', 'raceBlack_asa', 'raceHP_asa', 'raceIN_asa', 'raceMUOT_asa', 'raceUN_asa', 'hispanic_asa', 'raceAsian', 'raceBlack', 'raceHP', 'raceIN', 'raceMUOT', 'raceUN', 'raceWH', 'hispanic', 'raceAsian_f', 'raceBlack_f', 'raceHP_f', 'raceIN_f', 'raceMUOT_f', 'raceUN_f', 'hispanic_f', 'census_missing', 'hhld_inc_It40k', 'coll_deg_It25p', 'phqmode90_0', 'phqmode90_1', 'phqmode90_2', 'phqmax90_0', 'phqmax90_1', 'phqmax90_2', 'phqmax90_3', 'phqmode183_0', 'phqmode183_1', 'phqmode183_2', 'phqmax183_0', 'phqmax183_1', 'phqmax183_2', 'phqmax183_3', 'phqmode365_0', 'phqmode365_1', 'phqmode365_2', 'phqmax365_0', 'phqmax365_1', 'phqmax365_2', 'phqmax365_3', 'raceAsian_de', 'raceBlack_de', 'raceHP_de', 'raceIN_de', 'raceMUOT_de', 'raceUN_de', 'hispanic_de', 'raceAsian_an', 'raceBlack_an', 'raceHP_an', 'raceIN_an', 'raceMUOT_an', 'raceUN_an', 'hispanic_an', 'raceAsian_bi', 'raceBlack_bi', 'raceHP_bi', 'raceIN_bi', 'raceMUOT_bi', 'raceUN_bi', 'hispanic_bi', 'raceAsian_sc', 'raceBlack_sc', 'raceHP_sc', 'raceIN_sc', 'raceMUOT_sc', 'raceUN_sc', 'hispanic_sc', 'phq8_missing', 'phq8_missing_f', 'q9_0', 'q9_1', 'q9_2', 'q9_3', 'q9_0_f', 'q9_1_f', 'q9_2_f', 'q9_3_f', 'raceAsian_q90', 'raceBlack_q90', 'raceHP_q90', 'raceIN_q90', 'raceMUOT_q90', 'raceUN_q90', 'hispanic_q90', 'raceAsian_q91', 'raceBlack_q91', 'raceHP_q91', 'raceIN_q91', 'raceMUOT_q91', 'raceUN_q91', 'hispanic_q91', 'raceAsian_q92', 'raceBlack_q92', 'raceHP_q92', 'raceIN_q92', 'raceMUOT_q92', 'raceUN_q92', 'hispanic_q92', 'raceAsian_q93', 'raceBlack_q93', 'raceHP_q93', 'raceIN_q93', 'raceMUOT_q93', 'raceUN_q93', 'hispanic_q93', 'q9_0_de', 'q9_1_de', 'q9_2_de', 'q9_3_de', 'q9_0_an', 'q9_1_an', 'q9_2_an', 'q9_3_an', 'q9_0_bi', 'q9_1_bi', 'q9_2_bi', 'q9_3_bi', 'q9_0_sc', 'q9_1_sc', 'q9_2_sc', 'q9_3_sc', 'q9_0_al', 'q9_1_al', 'q9_2_al', 'q9_3_al', 'q9_0_dr', 'q9_1_dr', 'q9_2_dr', 'q9_3_dr', 'q9_0_pe', 'q9_1_pe', 'q9_2_pe', 'q9_3_pe', 'phqMax90_0_q90', 'phqMax90_1_q90', 'phqMax90_2_q90', 'phqMax90_3_q90', 'phqMax90_0_q91', 'phqMax90_1_q91', 'phqMax90_2_q91', 'phqMax90_3_q91', 'phqMax90_0_q92', 'phqMax90_1_q92', 'phqMax90_2_q92', 'phqMax90_3_q92', 'phqMax90_0_q93', 'phqMax90_1_q93', 'phqMax90_2_q93', 'phqMax90_3_q93']

num_cols = ['age', 'days_since_prev', 'dep_dx_pre5y_noi_cumulative', 'anx_dx_pre5y_noi_cumulative', 'bip_dx_pre5y_noi_cumulative', 'sch_dx_pre5y_noi_cumulative', 'oth_dx_pre5y_noi_cumulative', 'dem_dx_pre5y_noi_cumulative', 'add_dx_pre5y_noi_cumulative', 'asd_dx_pre5y_noi_cumulative', 'per_dx_pre5y_noi_cumulative', 'alc_dx_pre5y_noi_cumulative', 'pts_dx_pre5y_noi_cumulative', 'eat_dx_pre5y_noi_cumulative', 'tbi_dx_pre5y_noi_cumulative', 'dru_dx_pre5y_noi_cumulative', 'antidep_rx_pre1y_cumulative', 'antidep_rx_pre5y_cumulative', 'benzo_rx_pre1y_cumulative', 'benzo_rx_pre5y_cumulative', 'hypno_rx_pre1y_cumulative', 'hypno_rx_pre5y_cumulative', 'sga_rx_pre1y_cumulative', 'sga_rx_pre5y_cumulative', 'mh_ip_pre1y_cumulative', 'mh_ip_pre5y_cumulative', 'mh_op_pre1y_cumulative', 'mh_op_pre5y_cumulative', 'mh_ed_pre1y_cumulative', 'mh_ed_pre5y_cumulative', 'any_sui_att_pre1y_cumulative', 'any_sui_att_pre5y_cumulative', 'lvi_sui_att_pre1y_cumulative', 'lvi_sui_att_pre5y_cumulative', 'ovi_sui_att_pre1y_cumulative', 'ovi_sui_att_pre5y_cumulative', 'any_inj_poi_pre1y_cumulative', 'any_inj_poi_pre5y_cumulative', 'any_sui_att_pre5y_cumulative_f', 'any_sui_att_pre5y_cumulative_a', 'charlson_score', 'charlson_a', 'phqnumber90', 'phqnumber183', 'phqnumber365', 'dep_dx_pre5y_cumulative', 'dep_dx_pre5y_cumulative_f', 'dep_dx_pre5y_cumulative_a', 'anx_dx_pre5y_cumulative', 'anx_dx_pre5y_cumulative_f', 'anx_dx_pre5y_cumulative_a', 'bip_dx_pre5y_cumulative', 'bip_dx_pre5y_cumulative_f', 'bip_dx_pre5y_cumulative_a', 'sch_dx_pre5y_cumulative', 'sch_dx_pre5y_cumulative_f', 'sch_dx_pre5y_cumulative_a', 'oth_dx_pre5y_cumulative', 'dem_dx_pre5y_cumulative', 'add_dx_pre5y_cumulative', 'asd_dx_pre5y_cumulative', 'per_dx_pre5y_cumulative', 'alc_dx_pre5y_cumulative', 'dru_dx_pre5y_cumulative', 'pts_dx_pre5y_cumulative', 'eat_dx_pre5y_cumulative', 'tbi_dx_pre5y_cumulative', 'phq8_index_score_calc', 'phq8_index_score_calc_f', 'raceAsian_8', 'raceBlack_8', 'raceHP_8', 'raceIN_8', 'raceMUOT_8', 'raceUN_8', 'hispanic_8', 'age_8', 'q9_0_a', 'q9_1_a', 'q9_2_a', 'q9_3_a', 'q9_0_8', 'q9_1_8', 'q9_2_8', 'q9_3_8', 'q9_0_c', 'q9_1_c', 'q9_2_c', 'q9_3_c', 'any_sui_att_pre5y_cumulative_8', 'any_sui_att_pre5y_cumulative_c', 'any_sui_att_pre5y_cumulative_de', 'any_sui_att_pre5y_cumulative_an', 'any_sui_att_pre5y_cumulative_bi', 'any_sui_att_pre5y_cumulative_sc', 'any_sui_att_pre5y_cumulative_al', 'any_sui_att_pre5y_cumulative_dr', 'any_sui_att_pre5y_cumulative_pe']

# for i in missing_columns:
#     if sum(pc_data[i].isnull()) == pc_data.shape[0]: #446893:
#         # data[i].fillna(value=0,inplace=True)
#         pc_data = pc_data.drop(columns=[i])
#         # print(sum(data[i].isnull()))
#     else:
#         if i in bin_cols:
#             pc_data[i].fillna(value=-1, inplace=True)
#         elif i in num_cols:
#             mean_value=pc_data[i].mean()
#             pc_data[i].fillna(value=mean_value,inplace=True)
#         else:
#             print('Column not binary or numeric', i)
#         # print(sum(data[i].isnull()))


for i in missing_columns:
    if sum(pc_data[i].isnull()) == pc_data.shape[0]: #412045: #446893:
        # data[i].fillna(value=0,inplace=True)
        pc_data = pc_data.drop(columns=[i]) # if all rows are empty for a column, drop that column
        # print(sum(data[i].isnull()))
        print("Empty column:", i)
    else:
        pc_data[i].fillna(value=-1, inplace=True) # # if only some of rows have missing value for a column, then fill that with -1, for num or bin cols





# # In[23]:
# plt.figure(figsize=(10,6))
# sns.heatmap(pc_data.isna().transpose(),
#             cmap="YlGnBu",
#             cbar_kws={'label': 'Missing Data'})
# plt.show()



# ### Splitting data into train and validation

# In[24]:
train_X, X_test, train_y, y_test = train_test_split(pc_data, y, test_size=0.35, random_state=42, stratify = y)


# In[25]:
X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, test_size=0.35, random_state=42, stratify = train_y)

# In[26]:
# print(train_X)


# ### Creating dataframe for storing coefficients

# In[23]:
storing_coef_clf = pd.DataFrame(columns=X_train.columns)

# # Hyperparameter tuning Logistic Regression model using Optuna

print('Hyperparameter tuning Logistic Regression model using Optuna')

# In[32]:
def objective(trial):
    penalty = trial.suggest_categorical("penalty", ["l1"])
    tol = trial.suggest_float("tol", 0.0001, 0.01, log=True)
    # C = trial.suggest_float("C", 1.0, 10.0, log=True)
    # C = np.arange(0.0001, 0.1, 0.0005)
    C = trial.suggest_float("C", 0.1, 10, step=0.05)
    intercept = trial.suggest_categorical("fit_intercept", [True, False])
    solver = trial.suggest_categorical("solver", ["liblinear", "saga"])

    ## Create Model
    clf = LogisticRegression(penalty=penalty,
                             tol=tol,
                             C=C,
                             fit_intercept=intercept,
                             solver=solver,
                             multi_class="auto",random_state=42
                             )
    clf.fit(X_train, y_train)
    preds_valid = clf.predict_proba(X_val.values)
    valid_auc = roc_auc_score(y_score=preds_valid[:, 1], y_true=y_val)

    return valid_auc


# In[33]:
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




# In[34]:
# LogisticRegression(**study.best_params)
print()
print("Logistic Regression best params") # balanced:")
# In[40]:
clf = LogisticRegression(**study.best_params) #, class_weight="balanced")
clf.fit(X_train, y_train)
preds_val = clf.predict(X_val.values)
preds = clf.predict(X_test.values)

acc_val = accuracy_score(y_val, preds_val) * 100
print("Validation Accuracy for Optuna Tuned Model: ", acc_val)
preds_val = clf.predict_proba(X_val.values)
val_auc = roc_auc_score(y_score=preds_val[:, 1], y_true=y_val)
print("Validation AUC for Optuna Tuned Model: ", val_auc)

print("Validation ROC AUC is: ", roc_auc_score(y_score=preds_val[:, 1], y_true=y_val))
print("Validation F1 score is: ", metrics.f1_score(y_true=y_val, y_pred=clf.predict(X_val.values)))
print("Validation Precision score is: ", metrics.precision_score(y_true=y_val, y_pred=clf.predict(X_val.values)))
print("Validation Recall score is: ", metrics.recall_score(y_true=y_val, y_pred=clf.predict(X_val.values)))
print(metrics.classification_report(y_true=y_val, y_pred=clf.predict(X_val.values)))
print(metrics.confusion_matrix(y_true=y_val, y_pred=clf.predict(X_val.values)))

acc = accuracy_score(y_test, preds) * 100
print("Test Accuracy for Optuna Tuned Model: ", acc)
preds_test = clf.predict_proba(X_test.values)
test_auc = roc_auc_score(y_score=preds_test[:, 1], y_true=y_test)
print("Test AUC for Optuna Tuned Model: ", test_auc)

print("ROC AUC is: ", metrics.roc_auc_score(y_test, preds_test[:, 1]))
print("F1 score is: ", metrics.f1_score(y_true=y_test, y_pred=preds))
print("Preicision score is: ", metrics.precision_score(y_true=y_test, y_pred=preds))
print("Recall score is: ", metrics.recall_score(y_true=y_test, y_pred=preds))
print(metrics.classification_report(y_true=y_test, y_pred=preds))
print(metrics.confusion_matrix(y_true=y_test, y_pred=preds))
tn, fp, fn, tp = metrics.confusion_matrix(y_true=y_test, y_pred=preds).ravel()
specificity = tn / (tn + fp)
print("Specificity score is: ", specificity)
sensitivity = tp / (tp + fn)
print("Sensitivity score is: ", sensitivity)
print()
print()



joblib.dump(clf, 'logs_MH_subset/saved_models/log_reg_320scratch_not_balanced.pkl')
print("model saved")
print()


print()
print("Logistic Regression best params BALANCED:")
# In[40]:
clf = LogisticRegression(**study.best_params, class_weight="balanced")
clf.fit(X_train, y_train)
preds_val = clf.predict(X_val.values)
preds = clf.predict(X_test.values)

acc_val = accuracy_score(y_val, preds_val) * 100
print("Validation Accuracy for Optuna Tuned Model: ", acc_val)
preds_val = clf.predict_proba(X_val.values)
val_auc = roc_auc_score(y_score=preds_val[:, 1], y_true=y_val)
print("Validation AUC for Optuna Tuned Model: ", val_auc)

print("Validation ROC AUC is: ", roc_auc_score(y_score=preds_val[:, 1], y_true=y_val))
print("Validation F1 score is: ", metrics.f1_score(y_true=y_val, y_pred=clf.predict(X_val.values)))
print("Validation Precision score is: ", metrics.precision_score(y_true=y_val, y_pred=clf.predict(X_val.values)))
print("Validation Recall score is: ", metrics.recall_score(y_true=y_val, y_pred=clf.predict(X_val.values)))
print(metrics.classification_report(y_true=y_val, y_pred=clf.predict(X_val.values)))
print(metrics.confusion_matrix(y_true=y_val, y_pred=clf.predict(X_val.values)))

acc = accuracy_score(y_test, preds) * 100
print("Test Accuracy for Optuna Tuned Model: ", acc)
preds_test = clf.predict_proba(X_test.values)
test_auc = roc_auc_score(y_score=preds_test[:, 1], y_true=y_test)
print("Test AUC for Optuna Tuned Model: ", test_auc)

print("ROC AUC is: ", metrics.roc_auc_score(y_test, preds_test[:, 1]))
print("F1 score is: ", metrics.f1_score(y_true=y_test, y_pred=preds))
print("Preicision score is: ", metrics.precision_score(y_true=y_test, y_pred=preds))
print("Recall score is: ", metrics.recall_score(y_true=y_test, y_pred=preds))
print(metrics.classification_report(y_true=y_test, y_pred=preds))
print(metrics.confusion_matrix(y_true=y_test, y_pred=preds))
tn, fp, fn, tp = metrics.confusion_matrix(y_true=y_test, y_pred=preds).ravel()
specificity = tn / (tn + fp)
print("Specificity score is: ", specificity)
sensitivity = tp / (tp + fn)
print("Sensitivity score is: ", sensitivity)
print()
print()


joblib.dump(clf, 'logs_MH_subset/saved_models/log_reg_320scratch_balanced.pkl')
print("model saved")
print()




