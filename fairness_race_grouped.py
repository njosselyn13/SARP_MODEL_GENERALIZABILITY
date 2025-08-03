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
import random


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


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

data["event90"] = data["event90"].fillna(value=0)

y = data["event90"]

data = data.drop(columns=["person_id","event30","death30","death90","visit_mh"])
# print(data)

data = data.drop(columns=["income", "college", "hhld_inc_It40k", "coll_deg_It25p"])

# fix dropped cols above 
data.rename(columns={'hhld_inc_lt40k_NJ': 'hhld_inc_It40k', 'coll_deg_lt25p_NJ': 'coll_deg_It25p'}, inplace=True)


# ## Scaling data

columns_to_scale = ["age","days_since_prev","charlson_score","charlson_a","dep_dx_pre5y_cumulative_a","anx_dx_pre5y_cumulative_a","bip_dx_pre5y_cumulative_a","sch_dx_pre5y_cumulative_a","phqnumber90","phqnumber183","phqnumber365","phq8_index_score_calc_f","raceAsian_8","raceIN_8","hispanic_8","age_8","q9_0_a","q9_1_8","q9_2_8","q9_3_8","q9_1_c","q9_2_c","q9_3_c","any_sui_att_pre5y_cumulative_a","any_sui_att_pre5y_cumulative_8","any_sui_att_pre5y_cumulative_c"]

# count = 0
# for i in columns_to_scale:
#     if i in data.columns:
#         count += 1
#     else:
#         print(i)
# print(count)


data[columns_to_scale] = scale(data[columns_to_scale])

# data = data.drop(columns=["Unnamed: 0","person_id","event30","death30","death90","visit_mh"])


# ### Dealing with missing values

missing_columns = data.columns[data.isnull().any()]


bin_cols = ['event30', 'event90', 'death30', 'death90', 'visit_mh', 'ac1', 'ac2', 'ac3', 'ac4', 'ac5', 'ac1f', 'ac3f', 'ac4f', 'ac5f', 'Enrolled', 'medicaid', 'commercial', 'privatepay', 'statesubsidized', 'selffunded', 'medicare', 'highdedectible', 'other', 'first_visit', 'female', 'dep_dx_pre5y', 'anx_dx_pre5y', 'bip_dx_pre5y', 'sch_dx_pre5y', 'oth_dx_pre5y', 'dem_dx_pre5y', 'add_dx_pre5y', 'asd_dx_pre5y', 'per_dx_pre5y', 'alc_dx_pre5y', 'pts_dx_pre5y', 'eat_dx_pre5y', 'tbi_dx_pre5y', 'dru_dx_pre5y', 'antidep_rx_pre3m', 'benzo_rx_pre3m', 'hypno_rx_pre3m', 'sga_rx_pre3m', 'mh_ip_pre3m', 'mh_op_pre3m', 'mh_ed_pre3m', 'any_sui_att_pre3m', 'lvi_sui_att_pre3m', 'ovi_sui_att_pre3m', 'any_inj_poi_pre3m', 'current_pregnancy', 'del_pre_1_90', 'del_pre_1_180', 'del_pre_1_365', 'charlson_mi', 'charlson_chd', 'charlson_pvd', 'charlson_cvd', 'charlson_dem', 'charlson_cpd', 'charlson_rhd', 'charlson_pud', 'charlson_mlivd', 'charlson_diab', 'charlson_diabc', 'charlson_plegia', 'charlson_ren', 'charlson_malign', 'charlson_slivd', 'charlson_mst', 'charlson_aids', 'raceAsian_asa', 'raceBlack_asa', 'raceHP_asa', 'raceIN_asa', 'raceMUOT_asa', 'raceUN_asa', 'hispanic_asa', 'raceAsian', 'raceBlack', 'raceHP', 'raceIN', 'raceMUOT', 'raceUN', 'raceWH', 'hispanic', 'raceAsian_f', 'raceBlack_f', 'raceHP_f', 'raceIN_f', 'raceMUOT_f', 'raceUN_f', 'hispanic_f', 'census_missing', 'hhld_inc_It40k', 'coll_deg_It25p', 'phqmode90_0', 'phqmode90_1', 'phqmode90_2', 'phqmax90_0', 'phqmax90_1', 'phqmax90_2', 'phqmax90_3', 'phqmode183_0', 'phqmode183_1', 'phqmode183_2', 'phqmax183_0', 'phqmax183_1', 'phqmax183_2', 'phqmax183_3', 'phqmode365_0', 'phqmode365_1', 'phqmode365_2', 'phqmax365_0', 'phqmax365_1', 'phqmax365_2', 'phqmax365_3', 'raceAsian_de', 'raceBlack_de', 'raceHP_de', 'raceIN_de', 'raceMUOT_de', 'raceUN_de', 'hispanic_de', 'raceAsian_an', 'raceBlack_an', 'raceHP_an', 'raceIN_an', 'raceMUOT_an', 'raceUN_an', 'hispanic_an', 'raceAsian_bi', 'raceBlack_bi', 'raceHP_bi', 'raceIN_bi', 'raceMUOT_bi', 'raceUN_bi', 'hispanic_bi', 'raceAsian_sc', 'raceBlack_sc', 'raceHP_sc', 'raceIN_sc', 'raceMUOT_sc', 'raceUN_sc', 'hispanic_sc', 'phq8_missing', 'phq8_missing_f', 'q9_0', 'q9_1', 'q9_2', 'q9_3', 'q9_0_f', 'q9_1_f', 'q9_2_f', 'q9_3_f', 'raceAsian_q90', 'raceBlack_q90', 'raceHP_q90', 'raceIN_q90', 'raceMUOT_q90', 'raceUN_q90', 'hispanic_q90', 'raceAsian_q91', 'raceBlack_q91', 'raceHP_q91', 'raceIN_q91', 'raceMUOT_q91', 'raceUN_q91', 'hispanic_q91', 'raceAsian_q92', 'raceBlack_q92', 'raceHP_q92', 'raceIN_q92', 'raceMUOT_q92', 'raceUN_q92', 'hispanic_q92', 'raceAsian_q93', 'raceBlack_q93', 'raceHP_q93', 'raceIN_q93', 'raceMUOT_q93', 'raceUN_q93', 'hispanic_q93', 'q9_0_de', 'q9_1_de', 'q9_2_de', 'q9_3_de', 'q9_0_an', 'q9_1_an', 'q9_2_an', 'q9_3_an', 'q9_0_bi', 'q9_1_bi', 'q9_2_bi', 'q9_3_bi', 'q9_0_sc', 'q9_1_sc', 'q9_2_sc', 'q9_3_sc', 'q9_0_al', 'q9_1_al', 'q9_2_al', 'q9_3_al', 'q9_0_dr', 'q9_1_dr', 'q9_2_dr', 'q9_3_dr', 'q9_0_pe', 'q9_1_pe', 'q9_2_pe', 'q9_3_pe', 'phqMax90_0_q90', 'phqMax90_1_q90', 'phqMax90_2_q90', 'phqMax90_3_q90', 'phqMax90_0_q91', 'phqMax90_1_q91', 'phqMax90_2_q91', 'phqMax90_3_q91', 'phqMax90_0_q92', 'phqMax90_1_q92', 'phqMax90_2_q92', 'phqMax90_3_q92', 'phqMax90_0_q93', 'phqMax90_1_q93', 'phqMax90_2_q93', 'phqMax90_3_q93']

num_cols = ['age', 'days_since_prev', 'dep_dx_pre5y_noi_cumulative', 'anx_dx_pre5y_noi_cumulative', 'bip_dx_pre5y_noi_cumulative', 'sch_dx_pre5y_noi_cumulative', 'oth_dx_pre5y_noi_cumulative', 'dem_dx_pre5y_noi_cumulative', 'add_dx_pre5y_noi_cumulative', 'asd_dx_pre5y_noi_cumulative', 'per_dx_pre5y_noi_cumulative', 'alc_dx_pre5y_noi_cumulative', 'pts_dx_pre5y_noi_cumulative', 'eat_dx_pre5y_noi_cumulative', 'tbi_dx_pre5y_noi_cumulative', 'dru_dx_pre5y_noi_cumulative', 'antidep_rx_pre1y_cumulative', 'antidep_rx_pre5y_cumulative', 'benzo_rx_pre1y_cumulative', 'benzo_rx_pre5y_cumulative', 'hypno_rx_pre1y_cumulative', 'hypno_rx_pre5y_cumulative', 'sga_rx_pre1y_cumulative', 'sga_rx_pre5y_cumulative', 'mh_ip_pre1y_cumulative', 'mh_ip_pre5y_cumulative', 'mh_op_pre1y_cumulative', 'mh_op_pre5y_cumulative', 'mh_ed_pre1y_cumulative', 'mh_ed_pre5y_cumulative', 'any_sui_att_pre1y_cumulative', 'any_sui_att_pre5y_cumulative', 'lvi_sui_att_pre1y_cumulative', 'lvi_sui_att_pre5y_cumulative', 'ovi_sui_att_pre1y_cumulative', 'ovi_sui_att_pre5y_cumulative', 'any_inj_poi_pre1y_cumulative', 'any_inj_poi_pre5y_cumulative', 'any_sui_att_pre5y_cumulative_f', 'any_sui_att_pre5y_cumulative_a', 'charlson_score', 'charlson_a', 'phqnumber90', 'phqnumber183', 'phqnumber365', 'dep_dx_pre5y_cumulative', 'dep_dx_pre5y_cumulative_f', 'dep_dx_pre5y_cumulative_a', 'anx_dx_pre5y_cumulative', 'anx_dx_pre5y_cumulative_f', 'anx_dx_pre5y_cumulative_a', 'bip_dx_pre5y_cumulative', 'bip_dx_pre5y_cumulative_f', 'bip_dx_pre5y_cumulative_a', 'sch_dx_pre5y_cumulative', 'sch_dx_pre5y_cumulative_f', 'sch_dx_pre5y_cumulative_a', 'oth_dx_pre5y_cumulative', 'dem_dx_pre5y_cumulative', 'add_dx_pre5y_cumulative', 'asd_dx_pre5y_cumulative', 'per_dx_pre5y_cumulative', 'alc_dx_pre5y_cumulative', 'dru_dx_pre5y_cumulative', 'pts_dx_pre5y_cumulative', 'eat_dx_pre5y_cumulative', 'tbi_dx_pre5y_cumulative', 'phq8_index_score_calc', 'phq8_index_score_calc_f', 'raceAsian_8', 'raceBlack_8', 'raceHP_8', 'raceIN_8', 'raceMUOT_8', 'raceUN_8', 'hispanic_8', 'age_8', 'q9_0_a', 'q9_1_a', 'q9_2_a', 'q9_3_a', 'q9_0_8', 'q9_1_8', 'q9_2_8', 'q9_3_8', 'q9_0_c', 'q9_1_c', 'q9_2_c', 'q9_3_c', 'any_sui_att_pre5y_cumulative_8', 'any_sui_att_pre5y_cumulative_c', 'any_sui_att_pre5y_cumulative_de', 'any_sui_att_pre5y_cumulative_an', 'any_sui_att_pre5y_cumulative_bi', 'any_sui_att_pre5y_cumulative_sc', 'any_sui_att_pre5y_cumulative_al', 'any_sui_att_pre5y_cumulative_dr', 'any_sui_att_pre5y_cumulative_pe']


for i in missing_columns:
    if sum(data[i].isnull()) == data.shape[0]: #412045: #446893:
        # data[i].fillna(value=0,inplace=True)
        data = data.drop(columns=[i]) # if all rows are empty for a column, drop that column
        # print(sum(data[i].isnull()))
        print("Empty column:", i)
    else:
        data[i].fillna(value=-1, inplace=True) # # if only some of rows have missing value for a column, then fill that with -1, for num or bin cols



# ## Seperating Primary Care and Non Primary Care data

pc_data = data[data["PRIMARY_CARE_VISIT"] == 1]

non_pc_data = data[data["PRIMARY_CARE_VISIT"] == 0]


#########################################################################################


y_pc1 = pc_data["event90"]

y_non_pc1 = non_pc_data["event90"]

non_pc_data = non_pc_data.drop(columns=["PRIMARY_CARE_VISIT"]) #,"event90"])

pc_data = pc_data.drop(columns=["PRIMARY_CARE_VISIT"]) #,"event90"])


# ### Splitting data into train and validation

# In[29]:
train_X_pc, X_test_pc, train_y_pc, y_test_pc = train_test_split(pc_data, y_pc1, test_size=0.35, random_state=42, stratify = y_pc1)


# In[30]:
X_train_pc, X_val_pc, y_train_pc, y_val_pc = train_test_split(train_X_pc, train_y_pc, test_size=0.35, random_state=42, stratify = train_y_pc)


# In[31]:
train_X_non_pc, X_test_non_pc, train_y_non_pc, y_test_non_pc = train_test_split(non_pc_data, y_non_pc1, test_size=0.35, random_state=42, stratify = y_non_pc1)


# In[32]:
X_train_non_pc, X_val_non_pc, y_train_non_pc, y_val_non_pc = train_test_split(train_X_non_pc, train_y_non_pc, test_size=0.35, random_state=42, stratify = train_y_non_pc)



# Just running on TEST data
###
pc_data = X_test_pc
non_pc_data = X_test_non_pc
###


print()
print('--------------------------------------------------------------------------------------')
print()

#########################################################################################

# RACE
print('RACE')

# race_names = ['raceAsian', 'raceBlack', 'raceHP', 'raceIN', 'raceMUOT', 'raceUN', 'raceWH', 'hispanic']
race_names = ['other', 'raceBlack', 'raceWH']

# ALL DATA
asian = data['raceAsian'].dropna().tolist()
black = data['raceBlack'].dropna().tolist()
hp = data['raceHP'].dropna().tolist()
native = data['raceIN'].dropna().tolist()
muot = data['raceMUOT'].dropna().tolist()
unknown = data['raceUN'].dropna().tolist()
white = data['raceWH'].dropna().tolist()
hispanic = data['hispanic'].dropna().tolist()

# PC DATA
asian_pc = pc_data['raceAsian'].dropna().tolist()
black_pc = pc_data['raceBlack'].dropna().tolist()
hp_pc = pc_data['raceHP'].dropna().tolist()
native_pc = pc_data['raceIN'].dropna().tolist()
muot_pc = pc_data['raceMUOT'].dropna().tolist()
unknown_pc = pc_data['raceUN'].dropna().tolist()
white_pc = pc_data['raceWH'].dropna().tolist()
hispanic_pc = pc_data['hispanic'].dropna().tolist()

# NON-PC DATA
asian_nonpc = non_pc_data['raceAsian'].dropna().tolist()
black_nonpc = non_pc_data['raceBlack'].dropna().tolist()
hp_nonpc = non_pc_data['raceHP'].dropna().tolist()
native_nonpc = non_pc_data['raceIN'].dropna().tolist()
muot_nonpc = non_pc_data['raceMUOT'].dropna().tolist()
unknown_nonpc = non_pc_data['raceUN'].dropna().tolist()
white_nonpc = non_pc_data['raceWH'].dropna().tolist()
hispanic_nonpc = non_pc_data['hispanic'].dropna().tolist()



# GROUPING RACES -- WHITE, BLACK, OTHER (asian, hp, native, unknown) -- there are no muot nor hispanic to combine
combined_list = []
combined_list.extend(asian)
combined_list.extend(hp)
combined_list.extend(native)
# combined_list.extend(muot)
combined_list.extend(unknown)
# combined_list.extend(hispanic)

combined_list_pc = []
combined_list_pc.extend(asian_pc)
combined_list_pc.extend(hp_pc)
combined_list_pc.extend(native_pc)
# combined_list_pc.extend(muot_pc)
combined_list_pc.extend(unknown_pc)
# combined_list_pc.extend(hispanic_pc)


combined_list_nonpc = []
combined_list_nonpc.extend(asian_nonpc)
combined_list_nonpc.extend(hp_nonpc)
combined_list_nonpc.extend(native_nonpc)
# combined_list_nonpc.extend(muot_nonpc)
combined_list_nonpc.extend(unknown_nonpc)
# combined_list_nonpc.extend(hispanic_nonpc)

races = [combined_list, black, white]
races_pc = [combined_list_pc, black_pc, white_pc]
races_nonpc = [combined_list_nonpc, black_nonpc, white_nonpc]



# races = [asian, black, hp, native, muot, unknown, white, hispanic]
# races_pc = [asian_pc, black_pc, hp_pc, native_pc, muot_pc, unknown_pc, white_pc, hispanic_pc]
# races_nonpc = [asian_nonpc, black_nonpc, hp_nonpc, native_nonpc, muot_nonpc, unknown_nonpc, white_nonpc, hispanic_nonpc]

race_domains = [races, races_pc, races_nonpc]
names = ['All', 'PC', 'Non-PC']

for d in range(0, len(race_domains)):
    domain = race_domains[d]
    print(names[d])
    # print(domain)
    counts_1 = []
    counts_0 = []
    for race in domain:
        cnt_1 = 0
        cnt_0 = 0
        for p in race:
            if p == 1:
                cnt_1 = cnt_1 + 1
            elif p == 0:
                cnt_0 = cnt_0 + 1
        counts_1.append(cnt_1)
        counts_0.append(cnt_0)

    # print()
    print(counts_1)
    print(counts_0)
    print()

print()


print('PC:')
for rc in range(0, len(race_names)):
    rn = race_names[rc]
    print(rn)
    if rn == 'raceBlack' or rn == 'raceWH':
        pc_race_attempt = pc_data[(pc_data['event90'] == 1) & (pc_data[rn] == 1)]
        pc_race_NO_attempt = pc_data[(pc_data['event90'] == 0) & (pc_data[rn] == 1)]
    else:
        # race_names = ['raceAsian', 'raceHP', 'raceIN', 'raceUN']   NOT 'raceMUOT', 'hispanic', 'raceWH', 'raceBlack'
        pc_race_attempt_asian = pc_data[(pc_data['event90'] == 1) & (pc_data['raceAsian'] == 1)]
        pc_race_NO_attempt_asian = pc_data[(pc_data['event90'] == 0) & (pc_data['raceAsian'] == 1)]

        pc_race_attempt_hp = pc_data[(pc_data['event90'] == 1) & (pc_data['raceHP'] == 1)]
        pc_race_NO_attempt_hp = pc_data[(pc_data['event90'] == 0) & (pc_data['raceHP'] == 1)]

        pc_race_attempt_in = pc_data[(pc_data['event90'] == 1) & (pc_data['raceIN'] == 1)]
        pc_race_NO_attempt_in = pc_data[(pc_data['event90'] == 0) & (pc_data['raceIN'] == 1)]

        pc_race_attempt_un = pc_data[(pc_data['event90'] == 1) & (pc_data['raceUN'] == 1)]
        pc_race_NO_attempt_un = pc_data[(pc_data['event90'] == 0) & (pc_data['raceUN'] == 1)]

        pc_race_attempt = pd.concat([pc_race_attempt_asian, pc_race_attempt_hp, pc_race_attempt_in, pc_race_attempt_un], ignore_index=True)
        pc_race_NO_attempt = pd.concat([pc_race_NO_attempt_asian, pc_race_NO_attempt_hp, pc_race_NO_attempt_in, pc_race_NO_attempt_un], ignore_index=True)

    print('Attempt:', pc_race_attempt.shape[0])
    print('No Attempt:', pc_race_NO_attempt.shape[0])
    print()

print()
print()

print('Non-PC:')
for rc in range(0, len(race_names)):
    rn = race_names[rc]
    print(rn)
    if rn == 'raceBlack' or rn == 'raceWH':
        nonpc_race_attempt = non_pc_data[(non_pc_data['event90'] == 1) & (non_pc_data[rn] == 1)]
        nonpc_race_NO_attempt = non_pc_data[(non_pc_data['event90'] == 0) & (non_pc_data[rn] == 1)]
    else:
        # race_names = ['raceAsian', 'raceHP', 'raceIN', 'raceUN']   NOT 'raceMUOT', 'hispanic', 'raceWH', 'raceBlack'
        nonpc_race_attempt_asian = non_pc_data[(non_pc_data['event90'] == 1) & (non_pc_data['raceAsian'] == 1)]
        nonpc_race_NO_attempt_asian = non_pc_data[(non_pc_data['event90'] == 0) & (non_pc_data['raceAsian'] == 1)]

        nonpc_race_attempt_hp = non_pc_data[(non_pc_data['event90'] == 1) & (non_pc_data['raceHP'] == 1)]
        nonpc_race_NO_attempt_hp = non_pc_data[(non_pc_data['event90'] == 0) & (non_pc_data['raceHP'] == 1)]

        nonpc_race_attempt_in = non_pc_data[(non_pc_data['event90'] == 1) & (non_pc_data['raceIN'] == 1)]
        nonpc_race_NO_attempt_in = non_pc_data[(non_pc_data['event90'] == 0) & (non_pc_data['raceIN'] == 1)]

        nonpc_race_attempt_un = non_pc_data[(non_pc_data['event90'] == 1) & (non_pc_data['raceUN'] == 1)]
        nonpc_race_NO_attempt_un = non_pc_data[(non_pc_data['event90'] == 0) & (non_pc_data['raceUN'] == 1)]

        nonpc_race_attempt = pd.concat([nonpc_race_attempt_asian, nonpc_race_attempt_hp, nonpc_race_attempt_in, nonpc_race_attempt_un], ignore_index=True)
        nonpc_race_NO_attempt = pd.concat([nonpc_race_NO_attempt_asian, nonpc_race_NO_attempt_hp, nonpc_race_NO_attempt_in, nonpc_race_NO_attempt_un],ignore_index=True)

    print('Attempt:', nonpc_race_attempt.shape[0])
    print('No Attempt:', nonpc_race_NO_attempt.shape[0])
    print()


##########################################################################################################


# y_pc = pc_data["event90"]
#
# y_non_pc = non_pc_data["event90"]

# non_pc_data = non_pc_data.drop(columns=["event90"])
#
# pc_data = pc_data.drop(columns=["event90"])

print()
print('-------------------------')
print()
print()


#########################################################################################
#########################################################################################
#########################################################################################


parser = argparse.ArgumentParser(description='FAIRNESS')
# parser.add_argument('--p', '--pretrain', default='no', type=str, help='Pre-train or not')
# parser.add_argument('--d', '--train', default='single', type=str, help='Data to train classifier on: single or both') # 'test', 'train'
parser.add_argument('--m', '--model', default='logs/saved_models/xgboost_entire_data_testPC.pkl', type=str, help='Model to evaluate')
parser.add_argument('--mt', '--model_type', default='ML', type=str, help='ML (pkl) or DL (pth) model to load')
parser.add_argument('--sv', '--save_file', default='xgboost_entire_data', type=str, help='File name for the plot')
parser.add_argument('--td', '--test_domain', default=0, type=int, help='0=PC, 1=Non-PC')
args = parser.parse_args()


# Load Model
model_type = args.mt
save_file = args.sv
model_l = args.m
test_data_idx = args.td


# if test_data_idx == 0:
print('PC:')
data_pc_race = {}
label_pc_race = {}
for rc in range(0, len(race_names)):
    rn = race_names[rc]
    print(rn)

    if rn == 'raceBlack' or rn == 'raceWH':
        pc_race = pc_data[(pc_data[rn] == 1)]
    else:
        # race_names = ['raceAsian', 'raceHP', 'raceIN', 'raceUN']   NOT 'raceMUOT', 'hispanic', 'raceWH', 'raceBlack'
        pc_race_asian = pc_data[(pc_data['raceAsian'] == 1)]
        pc_race_hp = pc_data[(pc_data['raceHP'] == 1)]
        pc_race_in = pc_data[(pc_data['raceIN'] == 1)]
        pc_race_un = pc_data[(pc_data['raceUN'] == 1)]

        pc_race = pd.concat([pc_race_asian, pc_race_hp, pc_race_in, pc_race_un],ignore_index=True)

    y_pc = pc_race["event90"]
    pc_race = pc_race.drop(columns=["event90"])
    data_pc_race[rn] = pc_race
    label_pc_race[rn] = y_pc
    # data_pc_race.append(pc_race)
    # label_pc_race.append(y_pc)
    print(pc_race.shape)
    print(y_pc.shape)
    print()

# elif test_data_idx == 1:
print('Non-PC:')
data_nonpc_race = {}
label_nonpc_race = {}
for rc in range(0, len(race_names)):
    rn = race_names[rc]
    print(rn)

    if rn == 'raceBlack' or rn == 'raceWH':
        nonpc_race = non_pc_data[(non_pc_data[rn] == 1)]
    else:
        # race_names = ['raceAsian', 'raceHP', 'raceIN', 'raceUN']   NOT 'raceMUOT', 'hispanic', 'raceWH', 'raceBlack'
        nonpc_race_asian = non_pc_data[(non_pc_data['raceAsian'] == 1)]
        nonpc_race_hp = non_pc_data[(non_pc_data['raceHP'] == 1)]
        nonpc_race_in = non_pc_data[(non_pc_data['raceIN'] == 1)]
        nonpc_race_un = non_pc_data[(non_pc_data['raceUN'] == 1)]

        nonpc_race = pd.concat([nonpc_race_asian, nonpc_race_hp, nonpc_race_in, nonpc_race_un], ignore_index=True)

    y_non_pc = nonpc_race["event90"]
    nonpc_race = nonpc_race.drop(columns=["event90"])
    data_nonpc_race[rn] = nonpc_race
    label_nonpc_race[rn] = y_non_pc
    # data_nonpc_race.append(nonpc_race)
    # label_nonpc_race.append(y_non_pc)
    print(nonpc_race.shape)
    print(y_non_pc.shape)
    print()

print()

# print()
# print('Shapes after dropping target columns event90:')
# print(data_pc_race[0].shape)
# print(data_nonpc_race[0].shape)

# print(data_nonpc_race['raceHP'].shape)



data_race = [data_pc_race, data_nonpc_race]
label_race = [label_pc_race, label_nonpc_race]

X_races = data_race[test_data_idx]
y_races = label_race[test_data_idx]

# print('look here')
# print(type(X_races))
# print(type(y_races))

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

print()
print()


times = 1000 # how many times to sample
random_seed = 42

rng = np.random.RandomState(seed=random_seed)


for i in range(0, len(race_names)):

    results = {}

    sensitivities = []
    specificities = []
    f1s = []
    roc_aucs = []
    recalls = []
    precisions = []
    prc_aucs = []

    race_ = race_names[i]
    print(race_)

    if race_ == 'hispanic' or race_ == 'raceMUOT':
        continue # ignore 'hispanic', 'raceMUOT' --- there are zero

    X_gt = X_races[race_]
    y_gt = y_races[race_]

    preds = model.predict(X_gt.values)
    pred_probs = model.predict_proba(X_gt.values)

    # get predictions, probabilities, calculate metrics, plots/curves, save for each race to csv
    # do bootstrapping
    # ignore 'hispanic', 'raceMUOT' --- there are zero
    # some races dont have any suicide attempts

    for i in range(times):
        pred_idx = []
        for l in range(2):  # number of classes? -- mine is 2 not 3 (Ruofan)
            idx_l = np.where(y_gt.values == l)[0]
            pred_idx_l = rng.choice(idx_l, size=idx_l.shape[0], replace=True)
            pred_idx.extend(pred_idx_l)

        pred_idx = np.hstack(pred_idx)

        print(i)
        print('pred_idx:', pred_idx)
        print('GT pred_idx:', y_gt.values[pred_idx])
        print()

        try:
            roc_auc = metrics.roc_auc_score(y_gt.values[pred_idx], pred_probs[:, 1][pred_idx])
            roc_aucs.append(roc_auc)
            results['roc_auc'] = np.mean(roc_aucs)
        except Exception:
            print('Cannot calculate ROC AUC for:', race_)

        f1 = metrics.f1_score(y_gt.values[pred_idx], preds[pred_idx])
        f1s.append(f1)

        recall = metrics.recall_score(y_gt.values[pred_idx], preds[pred_idx])
        recalls.append(recall)

        precision = metrics.precision_score(y_gt.values[pred_idx], preds[pred_idx])
        precisions.append(precision)

        try:
            tn, fp, fn, tp = metrics.confusion_matrix(y_gt.values[pred_idx], preds[pred_idx]).ravel()
            cm = metrics.confusion_matrix(y_gt.values[pred_idx], preds[pred_idx])

            # save confusion matrix
            # Create a DataFrame from the confusion matrix
            cm_df = pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
            # Save DataFrame to CSV
            cm_df.to_csv('logs_MH_subset/fairness_race/' + race_ + '_' + save_file + '_confusion_matrix.csv', index=True)

            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'],
                        yticklabels=['Actual 0', 'Actual 1'], annot_kws={"size": 20}, cbar=False)
            plt.ylabel('Actual', fontsize=20)
            plt.xlabel('Predicted', fontsize=20)
            plt.title(race_, fontsize=25)
            # Save plot as image file
            plt.savefig('logs_MH_subset/fairness_race/' + race_ + '_' + save_file + '_confusion_matrix.png')

            specificity = tn / (tn + fp)
            # print("Specificity score is: ", specificity)
            sensitivity = tp / (tp + fn)
            # print("Sensitivity score is: ", sensitivity)
            sensitivities.append(sensitivity)
            specificities.append(specificity)
            results['spec'] = np.mean(specificities)
            results['sens'] = np.mean(sensitivities)
        except Exception:
            print('Cannot calculate sensitivity, specificity (confusion matrix) for:', race_)

        try:
            prec_test, recall_test, _ = metrics.precision_recall_curve(y_gt.values[pred_idx], pred_probs[:, 1][pred_idx])
            # Calculate AUC-PRC
            auc_prc_test = metrics.auc(recall_test, prec_test)
            prc_aucs.append(auc_prc_test)
            results['prc_auc'] = np.mean(prc_aucs)
        except Exception:
            print('Cannot calculate AUC PRC for:', race_)

    try:
        ci_lower_sensitivities = np.percentile(sensitivities, 2.5)
        ci_upper_sensitivities = np.percentile(sensitivities, 97.5)

        ci_lower_specificities = np.percentile(specificities, 2.5)
        ci_upper_specificities = np.percentile(specificities, 97.5)

        results['spec 95%CI'] = (ci_lower_specificities, ci_upper_specificities)
        results['sens 95%CI'] = (ci_lower_sensitivities, ci_upper_sensitivities)
    except Exception:
        print('Cannot calculate sensitivity, specificity CI for:', race_)

    ci_lower_f1s = np.percentile(f1s, 2.5)
    ci_upper_f1s = np.percentile(f1s, 97.5)

    try:
        ci_lower_roc_aucs = np.percentile(roc_aucs, 2.5)
        ci_upper_roc_aucs = np.percentile(roc_aucs, 97.5)
        results['roc_auc 95%CI:'] = (ci_lower_roc_aucs, ci_upper_roc_aucs)
    except Exception:
        print('Cannot calculate AUC ROC CI for:', race_)

    ci_lower_recalls = np.percentile(recalls, 2.5)
    ci_upper_recalls = np.percentile(recalls, 97.5)

    ci_lower_precisions = np.percentile(precisions, 2.5)
    ci_upper_precisions = np.percentile(precisions, 97.5)

    try:
        ci_lower_prc_aucs = np.percentile(prc_aucs, 2.5)
        ci_upper_prc_aucs = np.percentile(prc_aucs, 97.5)
        results['prc_auc 95%CI'] = (ci_lower_prc_aucs, ci_upper_prc_aucs)
    except Exception:
        print('Cannot calculate AUC PRC CI for:', race_)


    results['precision'] = np.mean(precisions)
    results['recall'] = np.mean(recalls)
    results['f1'] = np.mean(f1s)


    results['precision 95%CI'] = (ci_lower_precisions, ci_upper_precisions)
    results['recall 95%CI:'] = (ci_lower_recalls, ci_upper_recalls)
    results['f1 95%CI:'] = (ci_lower_f1s, ci_upper_f1s)


    print(results)

    # Convert dictionary to DataFrame
    df = pd.DataFrame(results)

    # Write DataFrame to CSV
    df.to_csv('logs_MH_subset/fairness_race/' + race_ + '_' + save_file + '.csv', index=False)


