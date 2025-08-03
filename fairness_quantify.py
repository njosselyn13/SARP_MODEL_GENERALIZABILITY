
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
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference, demographic_parity_ratio, equalized_odds_ratio
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

non_pc_data = non_pc_data.drop(columns=["PRIMARY_CARE_VISIT","event90"])

pc_data = pc_data.drop(columns=["PRIMARY_CARE_VISIT","event90"])


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

pc_gt = y_test_pc
non_pc_gt = y_test_non_pc
###


print()
print('--------------------------------------------------------------------------------------')
print()


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

X_test_data_options = [pc_data, non_pc_data]
y_test_data_options = [pc_gt, non_pc_gt]

X_gt = X_test_data_options[test_data_idx]
y_gt = y_test_data_options[test_data_idx]

# asian_pc = pc_data['raceAsian'].dropna().tolist()
# black_pc = pc_data['raceBlack'].dropna().tolist()
# hp_pc = pc_data['raceHP'].dropna().tolist()
# native_pc = pc_data['raceIN'].dropna().tolist()
# -------------------- muot_pc = pc_data['raceMUOT'].dropna().tolist()
# unknown_pc = pc_data['raceUN'].dropna().tolist()
# white_pc = pc_data['raceWH'].dropna().tolist()
# -------------------- hispanic_pc = pc_data['hispanic'].dropna().tolist()


if test_data_idx == 0:
    # PC data
    print("PC DATA")
    # subset_df = pc_data[['raceAsian', 'raceBlack', 'raceHP', 'raceIN', 'raceUN', 'raceWH']]
    # print(subset_df.head(50))
    sensitive_race_ethnic = []
    for index, row in pc_data.iterrows():
        if row['raceWH'] == 1:
            sensitive_race_ethnic.append('white')
        elif row['raceBlack'] == 1:
            sensitive_race_ethnic.append('black')
        # elif row['hispanic'] == 1 or row['raceMUOT'] == 1:
        #     continue
        else:
            sensitive_race_ethnic.append('other')
elif test_data_idx == 1:
    # Non-PC data
    print("Non-PC DATA")
    # subset_df = non_pc_data[['raceAsian', 'raceBlack', 'raceHP', 'raceIN', 'raceUN', 'raceWH']]
    # print(subset_df.head(50))
    sensitive_race_ethnic = []
    for index, row in non_pc_data.iterrows():
        if row['raceWH'] == 1:
            sensitive_race_ethnic.append('white')
        elif row['raceBlack'] == 1:
            sensitive_race_ethnic.append('black')
        else:
            sensitive_race_ethnic.append('other')


print()
# print('sensitive list:', sensitive_race_ethnic)
# print()
preds = model.predict(X_gt.values)
# print('preds:', preds)

# dpr = demographic_parity_difference(y_true=y_gt.values, y_pred=preds, sensitive_features=sensitive_race_ethnic)
# eqor = equalized_odds_difference(y_true=y_gt.values, y_pred=preds, sensitive_features=sensitive_race_ethnic)
#
# print('Demographic parity difference:', dpr)
# print('Equalized odds difference:', eqor)

# dpr_nonpc = demographic_parity_difference(y_true=non_pc_gt.values, y_pred=y_pred_nonpc, sensitive_features=sensitive_race_ethnic_nonpc)
# eqor_nonpc = equalized_odds_difference(y_true=non_pc_gt.values, y_pred=y_pred_nonpc, sensitive_features=sensitive_race_ethnic_nonpc)

# calcualte fairness metrics 
dpr_diff = demographic_parity_difference(y_true=y_gt.values, y_pred=preds, sensitive_features=sensitive_race_ethnic)
eqodd_diff = equalized_odds_difference(y_true=y_gt.values, y_pred=preds, sensitive_features=sensitive_race_ethnic)

dpr_ratio = demographic_parity_ratio(y_true=y_gt.values, y_pred=preds, sensitive_features=sensitive_race_ethnic)
eqodd_ratio = equalized_odds_ratio(y_true=y_gt.values, y_pred=preds, sensitive_features=sensitive_race_ethnic)

# fairlearn.metrics.demographic_parity_ratio(y_true, y_pred, *, sensitive_features, method='between_groups', sample_weight=None)
# fairlearn.metrics.equalized_odds_ratio(y_true, y_pred, *, sensitive_features, method='between_groups', sample_weight=None)

print('Demographic parity difference:', dpr_diff)
print('Equalized odds difference:', eqodd_diff)
print('Demographic parity ratio:', dpr_ratio)
print('Equalized odds ratio:', eqodd_ratio)

df = pd.DataFrame()

df['Demographic parity difference'] = pd.Series(dpr_diff)
df['Equalized odds difference'] = pd.Series(eqodd_diff)
df['Demographic parity ratio'] = pd.Series(dpr_ratio)
df['Equalized odds ratio'] = pd.Series(eqodd_ratio)

df.to_csv('logs_MH_subset/fairness_race/' + save_file + '_quantify.csv', index=False)


