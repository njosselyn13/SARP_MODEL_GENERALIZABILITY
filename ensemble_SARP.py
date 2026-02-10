import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score
from tqdm import tqdm
from sklearn.preprocessing import scale
from sklearn import metrics
from sklearn.metrics import accuracy_score
from scipy.stats import mode
from sklearn.linear_model import LogisticRegression

# import sys
# sys.path.append("/path/to/rtdl-revisiting-models/package")
# import rtdl_revisiting_models as rtdl

# from rtdl_revisiting_models import MLP, ResNet, FTTransformer

# import rtdl
import time
import csv
import os
import random
import joblib
import argparse

# =====================
# USER CONFIG
# =====================
# DATA_PATH = "data.csv"   # your tabular dataset
# TARGET_COL = "target"     # name of your binary label column
# BINARY_COLS = ["bin1", "bin2", "bin3"]  # list of binary/categorical columns
# NUMERIC_COLS = ["num1", "num2", "num3"]  # list of numeric columns

parser = argparse.ArgumentParser(description='Ensemble')
parser.add_argument('--setting', default='pc2pc', type=str, help='setting task: pc2pc, pc2nonpc, nonpc2pc, nonpc2nonpc')
# parser.add_argument('--m', '--model', default='logs/saved_models/xgboost_entire_data_testPC.pkl', type=str, help='Model to evaluate')
# parser.add_argument('--mc', default='xgboost', type=str, help='model: xgboost, RF, tabnet0, tabnet1, tabnet2, FT, MLP, resnet')
parser.add_argument('--ensemble', default='simple', type=str, help='ensemble type: simple, ...')
parser.add_argument('--metrics_csv', default='ensemble_results/ensemble_metrics', type=str, help='metrics save file')
# parser.add_argument('--n_trials', default=30, type=int, help='number optuna trials')
# parser.add_argument('--epochs_fm', '--epochs_final_model', default=100, type=int, help='number epochs for final model')
# parser.add_argument('--epochs_op', '--epochs_optuna', default=50, type=int, help='number epochs for optuna tuning')
args = parser.parse_args()

start_time = time.time()

METRICS_CSV = args.metrics_csv #"ftt_metrics.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# N_TRIALS = args.n_trials #30  # optuna trials

# model_chosen == args.mc
# model_l = args.m
ensemble = args.ensemble
setting = args.setting #'pc2pc'


# =====================
# LOAD DATA
# =====================
# df = pd.read_csv(DATA_PATH)
#
# X = df[BINARY_COLS + NUMERIC_COLS].fillna(-1)
# y = df[TARGET_COL].astype(int)
#
# X_train, X_val, y_train, y_val = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )


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


# In[7]:
data["event90"] = data["event90"].fillna(value=0)


data = data.drop(columns=["income", "college", "hhld_inc_It40k", "coll_deg_It25p"])

data.rename(columns={'hhld_inc_lt40k_NJ': 'hhld_inc_It40k', 'coll_deg_lt25p_NJ': 'coll_deg_It25p'}, inplace=True)



# ## Selecting Primary Care data where Flag(PRIMARY_CARE_VISIT) == 1

pc_data = data[data["PRIMARY_CARE_VISIT"] == 1]
# print(pc_data)


y = pc_data["event90"]


# ## Selecting Non Primary Care data where Flag(PRIMARY_CARE_VISIT) == 0

non_pc_data = data[data["PRIMARY_CARE_VISIT"] == 0]
# print(non_pc_data)


y_non_pc = non_pc_data["event90"]


pc_data = pc_data.drop(columns=["PRIMARY_CARE_VISIT","person_id","event30","event90","death30","death90","visit_mh"])


# In[25]:
non_pc_data = non_pc_data.drop(columns=["PRIMARY_CARE_VISIT","person_id","event30","event90","death30","death90","visit_mh"])
# print(non_pc_data)


# In[16]:
# pc_columns_to_scale = ["age","days_since_prev","charlson_score","charlson_a","dep_dx_pre5y_cumulative_a","anx_dx_pre5y_cumulative_a","bip_dx_pre5y_cumulative_a","sch_dx_pre5y_cumulative_a","phqnumber90","phqnumber183","phqnumber365","phq8_index_score_calc_f","raceAsian_8","raceIN_8","hispanic_8","age_8","q9_0_a","q9_1_8","q9_2_8","q9_3_8","q9_1_c","q9_2_c","q9_3_c","any_sui_att_pre5y_cumulative_a","any_sui_att_pre5y_cumulative_8","any_sui_att_pre5y_cumulative_c"]
pc_columns_to_scale = ['age', 'days_since_prev', 'dep_dx_pre5y_noi_cumulative', 'anx_dx_pre5y_noi_cumulative', 'bip_dx_pre5y_noi_cumulative', 'sch_dx_pre5y_noi_cumulative', 'oth_dx_pre5y_noi_cumulative', 'dem_dx_pre5y_noi_cumulative', 'add_dx_pre5y_noi_cumulative', 'asd_dx_pre5y_noi_cumulative', 'per_dx_pre5y_noi_cumulative', 'alc_dx_pre5y_noi_cumulative', 'pts_dx_pre5y_noi_cumulative', 'eat_dx_pre5y_noi_cumulative', 'tbi_dx_pre5y_noi_cumulative', 'dru_dx_pre5y_noi_cumulative', 'antidep_rx_pre1y_cumulative', 'antidep_rx_pre5y_cumulative', 'benzo_rx_pre1y_cumulative', 'benzo_rx_pre5y_cumulative', 'hypno_rx_pre1y_cumulative', 'hypno_rx_pre5y_cumulative', 'sga_rx_pre1y_cumulative', 'sga_rx_pre5y_cumulative', 'mh_ip_pre1y_cumulative', 'mh_ip_pre5y_cumulative', 'mh_op_pre1y_cumulative', 'mh_op_pre5y_cumulative', 'mh_ed_pre1y_cumulative', 'mh_ed_pre5y_cumulative', 'any_sui_att_pre1y_cumulative', 'any_sui_att_pre5y_cumulative', 'lvi_sui_att_pre1y_cumulative', 'lvi_sui_att_pre5y_cumulative', 'ovi_sui_att_pre1y_cumulative', 'ovi_sui_att_pre5y_cumulative', 'any_inj_poi_pre1y_cumulative', 'any_inj_poi_pre5y_cumulative', 'any_sui_att_pre5y_cumulative_f', 'any_sui_att_pre5y_cumulative_a', 'charlson_score', 'charlson_a', 'phqnumber90', 'phqnumber183', 'phqnumber365', 'dep_dx_pre5y_cumulative', 'dep_dx_pre5y_cumulative_f', 'dep_dx_pre5y_cumulative_a', 'anx_dx_pre5y_cumulative', 'anx_dx_pre5y_cumulative_f', 'anx_dx_pre5y_cumulative_a', 'bip_dx_pre5y_cumulative', 'bip_dx_pre5y_cumulative_f', 'bip_dx_pre5y_cumulative_a', 'sch_dx_pre5y_cumulative', 'sch_dx_pre5y_cumulative_f', 'sch_dx_pre5y_cumulative_a', 'oth_dx_pre5y_cumulative', 'dem_dx_pre5y_cumulative', 'add_dx_pre5y_cumulative', 'asd_dx_pre5y_cumulative', 'per_dx_pre5y_cumulative', 'alc_dx_pre5y_cumulative', 'dru_dx_pre5y_cumulative', 'pts_dx_pre5y_cumulative', 'eat_dx_pre5y_cumulative', 'tbi_dx_pre5y_cumulative', 'phq8_index_score_calc', 'phq8_index_score_calc_f', 'raceAsian_8', 'raceBlack_8', 'raceHP_8', 'raceIN_8', 'raceMUOT_8', 'raceUN_8', 'hispanic_8', 'age_8', 'q9_0_a', 'q9_1_a', 'q9_2_a', 'q9_3_a', 'q9_0_8', 'q9_1_8', 'q9_2_8', 'q9_3_8', 'q9_0_c', 'q9_1_c', 'q9_2_c', 'q9_3_c', 'any_sui_att_pre5y_cumulative_8', 'any_sui_att_pre5y_cumulative_c', 'any_sui_att_pre5y_cumulative_de', 'any_sui_att_pre5y_cumulative_an', 'any_sui_att_pre5y_cumulative_bi', 'any_sui_att_pre5y_cumulative_sc', 'any_sui_att_pre5y_cumulative_al', 'any_sui_att_pre5y_cumulative_dr', 'any_sui_att_pre5y_cumulative_pe']
# these columns to scale are same as all the numeric (num_cols) columns

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


# for i in missing_columns:
#     if sum(pc_data[i].isnull()) == pc_data.shape[0]: #188894:
#         print(i)
#         print(sum(pc_data[i].isnull()))
#     else:
#         print(i)
#         print(sum(pc_data[i].isnull()))



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


# non_pc_columns_to_scale = ["age","days_since_prev","charlson_score","charlson_a","dep_dx_pre5y_cumulative_a","anx_dx_pre5y_cumulative_a","bip_dx_pre5y_cumulative_a","sch_dx_pre5y_cumulative_a","phqnumber90","phqnumber183","phqnumber365","phq8_index_score_calc_f","raceAsian_8","raceIN_8","hispanic_8","age_8","q9_0_a","q9_1_8","q9_2_8","q9_3_8","q9_1_c","q9_2_c","q9_3_c","any_sui_att_pre5y_cumulative_a","any_sui_att_pre5y_cumulative_8","any_sui_att_pre5y_cumulative_c"]
non_pc_columns_to_scale = ['age', 'days_since_prev', 'dep_dx_pre5y_noi_cumulative', 'anx_dx_pre5y_noi_cumulative', 'bip_dx_pre5y_noi_cumulative', 'sch_dx_pre5y_noi_cumulative', 'oth_dx_pre5y_noi_cumulative', 'dem_dx_pre5y_noi_cumulative', 'add_dx_pre5y_noi_cumulative', 'asd_dx_pre5y_noi_cumulative', 'per_dx_pre5y_noi_cumulative', 'alc_dx_pre5y_noi_cumulative', 'pts_dx_pre5y_noi_cumulative', 'eat_dx_pre5y_noi_cumulative', 'tbi_dx_pre5y_noi_cumulative', 'dru_dx_pre5y_noi_cumulative', 'antidep_rx_pre1y_cumulative', 'antidep_rx_pre5y_cumulative', 'benzo_rx_pre1y_cumulative', 'benzo_rx_pre5y_cumulative', 'hypno_rx_pre1y_cumulative', 'hypno_rx_pre5y_cumulative', 'sga_rx_pre1y_cumulative', 'sga_rx_pre5y_cumulative', 'mh_ip_pre1y_cumulative', 'mh_ip_pre5y_cumulative', 'mh_op_pre1y_cumulative', 'mh_op_pre5y_cumulative', 'mh_ed_pre1y_cumulative', 'mh_ed_pre5y_cumulative', 'any_sui_att_pre1y_cumulative', 'any_sui_att_pre5y_cumulative', 'lvi_sui_att_pre1y_cumulative', 'lvi_sui_att_pre5y_cumulative', 'ovi_sui_att_pre1y_cumulative', 'ovi_sui_att_pre5y_cumulative', 'any_inj_poi_pre1y_cumulative', 'any_inj_poi_pre5y_cumulative', 'any_sui_att_pre5y_cumulative_f', 'any_sui_att_pre5y_cumulative_a', 'charlson_score', 'charlson_a', 'phqnumber90', 'phqnumber183', 'phqnumber365', 'dep_dx_pre5y_cumulative', 'dep_dx_pre5y_cumulative_f', 'dep_dx_pre5y_cumulative_a', 'anx_dx_pre5y_cumulative', 'anx_dx_pre5y_cumulative_f', 'anx_dx_pre5y_cumulative_a', 'bip_dx_pre5y_cumulative', 'bip_dx_pre5y_cumulative_f', 'bip_dx_pre5y_cumulative_a', 'sch_dx_pre5y_cumulative', 'sch_dx_pre5y_cumulative_f', 'sch_dx_pre5y_cumulative_a', 'oth_dx_pre5y_cumulative', 'dem_dx_pre5y_cumulative', 'add_dx_pre5y_cumulative', 'asd_dx_pre5y_cumulative', 'per_dx_pre5y_cumulative', 'alc_dx_pre5y_cumulative', 'dru_dx_pre5y_cumulative', 'pts_dx_pre5y_cumulative', 'eat_dx_pre5y_cumulative', 'tbi_dx_pre5y_cumulative', 'phq8_index_score_calc', 'phq8_index_score_calc_f', 'raceAsian_8', 'raceBlack_8', 'raceHP_8', 'raceIN_8', 'raceMUOT_8', 'raceUN_8', 'hispanic_8', 'age_8', 'q9_0_a', 'q9_1_a', 'q9_2_a', 'q9_3_a', 'q9_0_8', 'q9_1_8', 'q9_2_8', 'q9_3_8', 'q9_0_c', 'q9_1_c', 'q9_2_c', 'q9_3_c', 'any_sui_att_pre5y_cumulative_8', 'any_sui_att_pre5y_cumulative_c', 'any_sui_att_pre5y_cumulative_de', 'any_sui_att_pre5y_cumulative_an', 'any_sui_att_pre5y_cumulative_bi', 'any_sui_att_pre5y_cumulative_sc', 'any_sui_att_pre5y_cumulative_al', 'any_sui_att_pre5y_cumulative_dr', 'any_sui_att_pre5y_cumulative_pe']
# these columns to scale are same as all the numeric (num_cols) columns


count = 0
for i in non_pc_columns_to_scale:
    if i in non_pc_data.columns:
        count += 1
    else:
        print(i)
print(count)


non_pc_data[non_pc_columns_to_scale] = scale(non_pc_data[non_pc_columns_to_scale])
# print(non_pc_data)

missing_columns = non_pc_data.columns[non_pc_data.isnull().any()]
# print(missing_columns)



# # In[34]:
# for i in missing_columns:
#     if sum(non_pc_data[i].isnull()) == non_pc_data.shape[0]: #257999:
#         print(i)
#         print(sum(non_pc_data[i].isnull()))
#     else:
#         print(i)
#         print(sum(non_pc_data[i].isnull()))




for i in missing_columns:
    if sum(non_pc_data[i].isnull()) == non_pc_data.shape[0]: #412045: #446893:
        # data[i].fillna(value=0,inplace=True)
        non_pc_data = non_pc_data.drop(columns=[i]) # if all rows are empty for a column, drop that column
        # print(sum(data[i].isnull()))
        print("Empty column:", i)
    else:
        non_pc_data[i].fillna(value=-1, inplace=True) # # if only some of rows have missing value for a column, then fill that with -1, for num or bin cols







model_chosen_list = ['xgboost', 'RF', 'tabnet0', 'tabnet1', 'tabnet2', 'FT', 'MLP', 'resnet']


if setting == 'pc2pc':
    # pc2pc
    model_files = ['logs_MH_subset/saved_models/xgboost_train1_test1_pc2pc.pkl', 'logs_MH_subset/saved_models/RF_train1_test1_pc2pc.pkl',
                   'logs_MH_subset/saved_models/TabNet_train1_test1_pc2pc.zip', 'logs_MH_subset/saved_models/TabNet_pre1_train1_test1_pc2pc.zip',
                   'logs_MH_subset/saved_models/TabNet_pre2_train1_test1_pc2pc_over13.zip',
                   'logs_MH_subset/new_models/saved_models/FTT_train1_test1_pc2pc.pth',
                   'logs_MH_subset/new_models/saved_models/MLP_train1_test1_pc2pc.pth',
                   'logs_MH_subset/new_models/saved_models/ResNet_train1_test1_pc2pc.pth']

elif setting == 'pc2nonpc':
    # pc2nonpc
    model_files = ['logs_MH_subset/saved_models/xgboost_train1_test1_pc2nonpc.pkl', 'logs_MH_subset/saved_models/RF_train1_test1_pc2nonpc.pkl',
                   'logs_MH_subset/saved_models/TabNet_train1_test1_pc2nonpc.zip', 'logs_MH_subset/saved_models/TabNet_pre1_train1_test1_pc2nonpc.zip',
                   'logs_MH_subset/saved_models/TabNet_pre2_train1_test1_pc2nonpc_over13.zip',
                   'logs_MH_subset/new_models/saved_models/FTT_train1_test1_pc2nonpc.pth',
                   'logs_MH_subset/new_models/saved_models/MLP_train1_test1_pc2nonpc.pth',
                   'logs_MH_subset/new_models/saved_models/ResNet_train1_test1_pc2nonpc.pth']

elif setting == 'nonpc2nonpc':
    # nonpc2nonpc
    model_files = ['logs_MH_subset/saved_models/xgboost_train1_test1_nonpc2nonpc.pkl', 'logs_MH_subset/saved_models/RF_train1_test1_nonpc2nonpc.pkl',
                   'logs_MH_subset/saved_models/TabNet_train1_test1_nonpc2nonpc.zip', 'logs_MH_subset/saved_models/TabNet_pre1_train1_test1_nonpc2nonpc.zip',
                   'logs_MH_subset/saved_models/TabNet_pre2_run2_train1_test1_nonpc2nonpc_over13.zip',
                   'logs_MH_subset/new_models/saved_models/FTT_train1_test1_nonpc2nonpc.pth',
                   'logs_MH_subset/new_models/saved_models/MLP_train1_test1_nonpc2nonpc.pth',
                   'logs_MH_subset/new_models/saved_models/ResNet_train1_test1_nonpc2nonpc.pth']

elif setting == 'nonpc2pc':
    # nonpc2pc
    model_files = ['logs_MH_subset/saved_models/xgboost_train1_test1_nonpc2pc.pkl', 'logs_MH_subset/saved_models/RF_train1_test1_nonpc2pc.pkl',
                   'logs_MH_subset/saved_models/TabNet_train1_test1_nonpc2pc.zip', 'logs_MH_subset/saved_models/TabNet_pre1_train1_test1_nonpc2pc_over13.zip',
                   'logs_MH_subset/saved_models/TabNet_pre2_run2_train1_test1_nonpc2pc.zip',
                   'logs_MH_subset/new_models/saved_models/FTT_train1_test1_nonpc2pc.pth',
                   'logs_MH_subset/new_models/saved_models/MLP_train1_test1_nonpc2pc.pth',
                   'logs_MH_subset/new_models/saved_models/ResNet_train1_test1_nonpc2pc.pth']

print()
print()

model_dict_preds = {}
model_dict_probs = {}
model_dict_gt = {}

model_dict_preds_val = {}
model_dict_probs_val = {}
model_dict_gt_val = {}


for ii in range(0, len(model_files)):
    model_chosen = model_chosen_list[ii]
    model_l = model_files[ii]

    # ### Splitting data into train and validation

    # In[36]:
    train_X, X_test, train_y, y_test = train_test_split(pc_data, y, test_size=0.35, random_state=42, stratify = y)


    # In[37]:
    X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, test_size=0.35, random_state=42, stratify = train_y)


    # In[38]:
    train_X_non_pc, X_test_non_pc, train_y_non_pc, y_test_non_pc = train_test_split(non_pc_data, y_non_pc, test_size=0.35, random_state=42, stratify = y_non_pc)


    # In[39]:
    X_train_non_pc, X_val_non_pc, y_train_non_pc, y_val_non_pc = train_test_split(train_X_non_pc, train_y_non_pc, test_size=0.35, random_state=42, stratify = train_y_non_pc)



    ##############################################################################################
    ##############################################################################################
    ##############################################################################################
    ##############################################################################################
    ##############################################################################################


    # make a model_chosen list and model file list
    # dont need to use args for --m or --mc
    # pick a task (pc2pc)
    # define a list of the 8 models and model_chosen for that task in matching orders
    # loop through these in a big loop


    # # Convert to tensors
    # def to_tensor(data):
    #     return torch.tensor(data.values, dtype=torch.float32)

    # pc: (X_train, y_train) (X_val, y_val) (X_test, y_test)
    # non-pc: (X_train_non_pc, y_train_non_pc) (X_val_non_pc, y_val_non_pc) (X_test_non_pc, y_test_non_pc)


    print('Setting:', setting)
    print()

    if setting == 'pc2pc':
        print(setting)
        X_train = X_train
        y_train = y_train
        X_val = X_val
        y_val = y_val
        X_test = X_test
        y_test = y_test
    elif setting == 'pc2nonpc':
        print(setting)
        X_train = X_train
        y_train = y_train
        X_val = X_val_non_pc
        y_val = y_val_non_pc
        X_test = X_test_non_pc
        y_test = y_test_non_pc
    elif setting == 'nonpc2pc':
        print(setting)
        X_train = X_train_non_pc
        y_train = y_train_non_pc
        X_val = X_val
        y_val = y_val
        X_test = X_test
        y_test = y_test
    elif setting == 'nonpc2nonpc':
        print(setting)
        X_train = X_train_non_pc
        y_train = y_train_non_pc
        X_val = X_val_non_pc
        y_val = y_val_non_pc
        X_test = X_test_non_pc
        y_test = y_test_non_pc
    else:
        print('setting not found')

    print()

    print(model_chosen)
    print()


    if model_chosen == 'FT' or model_chosen == 'MLP' or model_chosen == 'resnet':

        print(model_chosen)
        print()

        print('number of binary columns BEORE filtering:', len(bin_cols))
        print('number of numeric columns BEORE filtering:', len(num_cols))
        print('number of binary+numeric columns BEORE filtering:', len(bin_cols+num_cols))
        print()

        BINARY_COLS = [c for c in bin_cols if c in X_train.columns]
        NUMERIC_COLS = [c for c in num_cols if c in X_train.columns]

        print('number of binary columns after filtering:', len(BINARY_COLS))
        print('number of numeric columns after filtering:', len(NUMERIC_COLS))
        print('number of binary+numeric columns after filtering:', len(BINARY_COLS+NUMERIC_COLS))
        print()

        # cat_cardinalities = []
        # for c in BINARY_COLS:
        #     # count unique values in the column
        #     n_unique = X_train[c].nunique(dropna=False)  # include -1 as a valid category
        #     cat_cardinalities.append(n_unique)

        cat_cardinalities = [3] * len(BINARY_COLS)

        print('unique cat_cardinality values:', np.unique(cat_cardinalities))
        print()

        # Split categorical and numeric for tensors
        X_train_cat = torch.tensor(X_train[BINARY_COLS].values, dtype=torch.long).to(DEVICE)
        X_train_cat = X_train_cat.clone()
        X_train_cat[X_train_cat == -1] = 2  # if you want to code missing as a separate category

        X_val_cat   = torch.tensor(X_val[BINARY_COLS].values, dtype=torch.long).to(DEVICE)
        X_val_cat = X_val_cat.clone()
        X_val_cat[X_val_cat == -1] = 2  # if you want to code missing as a separate category

        X_test_cat = torch.tensor(X_test[BINARY_COLS].values, dtype=torch.long).to(DEVICE)
        X_test_cat = X_test_cat.clone()
        X_test_cat[X_test_cat == -1] = 2  # if you want to code missing as a separate category

        X_train_num = torch.tensor(X_train[NUMERIC_COLS].values, dtype=torch.float32).to(DEVICE)
        X_val_num   = torch.tensor(X_val[NUMERIC_COLS].values, dtype=torch.float32).to(DEVICE)
        X_test_num   = torch.tensor(X_test[NUMERIC_COLS].values, dtype=torch.float32).to(DEVICE)

        y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(DEVICE)
        y_val_t   = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1).to(DEVICE)
        y_test_t = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1).to(DEVICE)


    if model_chosen == 'MLP' or model_chosen == 'resnet':
        print(model_chosen)
        print()
        x_cat_train_ohe = [
            F.one_hot(cat_column, c)
            for cat_column, c in zip(X_train_cat.T, cat_cardinalities)
        ]
        X_train = torch.column_stack([X_train_num] + x_cat_train_ohe)

        # print('X_train shape:', X_train.shape)
        # print()

        x_cat_val_ohe = [
            F.one_hot(cat_column, c)
            for cat_column, c in zip(X_val_cat.T, cat_cardinalities)
        ]
        X_val = torch.column_stack([X_val_num] + x_cat_val_ohe)

        x_cat_test_ohe = [
            F.one_hot(cat_column, c)
            for cat_column, c in zip(X_test_cat.T, cat_cardinalities)
        ]
        X_test = torch.column_stack([X_test_num] + x_cat_test_ohe)


    if model_chosen == 'xgboost' or model_chosen == 'RF':
        print('Model path:', model_l)
        model = joblib.load(model_l)
    elif model_chosen == 'FT' or model_chosen == 'MLP' or model_chosen == 'resnet':
        # from rtdl_revisiting_models import MLP, ResNet, FTTransformer
        # model = MLP()
        model = torch.load(model_l, map_location="cuda" if torch.cuda.is_available() else "cpu")
    elif model_chosen == 'tabnet0' or model_chosen == 'tabnet1' or model_chosen == 'tabnet2':
        from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
        print('Model path:', model_l)
        model = TabNetClassifier()
        model.load_model(model_l)

    else:
        print('Model path not found:', model_chosen, model_l)



    if model_chosen == 'xgboost':
        # eval model n test set, get probs and preds
        # save preds array to model_dict_preds for xgboost
        # save probs array to model_dict_probs for xgboost
        preds = model.predict(X_test.values)
        # print('xgboost preds before sigmoid:', preds)
        # print()
        pred_probs = model.predict_proba(X_test.values)
        model_dict_preds['xgboost'] = preds
        model_dict_probs['xgboost'] = pred_probs
        model_dict_gt['xgboost'] = y_test.values

        preds_val = model.predict(X_val.values)
        pred_probs_val = model.predict_proba(X_val.values)
        model_dict_preds_val['xgboost'] = preds_val
        model_dict_probs_val['xgboost'] = pred_probs_val
        model_dict_gt_val['xgboost'] = y_val.values
    elif model_chosen == 'RF':
        preds = model.predict(X_test.values)
        pred_probs = model.predict_proba(X_test.values)
        model_dict_preds['rf'] = preds
        model_dict_probs['rf'] = pred_probs
        model_dict_gt['rf'] = y_test.values

        preds_val = model.predict(X_val.values)
        pred_probs_val = model.predict_proba(X_val.values)
        model_dict_preds_val['rf'] = preds_val
        model_dict_probs_val['rf'] = pred_probs_val
        model_dict_gt_val['rf'] = y_val.values
    elif model_chosen == 'tabnet0':
        preds = model.predict(X_test.values)
        pred_probs = model.predict_proba(X_test.values)
        model_dict_preds['tabnet0'] = preds
        model_dict_probs['tabnet0'] = pred_probs
        model_dict_gt['tabnet0'] = y_test.values

        preds_val = model.predict(X_val.values)
        pred_probs_val = model.predict_proba(X_val.values)
        model_dict_preds_val['tabnet0'] = preds_val
        model_dict_probs_val['tabnet0'] = pred_probs_val
        model_dict_gt_val['tabnet0'] = y_val.values
    elif model_chosen == 'tabnet1':
        preds = model.predict(X_test.values)
        pred_probs = model.predict_proba(X_test.values)
        model_dict_preds['tabnet1'] = preds
        model_dict_probs['tabnet1'] = pred_probs
        model_dict_gt['tabnet1'] = y_test.values

        preds_val = model.predict(X_val.values)
        pred_probs_val = model.predict_proba(X_val.values)
        model_dict_preds_val['tabnet1'] = preds_val
        model_dict_probs_val['tabnet1'] = pred_probs_val
        model_dict_gt_val['tabnet1'] = y_val.values
    elif model_chosen == 'tabnet2':
        preds = model.predict(X_test.values)
        # print('tabnet2 preds before sigmoid:', preds)
        # print()
        pred_probs = model.predict_proba(X_test.values)
        model_dict_preds['tabnet2'] = preds
        model_dict_probs['tabnet2'] = pred_probs
        model_dict_gt['tabnet2'] = y_test.values

        preds_val = model.predict(X_val.values)
        pred_probs_val = model.predict_proba(X_val.values)
        model_dict_preds_val['tabnet2'] = preds_val
        model_dict_probs_val['tabnet2'] = pred_probs_val
        model_dict_gt_val['tabnet2'] = y_val.values

    elif model_chosen == 'FT':
        with torch.no_grad():
            preds_list, y_list = [], []
            preds_list_val, y_list_val = [], []
            test_batch_size = 512  # smaller batch size for validation

            # Loop through test set in batches
            for i in range(0, len(X_test_num), test_batch_size):
                xb_num = X_test_num[i:i + test_batch_size]
                xb_cat = X_test_cat[i:i + test_batch_size]

                # Run model forward pass and move to numpy
                preds1 = model(xb_num, xb_cat).detach().cpu().numpy()

                preds_list.append(preds1)
                y_list.append(y_test[i:i + test_batch_size].to_numpy())

            # Concatenate all batches
            preds_test = np.concatenate(preds_list).flatten()
            y_test_np = np.concatenate(y_list).flatten()

            pred_probs = 1 / (1 + np.exp(-preds_test))  # NumPy sigmoid
            y_true = np.array(y_test_np).flatten()
            preds = (pred_probs >= 0.5).astype(int)



            # Loop through val set in batches
            for i in range(0, len(X_val_num), test_batch_size):
                xb_num_val = X_val_num[i:i + test_batch_size]
                xb_cat_val = X_val_cat[i:i + test_batch_size]

                # Run model forward pass and move to numpy
                preds1_val = model(xb_num_val, xb_cat_val).detach().cpu().numpy()

                preds_list_val.append(preds1_val)
                y_list_val.append(y_val[i:i + test_batch_size].to_numpy())

            # Concatenate all batches
            preds_val = np.concatenate(preds_list_val).flatten()
            y_val_np = np.concatenate(y_list_val).flatten()

            pred_probs_val = 1 / (1 + np.exp(-preds_val))  # NumPy sigmoid
            y_true_val = np.array(y_val_np).flatten()
            preds_val = (pred_probs_val >= 0.5).astype(int)

        # preds = model.predict(X_test)
        # pred_probs = model.predict_proba(X_test)
        model_dict_preds['FT'] = preds
        model_dict_probs['FT'] = pred_probs
        # model_dict_gt['FT'] = y_list
        y_test_np = np.concatenate(y_list).flatten()
        model_dict_gt['FT'] = y_test_np

        model_dict_preds_val['FT'] = preds_val
        model_dict_probs_val['FT'] = pred_probs_val
        y_val_np = np.concatenate(y_list_val).flatten()
        model_dict_gt_val['FT'] = y_val_np
    elif model_chosen == 'MLP':
        preds1 = model(X_test)
        # print('mlp preds before sigmoid:', preds)
        # print()
        pred_probs = torch.sigmoid(preds1).detach().cpu().numpy().flatten()  # mlp, resnet
        preds = (pred_probs >= 0.5).astype(int)

        # preds = model.predict(X_test)
        # pred_probs = model.predict_proba(X_test)
        model_dict_preds['MLP'] = preds
        model_dict_probs['MLP'] = pred_probs
        model_dict_gt['MLP'] = y_test.values

        preds1_val = model(X_val)
        pred_probs_val = torch.sigmoid(preds1_val).detach().cpu().numpy().flatten()  # mlp, resnet
        preds_val = (pred_probs_val >= 0.5).astype(int)

        model_dict_preds_val['MLP'] = preds_val
        model_dict_probs_val['MLP'] = pred_probs_val
        model_dict_gt_val['MLP'] = y_val.values
    elif model_chosen == 'resnet':
        preds1 = model(X_test)
        pred_probs = torch.sigmoid(preds1).detach().cpu().numpy().flatten()  # mlp, resnet
        preds = (pred_probs >= 0.5).astype(int)

        # preds = model.predict(X_test)
        # pred_probs = model.predict_proba(X_test)
        model_dict_preds['resnet'] = preds
        model_dict_probs['resnet'] = pred_probs
        model_dict_gt['resnet'] = y_test.values

        preds1_val = model(X_val)
        pred_probs_val = torch.sigmoid(preds1_val).detach().cpu().numpy().flatten()  # mlp, resnet
        preds_val = (pred_probs_val >= 0.5).astype(int)

        model_dict_preds_val['resnet'] = preds_val
        model_dict_probs_val['resnet'] = pred_probs_val
        model_dict_gt_val['resnet'] = y_val.values

    # y_prob = torch.sigmoid(y_pred).detach().cpu().numpy().flatten() # mlp, resnet
    # y_prob = 1 / (1 + np.exp(-y_pred))  # NumPy sigmoid, ft-transformer
    # y_bin = (y_prob >= 0.5).astype(int)



    # with open("preds.csv", "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["key", "value"])  # header
    #     for k, v in model_dict_preds.items():
    #         writer.writerow([k, v])
    #
    #
    # with open("probs.csv", "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["key", "value"])  # header
    #     for k, v in model_dict_probs.items():
    #         writer.writerow([k, v])
    #
    # with open("gt.csv", "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["key", "value"])  # header
    #     for k, v in model_dict_gt.items():
    #         writer.writerow([k, v])
    #

    # save model_dict_preds, model_dict_probs, model_dict_gt to csvs


# CAN ONLY DO THIS GROUPING AFTER HAVE RAN EACH MODEL


probs_list = [ model_dict_probs['xgboost'], model_dict_probs['rf'], model_dict_probs['tabnet0'], model_dict_probs['tabnet1'],
                          model_dict_probs['tabnet2'], model_dict_probs['FT'], model_dict_probs['MLP'], model_dict_probs['resnet'] ]

preds_list = [ model_dict_preds['xgboost'], model_dict_preds['rf'], model_dict_preds['tabnet0'], model_dict_preds['tabnet1'],
                          model_dict_preds['tabnet2'], model_dict_preds['FT'], model_dict_preds['MLP'], model_dict_preds['resnet'] ]

gt_list = [ model_dict_gt['xgboost'], model_dict_gt['rf'], model_dict_gt['tabnet0'], model_dict_gt['tabnet1'],
                          model_dict_gt['tabnet2'], model_dict_gt['FT'], model_dict_gt['MLP'], model_dict_gt['resnet'] ]



probs_list_val = [ model_dict_probs_val['xgboost'], model_dict_probs_val['rf'], model_dict_probs_val['tabnet0'], model_dict_probs_val['tabnet1'],
                          model_dict_probs_val['tabnet2'], model_dict_probs_val['FT'], model_dict_probs_val['MLP'], model_dict_probs_val['resnet'] ]

preds_list_val = [ model_dict_preds_val['xgboost'], model_dict_preds_val['rf'], model_dict_preds_val['tabnet0'], model_dict_preds_val['tabnet1'],
                          model_dict_preds_val['tabnet2'], model_dict_preds_val['FT'], model_dict_preds_val['MLP'], model_dict_preds_val['resnet'] ]

gt_list_val = [ model_dict_gt_val['xgboost'], model_dict_gt_val['rf'], model_dict_gt_val['tabnet0'], model_dict_gt_val['tabnet1'],
                          model_dict_gt_val['tabnet2'], model_dict_gt_val['FT'], model_dict_gt_val['MLP'], model_dict_gt_val['resnet'] ]



# for j in range(0, len(gt_list)):
#     it_prob = probs_list[j]
#     it_pred = preds_list[j]
#     it_gt = gt_list[j]
#
#     print('prob type:', type(it_prob))
#     print(it_prob.shape)
#     print(it_prob[0])
#     print('pred type:', type(it_pred))
#     print(it_pred.shape)
#     print('gt type:', type(it_gt))
#     # print(it_gt.shape)
#     print()
#     print()


probs_stack = np.vstack([ model_dict_probs['xgboost'][:, 1], model_dict_probs['rf'][:, 1], model_dict_probs['tabnet0'][:, 1], model_dict_probs['tabnet1'][:, 1],
                          model_dict_probs['tabnet2'][:, 1], model_dict_probs['FT'], model_dict_probs['MLP'], model_dict_probs['resnet'] ]).T

preds_stack = np.vstack([ model_dict_preds['xgboost'], model_dict_preds['rf'], model_dict_preds['tabnet0'], model_dict_preds['tabnet1'],
                          model_dict_preds['tabnet2'], model_dict_preds['FT'], model_dict_preds['MLP'], model_dict_preds['resnet'] ]).T

gt_stack = np.vstack([ model_dict_gt['xgboost'], model_dict_gt['rf'],model_dict_gt['tabnet0'],model_dict_gt['tabnet1'],
                          model_dict_gt['tabnet2'],model_dict_gt['FT'],model_dict_gt['MLP'],model_dict_gt['resnet'] ]).T



probs_stack_val = np.vstack([ model_dict_probs_val['xgboost'][:, 1], model_dict_probs_val['rf'][:, 1], model_dict_probs_val['tabnet0'][:, 1], model_dict_probs_val['tabnet1'][:, 1],
                          model_dict_probs_val['tabnet2'][:, 1], model_dict_probs_val['FT'], model_dict_probs_val['MLP'], model_dict_probs_val['resnet'] ]).T

preds_stack_val = np.vstack([ model_dict_preds_val['xgboost'], model_dict_preds_val['rf'], model_dict_preds_val['tabnet0'], model_dict_preds_val['tabnet1'],
                          model_dict_preds_val['tabnet2'], model_dict_preds_val['FT'], model_dict_preds_val['MLP'], model_dict_preds_val['resnet'] ]).T

gt_stack_val = np.vstack([ model_dict_gt_val['xgboost'], model_dict_gt_val['rf'],model_dict_gt_val['tabnet0'],model_dict_gt_val['tabnet1'],
                          model_dict_gt_val['tabnet2'],model_dict_gt_val['FT'],model_dict_gt_val['MLP'],model_dict_gt_val['resnet'] ]).T


print()
print()

# check all the outputs and their structures/types
# check GT type and value consistency among all models


n_boot = 1000 #10 # how many times to sample
random_seed = 42

rng = np.random.RandomState(seed=random_seed)

results = {}


auc_list = []
sens_list = []
spec_list = []
ppv_list = []
tp_list = []
fp_list = []
tn_list = []
fn_list = []
f1_list = []
recall_list = []
precision_list = []
prc_auc_list = []



def ci(arr):
    return np.percentile(arr, 2.5), np.percentile(arr, 97.5)


def compute_metrics(y_true, y_pred):
    # y_prob = torch.sigmoid(y_pred).detach().cpu().numpy().flatten()
    # y_bin = (y_prob >= 0.5).astype(int)
    # auc = roc_auc_score(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp)
    return sensitivity, specificity, ppv, tn, fp, fn, tp



def save_results_to_csv(file_name, ensemble_name, auc, sensitivity, specificity, ppv, tn, fp, fn, tp):
    """Append ensemble results to a CSV file."""
    file_exists = os.path.isfile(file_name)
    with open(file_name, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["ensemble", "AUC", "Sensitivity", "Specificity", "PPV", "TN", "FP", "FN", "TP"])
        writer.writerow([ensemble_name, auc, sensitivity, specificity, ppv, tn, fp, fn, tp])


def save_results_to_csv_bootstrap(file_name, ensemble_name, auc, sensitivity, specificity, ppv, tn, fp, fn, tp, auc_ci, sens_ci, spec_ci, ppv_ci):
    """Append ensemble results to a CSV file."""
    file_exists = os.path.isfile(file_name)
    with open(file_name, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["ensemble", "AUC", "AUC CI", "Sensitivity", "Sensitivity CI", "Specificity", "Specificity CI", "PPV", "PPV CI", "TN", "FP", "FN", "TP"])
        writer.writerow([ensemble_name, auc, auc_ci, sensitivity, sens_ci, specificity, spec_ci, ppv, ppv_ci, tn, fp, fn, tp])



# if do a more complex, learning-based ensemble need to do the above for the VAL set to learn on, then eval on test set

if ensemble == 'just_eval':

    for q in range(0, len(preds_list)):
        pl = preds_list[q]
        probl = probs_list[q]
        mlc = model_chosen_list[q]

        if mlc in ['xgboost', 'RF', 'tabnet0', 'tabnet1', 'tabnet2']:
            # pass
            auc = roc_auc_score(model_dict_gt['xgboost'], probl[:, 1])
        else:
            auc = roc_auc_score(model_dict_gt['xgboost'], probl)

        sensitivity, specificity, ppv, tn, fp, fn, tp = compute_metrics(model_dict_gt['xgboost'], pl)
        # file_name, ensemble_name, auc, sensitivity, specificity, ppv, tn, fp, fn, tp
        save_results_to_csv(
            METRICS_CSV + '_' + setting + '_' + mlc + '_confmatsave.csv',
            "just_eval",
            auc,
            sensitivity,
            specificity,
            ppv,
            tn,
            fp,
            fn,
            tp
        )


elif ensemble == 'simple':
    print("\n=== Simple Majority Vote Ensemble ===")

    ensemble_pred = mode(preds_stack, axis=1).mode.flatten()
    ensemble_pred_val = mode(preds_stack_val, axis=1).mode.flatten()

    sensitivity, specificity, ppv, tn, fp, fn, tp = compute_metrics(model_dict_gt['xgboost'], ensemble_pred)

    print('TEST')
    # print(f"AUC: {auc:.4f}")
    print(f"Sensitivity (TPR): {sensitivity:.4f}")
    print(f"Specificity (TNR): {specificity:.4f}")
    print(f"PPV (Precision): {ppv:.4f}")
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print()

    save_results_to_csv(
        METRICS_CSV + '_' + setting + '_TEST_simple.csv',
        "simple",
        0,
        sensitivity,
        specificity,
        ppv,
        tn,
        fp,
        fn,
        tp
    )

    sensitivity, specificity, ppv, tn, fp, fn, tp = compute_metrics(model_dict_gt_val['xgboost'], ensemble_pred_val)

    print('VALIDATION')
    # print(f"AUC: {auc:.4f}")
    print(f"Sensitivity (TPR): {sensitivity:.4f}")
    print(f"Specificity (TNR): {specificity:.4f}")
    print(f"PPV (Precision): {ppv:.4f}")
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print()

    save_results_to_csv(
        METRICS_CSV + '_' + setting + '_VAL_simple.csv',
        "simple",
        0,
        sensitivity,
        specificity,
        ppv,
        tn,
        fp,
        fn,
        tp
    )

elif ensemble == 'log_reg':
    print("\n=== Logistic Regression Stacking Ensemble ===")

    # Use predicted probabilities on validation set as features for meta-model
    X_meta_train = probs_stack_val
    y_meta_train = model_dict_gt_val['xgboost']  # or any model’s GT, since all should match

    # Train meta-model
    meta_model = LogisticRegression(max_iter=1000)
    meta_model.fit(X_meta_train, y_meta_train)

    # Evaluate on test set
    X_meta_test = probs_stack
    y_meta_test = model_dict_gt['xgboost']

    N = len(y_meta_test)

    for i in range(n_boot):
        pred_idx = []
        for l in range(2):  # number of classes? -- mine is 2 not 3 (Ruofan)
            idx_l = np.where(y_meta_test == l)[0]
            pred_idx_l = rng.choice(idx_l, size=idx_l.shape[0], replace=True)
            pred_idx.extend(pred_idx_l)

        pred_idx = np.hstack(pred_idx)

        X_b = X_meta_test[pred_idx]
        y_b = y_meta_test[pred_idx]

        probs_b = meta_model.predict_proba(X_b)[:, 1]
        preds_b = (probs_b > 0.5).astype(int)

        auc_list.append(roc_auc_score(y_b, probs_b))



        f1 = metrics.f1_score(y_b, preds_b)
        f1_list.append(f1)

        recall = metrics.recall_score(y_b, preds_b)
        recall_list.append(recall)

        precision = metrics.precision_score(y_b, preds_b)
        precision_list.append(precision)

        prec_test, recall_test, _ = metrics.precision_recall_curve(y_b, probs_b)
        # Calculate AUC-PRC
        auc_prc_test = metrics.auc(recall_test, prec_test)
        prc_auc_list.append(auc_prc_test)



        sensitivity, specificity, ppv, tn, fp, fn, tp = compute_metrics(y_b, preds_b)

        sens_list.append(sensitivity)
        spec_list.append(specificity)
        ppv_list.append(ppv)
        tp_list.append(tp)
        fp_list.append(fp)
        tn_list.append(tn)
        fn_list.append(fn)


    avg_auc = np.mean(auc_list)
    avg_sensitivity = np.mean(sens_list)
    avg_specificity = np.mean(spec_list)
    avg_ppv = np.mean(ppv_list)
    avg_tp = np.mean(tp_list)
    avg_fp = np.mean(fp_list)
    avg_tn = np.mean(tn_list)
    avg_fn = np.mean(fn_list)

    auc_ci_low, auc_ci_up = ci(auc_list)
    sens_ci_low, sens_ci_up = ci(sens_list)
    spec_ci_low, spec_ci_up = ci(spec_list)
    ppv_ci_low, ppv_ci_up = ci(ppv_list)

    auc_ci = f"[{auc_ci_low:.3f}, {auc_ci_up:.3f}]"
    sens_ci = f"[{sens_ci_low:.3f}, {sens_ci_up:.3f}]"
    spec_ci = f"[{spec_ci_low:.3f}, {spec_ci_up:.3f}]"
    ppv_ci = f"[{ppv_ci_low:.3f}, {ppv_ci_up:.3f}]"

    results['sensitivities'] = sens_list
    results['specificities'] = spec_list
    results['f1s'] = f1_list
    results['roc_aucs'] = auc_list
    results['recalls'] = recall_list
    results['precisions'] = precision_list
    results['prc_aucs'] = prc_auc_list

    df_results = pd.DataFrame(results)
    df_results.to_csv('logs_MH_subset/mlp_mh2pc_ppv_bootsraps_corrected/NEWstatsig_ensemble_log_reg_' + setting + '.csv', index=False)
    # save_file = 'NEWstatsig_' + args.sv

    # meta_probs = meta_model.predict_proba(X_meta_test)[:, 1]
    # meta_preds = (meta_probs > 0.5).astype(int)
    #
    # auc = roc_auc_score(y_meta_test, meta_probs)
    # sensitivity, specificity, ppv, tn, fp, fn, tp = compute_metrics(y_meta_test, meta_preds)

    # print(f"AUC: {auc:.4f}")
    # print(f"Sensitivity (TPR): {sensitivity:.4f}")
    # print(f"Specificity (TNR): {specificity:.4f}")
    # print(f"PPV (Precision): {ppv:.4f}")
    # print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    # print()

    # save_results_to_csv(
    #     METRICS_CSV + '_' + setting + '_TEST_log_reg.csv',
    #     "log_reg",
    #     auc,
    #     sensitivity,
    #     specificity,
    #     ppv,
    #     tn,
    #     fp,
    #     fn,
    #     tp
    # )

    save_results_to_csv_bootstrap(
        METRICS_CSV + '_' + setting + '_TEST_log_reg_bootstraps.csv',
        "log_reg",
        avg_auc,
        avg_sensitivity,
        avg_specificity,
        avg_ppv,
        avg_tn,
        avg_fp,
        avg_fn,
        avg_tp,
        auc_ci,
        sens_ci,
        spec_ci,
        ppv_ci
    )

elif ensemble == 'weighted_avg':
    print("\n=== Weighted Average Ensemble ===")

    # You can set weights manually or compute from validation AUCs
    val_aucs = []
    for j in range(probs_stack_val.shape[1]):
        auc_j = roc_auc_score(model_dict_gt_val['xgboost'], probs_stack_val[:, j])
        val_aucs.append(auc_j)
    weights = np.array(val_aucs) / np.sum(val_aucs)

    print("Validation AUCs:", np.round(val_aucs, 4))
    print("Normalized weights:", np.round(weights, 4))
    print()

    # Weighted average probabilities
    weighted_probs = np.average(probs_stack, axis=1, weights=weights)
    weighted_preds = (weighted_probs > 0.5).astype(int)

    auc = roc_auc_score(model_dict_gt['xgboost'], weighted_probs)
    sensitivity, specificity, ppv, tn, fp, fn, tp = compute_metrics(model_dict_gt['xgboost'], weighted_preds)

    print(f"AUC: {auc:.4f}")
    print(f"Sensitivity (TPR): {sensitivity:.4f}")
    print(f"Specificity (TNR): {specificity:.4f}")
    print(f"PPV (Precision): {ppv:.4f}")
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print()

    save_results_to_csv(
        METRICS_CSV + '_' + setting + '_TEST_weighted_avg.csv',
        "weighted_avg",
        auc,
        sensitivity,
        specificity,
        ppv,
        tn,
        fp,
        fn,
        tp
    )



# Convert all to numpy arrays just in case
gt_list = [np.array(gt).flatten() for gt in gt_list]

# 1️⃣ Check if all are equal to the first one
reference = gt_list[0]
all_equal = all(np.array_equal(reference, gt) for gt in gt_list)

print()
print("All ground truths identical:", all_equal)

# 2️⃣ If not identical, show where they differ
if not all_equal:
    for i, gt in enumerate(gt_list):
        if not np.array_equal(reference, gt):
            diff_idx = np.where(reference != gt)[0]
            print(f"\n⚠️ Difference found in gt_list[{i}] ({len(diff_idx)} mismatches)")
            print(f"First 10 differing indices: {diff_idx[:10]}")
            print(f"Reference values: {reference[diff_idx[:10]]}")
            print(f"Different values: {gt[diff_idx[:10]]}")


print()
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")



print()
print()
print('-----------------------------------------------------------------------')
print('-----------------------------------------------------------------------')
print('-----------------------------------------------------------------------')
print('-----------------------------------------------------------------------')
print('-----------------------------------------------------------------------')
print('-----------------------------------------------------------------------')
print()
print()

print('running next ensemble approach...')
print()