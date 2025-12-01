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

# import sys
# sys.path.append("/path/to/rtdl-revisiting-models/package")
# import rtdl_revisiting_models as rtdl

from rtdl_revisiting_models import MLP, ResNet, FTTransformer

# import rtdl
import time
import os
import random
import argparse

# =====================
# USER CONFIG
# =====================
# DATA_PATH = "data.csv"   # your tabular dataset
# TARGET_COL = "target"     # name of your binary label column
# BINARY_COLS = ["bin1", "bin2", "bin3"]  # list of binary/categorical columns
# NUMERIC_COLS = ["num1", "num2", "num3"]  # list of numeric columns

parser = argparse.ArgumentParser(description='ResNet')
parser.add_argument('--setting', default='pc2pc', type=str, help='setting task: pc2pc, pc2nonpc, nonpc2pc, nonpc2nonpc')
parser.add_argument('--metrics_csv', default='ftt_metrics.csv', type=str, help='metrics save file')
parser.add_argument('--n_trials', default=30, type=int, help='number optuna trials')
parser.add_argument('--epochs_fm', '--epochs_final_model', default=100, type=int, help='number epochs for final model')
parser.add_argument('--epochs_op', '--epochs_optuna', default=50, type=int, help='number epochs for optuna tuning')
args = parser.parse_args()

METRICS_CSV = args.metrics_csv #"ftt_metrics.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_TRIALS = args.n_trials #30  # optuna trials


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


# # Convert to tensors
# def to_tensor(data):
#     return torch.tensor(data.values, dtype=torch.float32)

# pc: (X_train, y_train) (X_val, y_val) (X_test, y_test)
# non-pc: (X_train_non_pc, y_train_non_pc) (X_val_non_pc, y_val_non_pc) (X_test_non_pc, y_test_non_pc)

setting = args.setting #'pc2pc'
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

# X_train = torch.tensor(X_train.values, dtype=torch.float32).to(DEVICE)
# X_val   = torch.tensor(X_val.values, dtype=torch.float32).to(DEVICE)
# X_test   = torch.tensor(X_test.values, dtype=torch.float32).to(DEVICE)
#
# y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(DEVICE)
# y_val_t   = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1).to(DEVICE)
# y_test_t = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1).to(DEVICE)


x_cat_train_ohe = [
    F.one_hot(cat_column, c)
    for cat_column, c in zip(X_train_cat.T, cat_cardinalities)
]
X_train = torch.column_stack([X_train_num] + x_cat_train_ohe)

print('X_train shape:', X_train.shape)
print()

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

# -----------------------
# METRICS
# -----------------------
def compute_metrics(y_true, y_pred):
    y_prob = torch.sigmoid(y_pred).detach().cpu().numpy().flatten()
    y_bin = (y_prob >= 0.5).astype(int)
    auc = roc_auc_score(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_bin).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp)
    return auc, sensitivity, specificity, ppv, tn, fp, fn, tp, y_prob


def compute_bootstrap(y_true, y_pred):
    times = 1000  # 10 # how many times to sample
    random_seed = 42

    rng = np.random.RandomState(seed=random_seed)

    results = {}

    sensitivities = []
    specificities = []
    f1s = []
    roc_aucs = []
    recalls = []
    precisions = []
    prc_aucs = []
    tps = []
    tns = []
    fps = []
    fns = []

    for i in range(times):
        pred_idx = []
        for l in range(2):  # number of classes? -- mine is 2 not 3 (Ruofan)
            idx_l = np.where(y_true.values == l)[0]
            pred_idx_l = rng.choice(idx_l, size=idx_l.shape[0], replace=True)
            pred_idx.extend(pred_idx_l)

        pred_idx = np.hstack(pred_idx)

        # print(i)
        # print('pred_idx:', pred_idx)
        # print('GT pred_idx:', y_true.values[pred_idx])
        # print()

        y_ = y_true.values[pred_idx]
        y_p = y_pred[pred_idx]

        auc, sensitivity, specificity, ppv, tn, fp, fn, tp, y_prob = compute_metrics(y_, y_p)

        roc_aucs.append(auc)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        precisions.append(ppv)
        tns.append(tn)
        fps.append(fp)
        fns.append(fn)
        tps.append(tp)

    ci_lower_sensitivities = np.percentile(sensitivities, 2.5)
    ci_upper_sensitivities = np.percentile(sensitivities, 97.5)

    ci_lower_specificities = np.percentile(specificities, 2.5)
    ci_upper_specificities = np.percentile(specificities, 97.5)

    ci_lower_roc_aucs = np.percentile(roc_aucs, 2.5)
    ci_upper_roc_aucs = np.percentile(roc_aucs, 97.5)

    ci_lower_precisions = np.percentile(precisions, 2.5)
    ci_upper_precisions = np.percentile(precisions, 97.5)

    results['sensitivities'] = sensitivities
    results['specificities'] = specificities
    results['roc_aucs'] = roc_aucs
    results['precisions'] = precisions

    results2 = {}

    results2['precision'] = np.mean(precisions)
    results2['roc_auc'] = np.mean(roc_aucs)
    results2['spec'] = np.mean(specificities)
    results2['sens'] = np.mean(sensitivities)

    results2['precision 95%CI'] = (ci_lower_precisions, ci_upper_precisions)
    results2['roc_auc 95%CI:'] = (ci_lower_roc_aucs, ci_upper_roc_aucs)
    results2['spec 95%CI'] = (ci_lower_specificities, ci_upper_specificities)
    results2['sens 95%CI'] = (ci_lower_sensitivities, ci_upper_sensitivities)

    results2['TP_avg'] = np.mean(tps)
    results2['FP_avg'] = np.mean(fps)
    results2['FN_avg'] = np.mean(fns)
    results2['TN_avg'] = np.mean(tns)

    return results, results2


# -----------------------
# TRAINING FUNCTION FOR OPTUNA
# -----------------------
def train_one_trial(trial):
    d_blocks = trial.suggest_categorical("d_block", [100, 192, 256])
    n_blocks = trial.suggest_int("n_blocks", 2, 4)
    dropouts1 = trial.suggest_float("dropout1", 0.0, 0.2)
    dropouts2 = trial.suggest_float("dropout2", 0.0, 0.2)
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    model = ResNet(
        d_in=len(NUMERIC_COLS) + sum(cat_cardinalities),
        d_out=1,
        n_blocks=n_blocks,
        d_block=d_blocks,
        d_hidden=None,
        d_hidden_multiplier=2.0,
        dropout1=dropouts1,
        dropout2=dropouts2,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    n_epochs = args.epochs_op #50
    best_val_auc = 0

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(X_train.size(0))
        for i in range(0, X_train.size(0), batch_size):
            idx = perm[i:i+batch_size]
            xb = X_train[idx]
            yb = y_train_t[idx]
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            preds_val = model(X_val)
            auc, _, _, _, _, _, _, _, _ = compute_metrics(y_val, preds_val)
        trial.report(auc, epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()
        if auc > best_val_auc:
            best_val_auc = auc

    return best_val_auc

# -----------------------
# OPTUNA STUDY
# -----------------------
study = optuna.create_study(direction="maximize")
study.optimize(train_one_trial, n_trials=N_TRIALS)
best_params = study.best_params
print("\nBest hyperparameters:", best_params)
print()

# -----------------------
# FINAL TRAIN & EVAL
# -----------------------
final_model = ResNet(
        d_in=len(NUMERIC_COLS) + sum(cat_cardinalities),
        d_out=1,
        n_blocks=best_params["n_blocks"],
        d_block=best_params["d_block"],
        d_hidden=None,
        d_hidden_multiplier=2.0,
        dropout1=best_params["dropout1"],
        dropout2=best_params["dropout2"],
    ).to(DEVICE)


optimizer = torch.optim.AdamW(final_model.parameters(), lr=best_params["lr"])
criterion = nn.BCEWithLogitsLoss()
batch_size = best_params["batch_size"]
n_epochs = args.epochs_fm #100

metrics_log = []

for epoch in tqdm(range(n_epochs), desc="Training final model"):
    final_model.train()
    perm = torch.randperm(X_train.size(0))
    for i in range(0, X_train.size(0), batch_size):
        idx = perm[i:i+batch_size]
        xb = X_train[idx]
        yb = y_train_t[idx]
        optimizer.zero_grad()
        preds = final_model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

    final_model.eval()
    # make this the test data here
    with torch.no_grad():
        # preds_val = final_model(X_val_num, X_val_cat)
        # auc, sens, spec, ppv = compute_metrics(y_val, preds_val)
        preds_test = final_model(X_test)
        auc, sens, spec, ppv, tn, fp, fn, tp, y_prob = compute_metrics(y_test, preds_test)


    metrics_log.append({
        "epoch": epoch + 1,
        "AUC": auc,
        "Sensitivity": sens,
        "Specificity": spec,
        "PPV": ppv,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "y_prob": y_prob,
    })

# print()
# print('-------------------------------------------------')
preds_test = final_model(X_test)
# print(y_test[0:5])
# print(type(y_test))
# print()
# print(preds_test[0:5])
# print(type(preds_test))
results, results2 = compute_bootstrap(y_test, preds_test)
# print('-------------------------------------------------')
# print()

# Convert dictionary to DataFrame
df = pd.DataFrame(results)
# df = pd.DataFrame.from_records(results)
df2 = pd.DataFrame(results2)

# Write DataFrame to CSV
df.to_csv('logs_MH_subset/new_models/NEW_statsig/ResNet_tabular_' + setting + '.csv', index=False)

df2.to_csv('logs_MH_subset/new_models//NEW_statsig/ResNet_tabular_' + setting + '_summary.csv', index=False)

# Save metrics
pd.DataFrame(metrics_log).to_csv('logs_MH_subset/new_models/saved_metric_csvs/'+METRICS_CSV, index=False)
print(f"\nMetrics logged to {METRICS_CSV}")
print()

torch.save(final_model, 'logs_MH_subset/new_models/saved_models/ResNet_train1_test1_'+setting+".pth")
# final_model.save_model('logs_MH_subset/new_models/saved_models/MLP_train1_test1_'+setting)
print("model saved")
print()

end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")


# d_out = 1
# model = ResNet(
#     d_in=len(NUMERIC_COLS) + sum(cat_cardinalities),
#     d_out=d_out,
#     n_blocks=2,
#     d_block=192,
#     d_hidden=None,
#     d_hidden_multiplier=2.0,
#     dropout1=0.15,
#     dropout2=0.0,
# )
# y_pred = model(X_train)
# assert y_pred.shape == (batch_size, d_out)