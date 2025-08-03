import pandas as pd
import numpy as np
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn import metrics
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
# torch.manual_seed(SEED)


# pth_MHRN_OP_new = 'MHRN4_with_zip_income_college_binarized_new.csv'
pth_MHRN_OP_new = 'combined_pc_mh_data.csv'

# this script is for PC coeff from:  https://github.com/MHResearchNetwork/srpm-model
pth_pc_coeff_df = 'Primary care Coefficients.xlsx'


use_0s = True #True (True = pre-trained with 102 features only, False pre-trained with 102 features and then use the remaining trained from scratch 218 features)
use_balanced = True # doesnt matter here, no training

# compare my cols to MHRN 102 cols
mhrn_pc_coeff_df = pd.read_excel(pth_pc_coeff_df)
our_data_df1 = pd.read_csv(pth_MHRN_OP_new)

# our_data_df = our_data_df1

our_data_df = our_data_df1[our_data_df1['age'] >= 13]
min_value = our_data_df['age'].min()
print()
print('-----------------------')
print('MIN VALUE:', min_value)
print('-----------------------')
print()

# columns_to_drop_idx = ['antidep_rx_pre3m_idx', 'antidep_rx_pre1y_cumulative_idx', 'antidep_rx_pre5y_cumulative_idx', 'benzo_rx_pre3m_idx', 'benzo_rx_pre1y_cumulative_idx', 'benzo_rx_pre5y_cumulative_idx', 'hypno_rx_pre3m_idx', 'hypno_rx_pre1y_cumulative_idx', 'hypno_rx_pre5y_cumulative_idx', 'sga_rx_pre3m_idx', 'sga_rx_pre1y_cumulative_idx', 'sga_rx_pre5y_cumulative_idx', 'mh_ip_pre3m_idx', 'mh_ip_pre1y_cumulative_idx', 'mh_ip_pre5y_cumulative_idx', 'mh_op_pre3m_idx', 'mh_op_pre1y_cumulative_idx', 'mh_op_pre5y_cumulative_idx', 'mh_ed_pre3m_idx', 'mh_ed_pre1y_cumulative_idx', 'mh_ed_pre5y_cumulative_idx', 'any_sui_att_pre3m_idx', 'any_sui_att_pre1y_cumulative_idx', 'any_sui_att_pre5y_cumulative_idx', 'any_sui_att_pre5y_cumulative_idx_a', 'any_sui_att_pre5y_cumulative_idx_f', 'lvi_sui_att_pre3m_idx', 'lvi_sui_att_pre1y_cumulative_idx', 'lvi_sui_att_pre5y_cumulative_idx', 'ovi_sui_att_pre3m_idx', 'ovi_sui_att_pre1y_cumulative_idx', 'ovi_sui_att_pre5y_cumulative_idx', 'any_inj_poi_pre3m_idx', 'any_inj_poi_pre1y_cumulative_idx', 'any_inj_poi_pre5y_cumulative_idx']
# our_data_df = our_data_df.drop(columns=columns_to_drop_idx)

columns_to_drop_idx = ['antidep_rx_pre3m_idx', 'antidep_rx_pre1y_cumulative_idx', 'antidep_rx_pre5y_cumulative_idx', 'benzo_rx_pre3m_idx', 'benzo_rx_pre1y_cumulative_idx', 'benzo_rx_pre5y_cumulative_idx', 'hypno_rx_pre3m_idx', 'hypno_rx_pre1y_cumulative_idx', 'hypno_rx_pre5y_cumulative_idx', 'sga_rx_pre3m_idx', 'sga_rx_pre1y_cumulative_idx', 'sga_rx_pre5y_cumulative_idx', 'mh_ip_pre3m_idx', 'mh_ip_pre1y_cumulative_idx', 'mh_ip_pre5y_cumulative_idx', 'mh_op_pre3m_idx', 'mh_op_pre1y_cumulative_idx', 'mh_op_pre5y_cumulative_idx', 'mh_ed_pre3m_idx', 'mh_ed_pre1y_cumulative_idx', 'mh_ed_pre5y_cumulative_idx', 'any_sui_att_pre3m_idx', 'any_sui_att_pre1y_cumulative_idx', 'any_sui_att_pre5y_cumulative_idx', 'any_sui_att_pre5y_cumulative_idx_a', 'any_sui_att_pre5y_cumulative_idx_f', 'lvi_sui_att_pre3m_idx', 'lvi_sui_att_pre1y_cumulative_idx', 'lvi_sui_att_pre5y_cumulative_idx', 'ovi_sui_att_pre3m_idx', 'ovi_sui_att_pre1y_cumulative_idx', 'ovi_sui_att_pre5y_cumulative_idx', 'any_inj_poi_pre3m_idx', 'any_inj_poi_pre1y_cumulative_idx', 'any_inj_poi_pre5y_cumulative_idx']
our_data_df = our_data_df.drop(columns=columns_to_drop_idx)


our_data_df = our_data_df.drop(columns=["income", "college", "hhld_inc_It40k", "coll_deg_It25p"])

# fix dropped cols above
our_data_df.rename(columns={'hhld_inc_lt40k_NJ': 'hhld_inc_It40k', 'coll_deg_lt25p_NJ': 'coll_deg_It25p'}, inplace=True)



# mhrn_pc_coeff_df = pd.read_excel(pth_pc_coeff_df)
# our_data_df = pd.read_csv(pth_MHRN_OP_new)

# load MHRN pc coeff info excel

# print(mhrn_pc_coeff_df)

# col names
mhrn_col_names = mhrn_pc_coeff_df['event90'].tolist()
mhrn_col_names.remove('-------------------------------')
mhrn_col_names.remove('_cons')
mhrn_col_names.pop(-1)
mhrn_col_names.pop(-1)
mhrn_col_names.pop(-1)
print('mhrn_col_names:')
print(mhrn_col_names)
print(len(mhrn_col_names))
print()

# coefficients
mhrn_coeff_int = mhrn_pc_coeff_df['Coef.'].tolist()
mhrn_coeff_int.remove('------------')
mhrn_int = mhrn_coeff_int[0]
mhrn_coeff_int.pop(0)
mhrn_coeff_int.pop(-1)
mhrn_coeff_int.pop(-1)
mhrn_coeff_int.pop(-1)
mhrn_coeff = mhrn_coeff_int
print('mhrn_coeff:')
print(mhrn_coeff)
print(len(mhrn_coeff))
print('mhrn_int:')
print(mhrn_int)
print()

# load our data info
our_data_df = our_data_df.drop(columns=["PRIMARY_CARE_VISIT","person_id","event30","event90","death30","death90","visit_mh"])
# print(our_data_df)
our_col_names = our_data_df.columns.tolist()
print('our_col_names:')
print(our_col_names)
print(len(our_col_names))
print()

def create_dictionary(keys, values):
    # Check if the lengths of the lists are the same
    if len(keys) != len(values):
        raise ValueError("The lengths of the lists must be the same")

    # Create a dictionary using zip
    result_dict = dict(zip(keys, values))
    return result_dict


mhrn_dict = create_dictionary(mhrn_col_names, mhrn_coeff)
print('mhrn_dict:')
print(mhrn_dict)
print(len(mhrn_dict))
print()

our_dict = {key: 0 for key in our_col_names}
print('our_dict:')
print(our_dict)
print(len(our_dict))
print()

joined_dict = {**our_dict, **mhrn_dict}
print('joined_dict:')
print(joined_dict)
print(len(joined_dict))
print()

# correct mismatched names between mhrn and our data 
our_problem_keys_names = ['highdedectible', 'lvi_sui_att_pre5y', 'raceAsian', 'raceBlack', 'coll_deg_It25p',
                      'raceBlack_de', 'raceBlack_an', 'raceUN_bi', 'raceAsian_8', 'raceIN_8']

mhrn_problem_key_names = ['highdeductible', 'lvi_sui_att_pre5y_cumulative', 'raceasian', 'raceblack', 'coll_deg_lt25p',
                          'raceblack_de', 'raceblack_an', 'raceun_bi', 'raceasian_8', 'racein_8']

for k in range(0, len(our_problem_keys_names)):
    vv = joined_dict[mhrn_problem_key_names[k]]
    joined_dict[our_problem_keys_names[k]] = vv


for kk in mhrn_problem_key_names:
    del joined_dict[kk]

print('joined_dict:')
print(joined_dict)
print(len(joined_dict))
print()

# Specify the output CSV file path
mhrn_dict_csv_file = 'mhrn_dict_NEW2.csv'
our_dict_csv_file = 'our_dict_NEW2.csv'
joined_dict_csv_file = 'joined_dict_NEW2.csv'

# Write the dictionary to the CSV file
def save_dict_to_csv(csv_file, my_dict):
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        for key, value in my_dict.items():
            writer.writerow([key, value])

# Save dict to csv
# save_dict_to_csv(mhrn_dict_csv_file, mhrn_dict)
# save_dict_to_csv(our_dict_csv_file, our_dict)
# save_dict_to_csv(joined_dict_csv_file, joined_dict)


#################
#################
#################


data1 = pd.read_csv(pth_MHRN_OP_new)

# data = our_data_df

data = data1[data1['age'] >= 13]
min_value = data['age'].min()
print()
print('-----------------------')
print('MIN VALUE:', min_value)
print('-----------------------')
print()

columns_to_drop_idx = ['antidep_rx_pre3m_idx', 'antidep_rx_pre1y_cumulative_idx', 'antidep_rx_pre5y_cumulative_idx', 'benzo_rx_pre3m_idx', 'benzo_rx_pre1y_cumulative_idx', 'benzo_rx_pre5y_cumulative_idx', 'hypno_rx_pre3m_idx', 'hypno_rx_pre1y_cumulative_idx', 'hypno_rx_pre5y_cumulative_idx', 'sga_rx_pre3m_idx', 'sga_rx_pre1y_cumulative_idx', 'sga_rx_pre5y_cumulative_idx', 'mh_ip_pre3m_idx', 'mh_ip_pre1y_cumulative_idx', 'mh_ip_pre5y_cumulative_idx', 'mh_op_pre3m_idx', 'mh_op_pre1y_cumulative_idx', 'mh_op_pre5y_cumulative_idx', 'mh_ed_pre3m_idx', 'mh_ed_pre1y_cumulative_idx', 'mh_ed_pre5y_cumulative_idx', 'any_sui_att_pre3m_idx', 'any_sui_att_pre1y_cumulative_idx', 'any_sui_att_pre5y_cumulative_idx', 'any_sui_att_pre5y_cumulative_idx_a', 'any_sui_att_pre5y_cumulative_idx_f', 'lvi_sui_att_pre3m_idx', 'lvi_sui_att_pre1y_cumulative_idx', 'lvi_sui_att_pre5y_cumulative_idx', 'ovi_sui_att_pre3m_idx', 'ovi_sui_att_pre1y_cumulative_idx', 'ovi_sui_att_pre5y_cumulative_idx', 'any_inj_poi_pre3m_idx', 'any_inj_poi_pre1y_cumulative_idx', 'any_inj_poi_pre5y_cumulative_idx']
data = data.drop(columns=columns_to_drop_idx)


data["event90"] = data["event90"].fillna(value=0)



data = data.drop(columns=["income", "college", "hhld_inc_It40k", "coll_deg_It25p"])

data.rename(columns={'hhld_inc_lt40k_NJ': 'hhld_inc_It40k', 'coll_deg_lt25p_NJ': 'coll_deg_It25p'}, inplace=True)



pc_data = data[data["PRIMARY_CARE_VISIT"] == 1]

y = pc_data["event90"]

# pc_columns_to_scale = ["age","days_since_prev","charlson_score","charlson_a","dep_dx_pre5y_cumulative_a","anx_dx_pre5y_cumulative_a","bip_dx_pre5y_cumulative_a","sch_dx_pre5y_cumulative_a","phqnumber90","phqnumber183","phqnumber365","phq8_index_score_calc_f","raceAsian_8","raceIN_8","hispanic_8","age_8","q9_0_a","q9_1_8","q9_2_8","q9_3_8","q9_1_c","q9_2_c","q9_3_c","any_sui_att_pre5y_cumulative_a","any_sui_att_pre5y_cumulative_8","any_sui_att_pre5y_cumulative_c"]
#
# count = 0
# for i in pc_columns_to_scale:
#     if i in pc_data.columns:
#         count += 1
#     else:
#         print(i)
# print(count)
#
# pc_data[pc_columns_to_scale] = scale(pc_data[pc_columns_to_scale])

pc_data = pc_data.drop(columns=["PRIMARY_CARE_VISIT","person_id","event30","event90","death30","death90","visit_mh"])


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

missing_columns = pc_data.columns[pc_data.isnull().any()]

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

        # pc_data = pc_data.drop(columns=[i]) # if all rows are empty for a column, drop that column
        pc_data[i].fillna(value=-1, inplace=True) # for this pretrianed model, fill the fully empty with unknown -1, should just be enrolled column

        # print(sum(data[i].isnull()))
        print("Empty column:", i)
    else:
        pc_data[i].fillna(value=-1, inplace=True) # # if only some of rows have missing value for a column, then fill that with -1, for num or bin cols






# ### Splitting data into train and validation

# In[24]:
train_X, X_test, train_y, y_test = train_test_split(pc_data, y, test_size=0.35, random_state=42, stratify = y)


# In[25]:
X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, test_size=0.35, random_state=42, stratify = train_y)


print('ORIGINAL MODEL')
# can also try 'BALANCED'
if use_balanced == True:
    print('Using Balancing in Logistic Regression')
    # clf = LogisticRegression(penalty='l1', tol=0.0007524244681796938, C=0.35, solver='liblinear', fit_intercept=True, class_weight="balanced") # class_weight="balanced"
    # clf = LogisticRegression(penalty='l1', tol=0.0009886306860051027, C=9.4, solver='saga', fit_intercept=False,
    #                          class_weight="balanced") # tuned for balancing
    clf = LogisticRegression(penalty='l1', tol=0.006569428962904122, C=0.2, solver='liblinear', fit_intercept=True,
                             class_weight="balanced") # also tuned for NOT balancing (unfortunately)
else:
    print('Not Using Balancing in Logistic Regression')
    # clf = LogisticRegression(penalty='l1', tol=0.0007524244681796938, C=0.35, solver='liblinear', fit_intercept=True) # tuned for not balancing
    clf = LogisticRegression(penalty='l1', tol=0.006569428962904122, C=0.2, solver='liblinear', fit_intercept=True) # tuned for not balancing
clf.fit(X_train, y_train)
preds_y = clf.predict(X_test)



if use_0s == False:
    print('use_0s is FALSE')
    print()
    print(type(clf.coef_.tolist()))
    print(clf.coef_.tolist()[0])
    our_dict = create_dictionary(our_col_names, clf.coef_.tolist()[0])
    joined_dict = {**our_dict, **mhrn_dict}

    for k in range(0, len(our_problem_keys_names)):
        vv = joined_dict[mhrn_problem_key_names[k]]
        joined_dict[our_problem_keys_names[k]] = vv

    for kk in mhrn_problem_key_names:
        del joined_dict[kk]

# Print default coefficients and intercept
print("Default Coefficients:", clf.coef_)
print("Default Intercept:", clf.intercept_)
print("Shape of Coefficients:", clf.coef_.shape)
print("Shape of intercept:", clf.coef_.shape)
print()

print('Training accuracy:', clf.score(X_train, y_train))
print('Test accuracy:', clf.score(X_test, y_test))
print('Training accuracy:', accuracy_score(y_train, clf.predict(X_train)))
print('Test accuracy:', accuracy_score(y_test, preds_y))
print("ROC AUC is: ",metrics.roc_auc_score(y_test, clf.predict_proba(X_test)[:,-1]))
print("F1 score is: ", f1_score(y_test, clf.predict(X_test)))
print("Precision score is: ", metrics.precision_score(y_test, clf.predict(X_test)))
print("Recall score is: ", metrics.recall_score(y_test, clf.predict(X_test)))
print(metrics.classification_report(y_test, preds_y))
tn, fp, fn, tp = metrics.confusion_matrix(y_test, preds_y).ravel()
specificity = tn / (tn+fp)
print("Specificity score is: ", specificity)
sensitivity = tp / (tp+fn)
print("Sensitivity score is: ", sensitivity)
print()
print()

print('NEW MODEL:')
new_model = clf

joined_dict_vals = list(joined_dict.values())
coef_arr1 = np.array(joined_dict_vals)
int_array1 = np.array(mhrn_int)
coef_arr = coef_arr1.reshape(1,-1)
int_array = int_array1.reshape(1,-1)

new_model.coef_ = coef_arr
new_model.intercept_ = int_array

# Print new_model coefficients and intercept
print("Default Coefficients:", new_model.coef_)
print("Default Intercept:", new_model.intercept_)
print("Shape of Coefficients:", new_model.coef_.shape)
print("Shape of intercept:", new_model.coef_.shape)
print()

preds_new_model = new_model.predict(X_test)
print(preds_new_model)

# joined_dict_vals = list(joined_dict.values())
# print(type(joined_dict_vals))
# # print(joined_dict_vals)
# coef_arr = np.array(joined_dict_vals)
# print(coef_arr)
# int_array = np.array(mhrn_int)
# print('int')
# print(int_array)
# print(type(int_array))


# print('Training accuracy:', preds_new_model.score(X_train, y_train))
# print('Test accuracy:', preds_new_model.score(X_test, y_test))
print('Training accuracy:', accuracy_score(y_train, new_model.predict(X_train)))
print('Test accuracy:', accuracy_score(y_test, preds_new_model))
print("ROC AUC is: ",metrics.roc_auc_score(y_test, new_model.predict_proba(X_test)[:,-1]))
print("F1 score is: ", f1_score(y_test, new_model.predict(X_test)))
print("Precision score is: ", metrics.precision_score(y_test, new_model.predict(X_test)))
print("Recall score is: ", metrics.recall_score(y_test, new_model.predict(X_test)))
print(metrics.classification_report(y_test, preds_new_model))
tn, fp, fn, tp = metrics.confusion_matrix(y_test, preds_new_model).ravel()
specificity = tn / (tn+fp)
print("Specificity score is: ", specificity)
sensitivity = tp / (tp+fn)
print("Sensitivity score is: ", sensitivity)
print()
print()


# mhrn_pc_coeff = np.array([[-0.0177708,0.2800567,0,-0.1544488,-0.169057,0,0.7085663,0,-0.01017,0,0,0.2015132,
#                         -0.0605431,0,0,0.2860064,0,-0.1090385,0,0,-0.0003243,-0.2056298,-0.4051398,-0.0878258,
#                         0,-0.2380795,0,-0.1424411,0,0,0,0,0,0,0,-0.1410835,0,0,0,0,0,-0.1626887,0,0,0,0,0,0,
#                         -0.8148915,-0.0480855,0.1362236,0.0881055,0,0.340881,0,0,0.1163377,
#                         0,0.0725226,0,0.1072703,0.1582301,0.113767,0.1660328,0.2134675,0.089926,0.168969,-0.1065041,
#                         0.2589882,0.2320369,0.0835463,0.4740714,0.4714488,1.668482,0,0,0.1475331,0,0,0,0.1329157,
#                         0.2464229,-0.0830589,0,0,0,0,0,0.0223427,-0.0997792,0.0018103,0,0,0,0.1381357,0,0,0,0,0,
#                         -0.1207415,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.1652086,-0.029999,0,0,0,0,0,-0.3205841,0,0,0,0,0,0,0,0,0,
#                         -0.0356545,0.0055312,0,0,0,0,0,0,0,0.0293979,0,0,0,0,0,0,0,-0.0217082,0,0,0.1220448,-0.0625263,
#                         0.2185043,0.1852716,0.4865901,0,-0.1041901,0,0,0,0,0,0,-0.1074305,0,0,0,0,0,0,0,0,0,0,
#                         0.1497073,0.1359814,0,0,0,0,0,0,0,1.718887,0,-0.0119109,0.3794353,0,0.0013635,0.1141705,
#                         0.1409446,0.005867,0,0,-0.0005691,0.2136741,0.0835573,0,0,0.443662,0.5796565,1.295697,0.1123629,
#                         0.2631496,0.1512429,0,0,-0.0022386,0,0.0202408,0,0,0.0063296,0,0,0.0069394,0.0004031,0,0,0,0,0,0,0,0,
#                         0.0049376,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0141446,0.0296125,
#                         0.0542913,0,0.0594748,-0.0316618,-0.218713,-0.5715746,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
#                         -0.0159464,-0.0143564,-0.995897,-0.4112843,-0.2070626,0.3432375,-0.2630062,-0.4859498,-0.1867706,
#                         0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
#
# print(len(mhrn_pc_coeff[0]))

