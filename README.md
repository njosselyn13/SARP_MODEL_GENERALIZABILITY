# Model_Generalizability_Across_Clinical_Settings_SARP

# Assessing Model Generalizability Across Clinical Settings in Suicide Attempt Risk Prediction

Corresponding Authors: Edwin D. Boudreaux, Feifan Liu

Contact: Edwin.Boudreaux [at] umassmed [dot] edu, feifan.liu [at] umassmed [dot] edu

Nicholas Josselyn

Contact: njjosselyn [at] wpi [dot] edu

Link to paper: TBA

Nicholas Josselyn, Sahil Sawant, Rachel Davis-Martin, Elke Rundensteiner, Ben S. Gerber, Bo Wang, Anthony Rothschild, Emmanuel Agu, Edwin D. Boudreaux, Feifan Liu. "Assessing Model Generalizability Across Clinical Settings in Suicide Attempt Risk Prediction". In submission.

## Introduction
This code was used to produce the results in the above paper title. Included here will be:

- All machine and deep learning model codes (logistic regression, Random Forest, XGBoost, TabNet, TabNet+pretraining)
- Training scripts
- Optuna tuning
- Bootstrap sampling and 95% confidence intervals code (test set)
- SHAP feature analysis
- Fairness analysis
- Sample bash scripts for running (.sh files)
- Environment setup
- Supplementary tables and figures (PDF)
- Citation information


## Environment Setup

In this work, a miniconda environment was setup on a remote compute cluster running Linux. Experiments were all run on CPU. The codes can be updated by users to use GPU if desired. 

Python version: 3.8.18

First, download miniconda and move download to compute cluster if applicable: 
- Download miniconda: https://docs.conda.io/en/latest/miniconda.html
Then, run the following commands in order:
- conda create -n my_env1
- conda activate "path to environment" (e.x. /home/username/miniconda3/envs/my_env1/)
- conda install python=3.8.18
- pip install -r requirements_sap.txt

## Model codes and analysis

In this work we use Logistic Regression (pre-trained and from scratch), Random Forest, XGBoost, TabNet, and TabNet+pretraining models. As explained in the paper, we run MHRN (logistic regression) model generalizability experiments using pre-trained MHRN group coefficients compared to training from scratch on our dataset. We then run a set of comparitive experiments of ML and DL models for all source-target transfer tasks of primary care (PC) and non-primary care (Non-PC) healthcare settings. These are all in-domain (pc2pc, nonpc2nonpc), out-of-domain (pc2nonpc, nonpc2pc), and combined (pc+nonpc2pc, pc+nonpc2nonpc). We then do feature importance analysis (SHAP) and a fairness study. All codes and descriptions are listed below, Optuna hyperparameter tuning is done for all relevant experiments. 

MHRN Generalizability experiments:

- tune_log_reg_102feats.py: train logistic regression from scratch on 102 features it (tuning with and without balancing separately)
- tune_log_reg.py: train logistic regression from scratch on 320 features it (tuning with and without balancing separately)

Machine vs Deep Learning experiments:

- XGboost_entire_data.py: Train XGBoost PC+NonPC, Test PC or NonPC
- XGboost_train1_test1.py: Train XGBoost PC or NonPC, Test PC or NonPC
- RandomForest_1to1.py​: Train Random Forest PC or NonPC, Test PC or NonPC​
- MHRN_TabNet_Train_entire_data_Nick.py​:
  - Random Forest training: Train Random Forest PC+NonPC, Test PC or NonPC
  - TabNet training: Train TabNet-0 PC+NonPC, Test PC or NonPC
- Sandbox_mhrn_tabnet_updated.py​: Train TabNet-0 NonPC, Test PC or NonPC
- MHRN_TabNet_updated_Nick.py: Train TabNet-0 PC, Test PC or NonPC
- TabNet_pretraining_pc2pc.py: Train TabNet-1 PC, Test PC (pre-train ratio=0.5)
- TabNet_pretraining_pc2nonpc.py: Train TabNet-1 PC, Test NonPC (pre-train ratio=0.5)
- TabNet_pretraining_nonpc2pc.py: Train TabNet-1 NonPC, Test PC (pre-train ratio=0.5)
- TabNet_pretraining_nonpc2nonpc.py: Train TabNet-1 NonPC, Test NonPC (pre-train ratio=0.5)
- TabNet_pretraining_entire_data.py: Train TabNet-2, pre-train pc+nonpc, train classifier pc+nonpc, test pc OR nonpc (pre-train ratio=0.5)
- TabNet_pretraining_all_pc2pc.py: Train TabNet-2, pre-train pc+nonpc, train classifier pc, test pc (pre-train ratio=0.5)
- TabNet_pretraining_all_pc2nonpc.py: Train TabNet-2, pre-train pc+nonpc, train classifier pc, test nonpc (pre-train ratio=0.5)
- TabNet_pretraining_all_nonpc2pc.py: Train TabNet-2, pre-train pc+nonpc, train classifier nonpc, test pc (pre-train ratio=0.5)
- TabNet_pretraining_all_nonpc2nonpc.py: Train TabNet-2, pre-train pc+nonpc, train classifier nonpc, test nonpc (pre-train ratio=0.5)

SHAP Feature Importance analysis:

- SHAP_figures.py: run and plot SHAP results, run on train data

Fairness analysis:

- fairness_gender.py: model analysis for subsets of male and female
- fairness_race_grouped.py: model analysis for race subsets of white, black other

Other scripts:

- bootstrap_nick.py: perform bootstrapping with 95% confidence interval results on test set for all experiments
- fairness_plotting.py: create the fairness plot

- geenerate_sh_jobs_bootstrap_exps.py: helper function to generate bash scripts to run
- pre_process_320_to_102.py: matching the overlapping and non-overlapping feature names of the 102 coefficients released by MHRN and the overall 320. Also correcting mismatched names between MHRN and our data columns. 


## Saved Models

TBA. Pending release approval. 

## Data

Link to MHRN PC and MH coefficients: https://github.com/MHResearchNetwork/srpm-model 
(Look in "Results documents/", use "Primary care Coefficients.xlsx", "Mental health specialty coefficients.xlsx")

Our data collected at UMass Chan Medical is not publicly released at this time. 

List of data feature columns and descriptions can be found in: "Analytic data dictionary original.xlsx"


## Citation

If you use this repository, please cite our paper using: TBA

Please also be sure to cite the original TabNet paper and paper associated with MHRN by Simon et al.:

```latex
@inproceedings{arik2021tabnet,
  title={Tabnet: Attentive interpretable tabular learning},
  author={Arik, Sercan {\"O} and Pfister, Tomas},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={35},
  number={8},
  pages={6679--6687},
  year={2021}
}
  ```

```latex
@article{simon2018predicting,
  title={Predicting suicide attempts and suicide deaths following outpatient visits using electronic health records},
  author={Simon, Gregory E and Johnson, Eric and Lawrence, Jean M and Rossom, Rebecca C and Ahmedani, Brian and Lynch, Frances L and Beck, Arne and Waitzfelder, Beth and Ziebell, Rebecca and Penfold, Robert B and others},
  journal={American Journal of Psychiatry},
  volume={175},
  number={10},
  pages={951--960},
  year={2018},
  publisher={Am Psychiatric Assoc}
}
  ```
