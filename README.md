# Model_Generalizability_Across_Clinical_Settings_SARP

# Evaluating Model Generalizability for Suicide Attempt Risk Prediction: Traditional Machine vs Deep Learning

Corresponding Authors: Nicholas Josselyn, Ed Boudreaux, Feifan Liu

Contact: njjosselyn [at] wpi.edu, Edwin.Boudreaux [at] umassmed.edu, feifan.liu [at] umassmed.edu

Link to paper: TBA

Nicholas Josselyn, Sahil Sawant, Rachel E. Davis-Martin, Elke Rundensteiner, Ben S. Gerber, Bo Wang, Anthony J. Rothschild, Emmanuel Agu, Edwin D. Boudreaux, Feifan Liu. "Evaluating Model Generalizability for Suicide Attempt Risk Prediction: Traditional Machine vs Deep Learning". In submission.

In this work we investigate the generalizability of pre-trained logistic regression models by the NIMH-funded MHRN group on our UMass Memorial Health dataset of over 750,000 patient encounters for the task of suicide attempt risk prediction (SARP). Additionally, we train and evaluate machine (ML) and deep learning (DL) models on our UMass Memorial Health dataset and study the generalizability of these models across different patient healthcare visits, in particular primary care (PC) and mental health specialty (MH) healthcare visits. Additionally, we provide feature importance analysis using SHAP to identify top contributing features to making predictions and a fairness anlaysis study on demographic subgroups of race (white, black, other), sex (male, female), and ethnicity (Hispanic, not Hispanic). 

We provide this code repository for reproducibility of results. 

## Introduction
This code was used to produce the results in the above paper title. Included here will be:

- All machine and deep learning model codes (Logistic Regression, Random Forest, XGBoost, MLP, ResNet, FT-Transformer, TabNet, TabNet+pretraining)
- Training scripts
- Optuna tuning
- Bootstrap sampling and 95% confidence intervals code (test set)
- SHAP feature analysis
- Fairness analysis
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

In this work we use Logistic Regression (pre-trained and from scratch), Random Forest, XGBoost, TabNet, and TabNet+pretraining models. As explained in the paper, we run MHRN (logistic regression) model generalizability experiments using pre-trained MHRN group coefficients compared to training from scratch on our dataset. We then run a set of comparitive experiments of ML and DL models for all source-target transfer tasks of primary care (PC) and mental health specialty (MH) healthcare settings. These are all in-domain (pc2pc, mh2mh) or out-of-domain (pc2mh, mhc2pc). We then do feature importance analysis (SHAP) and a fairness study. All codes and descriptions are listed below, Optuna hyperparameter tuning is done for all relevant experiments. NOTE: When referring to Non-PC, we mean MH. 

MHRN Generalizability experiments:

- ML_log_reg_pretrain.py: evaluate MHRN pretrained 102 PC coefficients directly on our PC data (no training)
- ML_log_pretrain_MH.py: evaluate MHRN pretrained 94 MH coefficients directly on our MH data (no training)
- ML_tune_log_reg_102feats.py: train logistic regression from scratch on 102 PC features for pc2pc (tuning with and without balancing separately)
- ML_tune_log_reg_94feats_MH.py: train logistic regression from scratch on 94 MH features for mh2mh (tuning with and without balancing separately)
- ML_tune_log_reg.py: train logistic regression from scratch on 320 features for pc2pc (tuning with and without balancing separately)
- ML_tune_log_reg_MH.py: train logistic regression from scratch on 320 features for mh2mh (tuning with and without balancing separately)

Machine vs Deep Learning experiments:

- ML_XGboost_entire_data.py: Train XGBoost PC+MH, Test PC or MH
- ML_XGboost_train1_test1.py: Train XGBoost PC or MH, Test PC or MH
- ML_RandomForest_1to1.py​: Train Random Forest PC or MH, Test PC or MH​
- ML_DL_T0_RF_train_entire_data.py​:
  - Random Forest training: Train Random Forest PC+MH, Test PC or MH
  - TabNet training: Train TabNet-0 PC+MH, Test PC or MH
- DL_TabNet0_train_pc.py​: Train TabNet-0 PC, Test PC or MH
- DL_TabNet0_train_nonpc.py: Train TabNet-0 MH, Test PC or MH
- DL_TabNet_pretraining_pc2pc.py: Train TabNet-1 PC, Test PC (pre-train ratio=0.5)
- DL_TabNet_pretraining_pc2nonpc.py: Train TabNet-1 PC, Test MH (pre-train ratio=0.5)
- DL_TabNet_pretraining_nonpc2pc.py: Train TabNet-1 MH, Test PC (pre-train ratio=0.5)
- DL_TabNet_pretraining_nonpc2nonpc.py: Train TabNet-1 MH, Test MH (pre-train ratio=0.5)
- DL_TabNet_pretraining_entire_data.py: Train TabNet-2, pre-train pc+mh, train classifier pc+mh, test pc OR mh (pre-train ratio=0.5)
- DL_TabNet_pretraining_all_pc2pc.py: Train TabNet-2, pre-train pc+mh, train classifier pc, test pc (pre-train ratio=0.5)
- DL_TabNet_pretraining_all_pc2nonpc.py: Train TabNet-2, pre-train pc+mh, train classifier pc, test mh (pre-train ratio=0.5)
- DL_TabNet_pretraining_all_nonpc2pc.py: Train TabNet-2, pre-train pc+mh, train classifier mh, test pc (pre-train ratio=0.5)
- DL_TabNet_pretraining_all_nonpc2nonpc.py: Train TabNet-2, pre-train pc+mh, train classifier mh, test mh (pre-train ratio=0.5)
- ft_transformer.py: Train FT-Transformer for any of the 4 tasks AND perform bootstrap metrics on test set
- ResNet_tabular.py: Train ResNet for any of the 4 tasks AND perform bootstrap metrics on test set
- mlp_tabular.py: Train MLP for any of the 4 tasks AND perform bootstrap metrics on test set

SHAP Feature Importance analysis:

- SHAP_figures.py: run and plot SHAP results, run on train data

Fairness analysis:

- fairness_gender.py: model analysis for subsets of male and female sexes
- fairness_race_grouped.py: model analysis for race subsets of white, black other
- fairness_ethnicity.py: model analysis for ethnicity subsets of Hispanic and not Hispanic
- fairness_quantify: fairness metrics for race
- fairness_quantify_ethnicity.py: fairness metrics for ethnicity
- fairness_quantify_gender.py: fairness metrics for sex
- fairness_plotting.py: create the fairness plot

Other scripts:

- bootstrap_over13.py: perform bootstrapping with 95% confidence interval results on test set for all experiments
- stat_sig_test_bootstrap.py: calculate all pairwise model statistical significance
- geenerate_sh_jobs_bootstrap_exps.py: helper function to generate bash scripts to run


## Saved Models

TBA. Pending release approval. 

## Data

Link to MHRN PC and MH coefficients: https://github.com/MHResearchNetwork/srpm-model 
(Look in "Results documents/", use "Primary care Coefficients.xlsx", "Mental health specialty coefficients.xlsx")

Our data collected at UMass Memorial Health is not publicly released at this time, pending approval.

List of data feature columns and descriptions can be found in: "Analytic data dictionary original.xlsx"


## Citation

If you use this repository, please cite our paper using: TBA

Please also be sure to cite the original TabNet and FT-Transformer papers and paper associated with MHRN by Simon et al.:

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

```latex
@article{gorishniy2021revisiting,
  title={Revisiting deep learning models for tabular data},
  author={Gorishniy, Yury and Rubachev, Ivan and Khrulkov, Valentin and Babenko, Artem},
  journal={Adv. in Neur. Inf. Proc. Sys.},
  volume={34},
  pages={18932--18943},
  year={2021}
}
  ```
