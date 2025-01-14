# DAR-SFRL: Degradation-adaptive Attack-Robust Self-Supervised Facial Representation Learning

## 0. Contents
1. Requirements
2. Data Preparation
3. Pre-trained Models
4. Training
5. Evaluation

## 1. Requirements

To install requirements:
Python Version: 3.7.9

```
pip install -r Requirements.txt
```

## 2. Data Preparation

You need to download the related datasets  and put in the folder which namely dataset.

## 3. Pre-trained Models

You can download our trained models from [Baidu Drive](https://pan.baidu.com/s/10j21PCyhi9cbJqRvH7KDHw) (2qia) and [Google Drive](https://drive.google.com/drive/folders/1wx5PTGDCqDWsjhXimjHqz_7WUwxr54uh?usp=sharing) .

## 4. Training

To train the model in the paper, run this command:

```
nohup python main.py --config_file configs/remote_PCL_vox.yaml > ADM_SFRL_train_learnable.log
nohup python main_dcl.py --config_file configs/remote_PCL_vox.yaml > ADM_SFRL_train_2a.log
```

## 5. Evaluation

We used the linear evaluation protocol for evaluation.

### 5.1 FER

To evaluate on RAF-DB, run:

```
nohup python main.py --config_file configs/remote_PCL_linear_eval.yaml > ADM_SRFL_test_learn.log
nohup python main_dcl.py --config_file configs/remote_PCL_linear_eval.yaml > ADM_SFRL_test_adcl.log

nohup python robustness.py --config_file configs/remote_PCL_linear_robustness.yaml > ADM_SRFL_test_anet_robust_pgd.log

nohup python feature.py --config_file configs/remote_PCL_linear_eval_fea.yaml > TAR_anet_feature.log


nohup python main_id.py --config_file configs/remote_ExpPose_linear_eval_id_cplfw.yaml > test_normal_id_pgd

nohup python main_au.py --config_file configs/remote_ExpPose_linear_eval_au_disfa.yaml > test_normal_au_set2_sinifgsm

nohup python main_feature.py --config_file configs/remote_PCL_linear_eval_fea.yaml > test_normal_1001

nohup python main_feature_id.py --config_file configs/remote_PCL_linear_eval_fea_id.yaml > test_normal_1001
```

### 5.2 Pose regression

To trained on 300W-LP and evaluated on AFLW2000, run:

```
python main_pose.py --config_file configs/remote_PCL_linear_eval_pose.yaml
```

### 5.3 Visualization

To visualize on RAF-DB, run:

```
python visualize.py
```



## TODO 

- [ ] Refactor the codes of AU detection and face recognition.

**IF YOU HAVE ANY PROBLEM, PLEASE CONTACT wangwenbin@cug.edu.cn OR COMMIT ISSUES**
