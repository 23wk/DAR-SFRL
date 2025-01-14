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

You can download our trained models from [Baidu Drive](https://pan.baidu.com/s/10j21PCyhi9cbJqRvH7KDHw) (2qia).

## 4. Training

To train the model in the paper, run this command:

```
python main.py --config_file configs/remote_DAR_vox.yaml
```

## 5. Evaluation

We used the linear evaluation protocol for evaluation.

### 5.1 FER

To evaluate on RAF-DB, run:

```
python main.py --config_file configs/remote_DAR_linear_eval.yaml

```


## TODO 

- [ ] Refactor the codes of AU detection and face recognition.

**IF YOU HAVE ANY PROBLEM, PLEASE CONTACT wangwenbin@cug.edu.cn OR COMMIT ISSUES**
