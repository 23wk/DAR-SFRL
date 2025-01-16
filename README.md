# DAR-SFRL: Degradation-adaptive Attack-Robust Self-Supervised Facial Representation Learning
![b](https://github.com/user-attachments/assets/f843c1d2-efc7-48a3-85e6-bb865abd0146)

We propose a novel framework, Degradation-based Attack-Robust Self-supervised Face Representation Learning (DAR-SFRL), which is designed to address the challenges posed by DIAs on clean facial data. DAR-SFRL first summarizes various DIAs into a unified degradation formula ($\mathbf{F'} = \mathbf{F} \cdot  \mathbf{M} + \boldsymbol{\epsilon}$) and then learns complex and diverse degradation patterns through k stages of iterative refinement, gradually reducing the impact of degradation attacks. Each stage consists of Degradation-Adaptive Face Restoration (DAFR) and Noise-Orthogonal Contrastive Learning (NOCL}), where DAFR addresses the degradation matrix $M$ from structured distortions and NOCL further alleviate the influence of the additive noise $\epsilon$ from unstructured  artifacts.

## Results
Evaluation for Facial Expression Recognition (FER) on RAF-DB Dataset Using Accuracy (Acc) and Drop Accuracy (Drop Acc). The $\uparrow$ represents the larger is better, while the $\downarrow$ represents the smaller is better. The best results are in bold.
![image](https://github.com/user-attachments/assets/70abe2ae-16f1-41fa-8577-b912289bd401)

Evaluation for Facial Recognition (FR) on CPLFW Dataset Using Accuracy (Acc) and Drop Accuracy (Drop Acc). The $\uparrow$ represents the larger is better, while the $\downarrow$ represents the smaller is better. The best results are in bold
![image](https://github.com/user-attachments/assets/05a94f00-1907-46a7-8d98-ccd38f2af689)

Evaluation for Facial AU detection on BP4D Dataset Using F1 score (F1) and Drop F1 score (Drop F1). The $\uparrow$ represents the larger is better, while the $\downarrow$ represents the smaller is better.The best results are in bold
![image](https://github.com/user-attachments/assets/4fac4f01-246d-4fe6-a481-d9d0c88327ac)

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

You can download our trained models from [Baidu Drive].

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
### 5.2 FR

To evaluate on RAF-DB, run:

```
python main.py --config_file configs/remote_DAR_linear_eval.yaml

```
### 5.3 AU

To evaluate on RAF-DB, run:

```
python main.py --config_file configs/remote_DAR_linear_eval.yaml

```

## TODO 

- [ ] Refactor the codes of AU detection and face recognition.

**IF YOU HAVE ANY PROBLEM, PLEASE CONTACT WK2023@cug.edu.cn OR COMMIT ISSUES**
