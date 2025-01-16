# DAR-SFRL: Degradation-adaptive Attack-Robust Self-Supervised Facial Representation Learning
![b](https://github.com/user-attachments/assets/f843c1d2-efc7-48a3-85e6-bb865abd0146)

We propose a novel framework, Degradation-based Attack-Robust Self-supervised Face Representation Learning (DAR-SFRL), which is designed to address the challenges posed by DIAs on clean facial data. Although DIAs take diverse forms, we summarize their perturbations into a unified degradation-based formula that formalizes them as a degradation function and additive noise. This formulation allows for comprehensive and targeted handling of DIAs. To systematically address these two components, we introduce two key modules in DAR-SFRL: Degradation-Adaptive Face Recovery (DAFR) and Noise-Orthogonal Contrastive Learning (NOCL). DAFR utilizes maximum a posteriori (MAP) estimation to progressively reverse the degradation function and recover fine-grained image details. It accurately models the relationship between degradation patterns and clean data and learns different degradation patterns during the gradual disentangling process.To further enhance robustness, NOCL incorporates a noise-orthogonal disentangling loss, a facial-robust contrastive loss, and a noise-sensitive contrastive loss. This ensures that the model not only discriminates between the additive noise of DIAs and clean images but also generalizes well to various DIAs.Through the synergistic training of DAFR and NOCL, DAR-SFRL effectively captures the perturbations caused by DIAs, enabling a more precise understanding of DIA patterns. This enhances the robustness of DAR-SFRL in face-related tasks, providing greater resilience against various out-of-distribution DIAs during inference. Despite the effectiveness of our approach, we find that DAR-SFRL still has some room for improvement. At present, the model lacks learning of the degradation process, which introduces uncertainty in the progressive recovery process.

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
