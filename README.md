# MRE text-only experiments (random 40 base / 40 novel)

本分支直接基于 `fc25742`，不包含审计提交 `2ccf1b1`。类别划分改为 MRE 默认：合并 train.txt/test.txt 的 80 类，按 seed 随机分为 40 Base / 40 Novel，再按原 MRE 随机数顺序划分样本。其余保留该 baseline 的普通随机采样、k=50、batch=64/128/64、RTR、末轮模型、测试集 KMeans 和 MRE 计分。GPT 仍为 `gpt-3.5-turbo`，FewRel Head→Tail 提示词保持不变，中转站后端快照未核实。入口为 `run_mre.py`，配置为 `mre_config.py`，新实验标识 `loop_mre_random40_gpt35_v3`。旧固定 64/16 的六组结果单独存档。

**[查看中文实验协议、改动说明与服务器操作步骤 → README_MRE.md](README_MRE.md)**

以下保留原始 LOOP 项目说明。

# GCD with LLMs in the Loop (LOOP)
Data and code for paper titled [Generalized Category Discovery with Large Language Models in the Loop](https://arxiv.org/abs/2312.10897) (ACL 2024 Findings paper)

*Generalized Category Discovery (GCD)* is a crucial task that aims to recognize both known and novel categories from a set of unlabeled data by utilizing a few labeled data with only known categories. Due to the lack of supervision and category information, current methods usually perform poorly on novel categories and struggle to reveal semantic meanings of the discovered clusters, which limits their applications in the real world. In this paper, we propose Loop, an end-to-end active-learning framework that introduces LLMs into the training loop, which can boost model performance and generate category names without relying on any human efforts.


## Contents
[1. Data](#data)

[2. Model](#model)

[3. Requirements](#requirements)

[4. Running](#running)

[5. Results](#results)

[6. Thanks](#thanks)

[7. Citation](#citation)

## Data
We performed experiments on three public datasets: [clinc](https://aclanthology.org/D19-1131/), [banking](https://aclanthology.org/2020.nlp4convai-1.5/) and [stackoverflow](https://aclanthology.org/W15-1509/), which have been included in our repository in the data folder ' ./data '.

## Model
An overview of our model is shown in the figure.
<div align=center>
<img src="./figures/model.png"/>
</div>

## Requirements
* python==3.8
* pytorch==1.11.0
* transformers==4.15.0
* openai==0.28.0
* scipy==1.9.3
* numpy==1.23.5
* scikit-learn==1.2.0
* faiss-gpu==1.7.2

## Running
Pre-training, training and testing our model through the bash scripts:
```
sh run.sh
```
You can also add or change parameters in run.sh (More parameters are listed in init_parameter.py)

## Results
<div align=center>
<img src="./figures/results.jpg"/>
</div>
It should be noted that the experimental results may be different because of the randomness of clustering when testing even though we fixed the random seeds.

## Thanks
Some code references the following repositories:
* [DPN](https://github.com/Lackel/DPN)
* [NID](https://github.com/fanolabs/NID_ACLARR2022)

## Citation
If our paper or code is helpful to you, please consider citing our paper:
```
@article{an2023generalized,
  title={Generalized Category Discovery with Large Language Models in the Loop},
  author={An, Wenbin and Shi, Wenkai and Tian, Feng and Lin, Haonan and Wang, QianYing and Wu, Yaqiang and Cai, Mingxiang and Wang, Luyan and Chen, Yan and Zhu, Haiping and others},
  journal={arXiv preprint arXiv:2312.10897},
  year={2023}
}
```
