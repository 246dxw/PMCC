# Prompt-affinity-Multi-modal-Class-Centroids-for-Unsupervised-Domain-Adaption



------

## Highlights

![Architecture](https://github.com/246dxw/PMCC/blob/main/Architecture.jpg)

> **Abstract:** In recent years, the advancements in large vision-language models (VLMs) like CLIP have sparked a renewed interest in leveraging the prompt learning mechanism to preserve semantic consistency between source and target domains in unsupervised domain adaptation (UDA). While these approaches show promising results, they encounter fundamental limitations when quantifying the similarity between source and target domain data, primarily stemming from the redundant and modality-missing class centroids. To address these limitations, we propose Prompt-affinity Multi-modal Class Centroids for UDA (termed as PMCC). Firstly, we fuse the text class centroids (directly generated from the text encoder of CLIP with manual prompts for each class) and image class centroids (generated from the image encoder of CLIP for each class based on source domain images) to yield the multi-modal class centroids. Secondly, we conduct the cross-attention operation between each source or target domain image and these multi-modal class centroids. In this way, these class centroids that contain rich semantic information of each class will serve as a bridge to effectively measure the semantic similarity between different domains. Finally, we employ a multi-modal prompt learning mechanism to accurately predict the true class of each image for both source and target domains. Extensive experiments on 3 popular UDA datasets (i.e., Office-31, Office-Home and VisDA-2017) validate the superiority of our PMCC compared with the state-of-the-art (SOTA) UDA methods. 

## Main Contributions



- **New perspective：** To the best of our knowledge, this is the first attempt to leverage both the visual and textual semantic information of each class from VLMs to preserve the semantic consistency between source and target domains in UDA.

- **Novel method：** We introduce a novel UDA method
  PMCC by constructing multi-modal class centroids that
  serve as a semantic bridge to effectively measure the sim-
  ilarity between data across domains with the multi-modal
  prompt learning mechanism.

- **High Performance：** We conduct extensive experiments
  on 3 popular UDA datasets including Office-31, Office-
  Home and VisDA-2017. The experimental results val-
  idate the superiority of our PMCC compared with the
  state-of-the-art (SOTA) UDA methods.

------

## Results



### PMCC in comparison with existing prompt tuning methods



Results reported below show accuracy across 3 UDA datasets with ViT-B/16 backbone. Our PMCC method adopts the paradigm of multi-modal prompt tuning.

| Name                                         | Office-Home Acc. | Office-31 Acc. | VisDA-2017 Acc. |
| -------------------------------------------- | ---------------- | -------------- | --------------- |
| [CLIP](https://arxiv.org/abs/2103.00020)     | 82.1             | 77.5           | 88.9            |
| [CoOp](https://arxiv.org/abs/2109.01134)     | 83.9             | 89.4           | 82.7            |
| [CoCoOp](https://arxiv.org/abs/2203.05557)   | 84.1             | 88.9           | 84.2            |
| [VPT-deep](https://arxiv.org/abs/2203.17274) | 83.9             | 89.4           | 86.2            |
| [MaPLe](https://arxiv.org/abs/2210.03117)    | 84.2             | 89.6           | 83.5            |
| [DAPL](https://arxiv.org/abs/2202.06687)     | 84.4             | 81.2           | 89.5            |
| [PDA](https://arxiv.org/abs/2312.09553)      | 85.7             | 91.2           | 89.6            |
| **PMCC(Ours)**                               | **86.0**         | **92.3**       | **89.8**        |

## Installation



For installation and other package requirements, please follow the instructions as follows. This codebase is tested on Ubuntu 18.04 LTS with python 3.7. Follow the below steps to create environment and install dependencies.

- Setup conda environment.

```
# Create a conda environment
conda create -y -n pmcc python=3.7

# Activate the environment
conda activate pmcc

# Install torch (requires version >= 1.8.1) and torchvision
# Please refer to https://pytorch.org/get-started/previous-versions/ if your cuda version is different
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
```



- Install dassl library.

```
# Instructions borrowed from https://github.com/KaiyangZhou/Dassl.pytorch#installation

# Clone this repo
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch

# Install dependencies
pip install -r requirements.txt

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
cd ..
```



- Clone PMCC code repository and install requirements.

```
# Clone PMCC code base
git clone https://github.com/246dxw/PMCC.git
cd PMCC

# Install requirements
pip install -r requirements.txt
```



## Data preparation



Please follow the instructions as follows to prepare all datasets. Datasets list:

- [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view?pli=1&resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw)
- [Office-31](https://faculty.cc.gatech.edu/~judy/domainadapt/#datasets_code)
- [VisDA-2017](http://ai.bu.edu/visda-2017/#download)

------

## Training and Evaluation

Please follow the instructions for training, evaluating and reproducing the results. Firstly, you need to **modify the directory of data by yourself**.

### Training



```
# Example: trains on Office-Home dataset, and the source domian is art and the target domain is clipart (a-c)
bash scripts/pmcc/main_pmcc.sh officehome b32_ep10_officehome PMCC ViT-B/16 2 a-c 0
```



### Evaluation



```
# evaluates on Office-Home dataset, and the source domian is art and the target domain is clipart (a-c)
bash scripts/pmcc/eval_pmcc.sh officehome b32_ep10_officehome PMCC ViT-B/16 2 a-c 0
```



The details are at each method folder in [scripts folder]([PMCC/scripts at main · 246dxw/PMCC (github.com)](https://github.com/246dxw/PMCC/tree/main/scripts)).



## Acknowledgements



Our style of reademe refers to [PDA](https://github.com/BaiShuanghao/Prompt-based-Distribution-Alignment). And our code is based on [CoOp and CoCoOp](https://github.com/KaiyangZhou/CoOp), [DAPL](https://github.com/LeapLabTHU/DAPrompt/tree/main) ，[MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning)  and [PDA](https://github.com/BaiShuanghao/Prompt-based-Distribution-Alignment) etc. repository. We thank the authors for releasing their code. If you use their model and code, please consider citing these works as well. Supported methods are as follows:

| Method       | Paper                                          | Code                                                         |
| ------------ | ---------------------------------------------- | ------------------------------------------------------------ |
| CoOp         | [IJCV 2022](https://arxiv.org/abs/2109.01134)  | [link](https://github.com/KaiyangZhou/CoOp)                  |
| CoCoOp       | [CVPR 2022](https://arxiv.org/abs/2203.05557)  | [link](https://github.com/KaiyangZhou/CoOp)                  |
| VPT          | [ECCV 2022](https://arxiv.org/abs/2203.17274)  | [link](https://github.com/KMnP/vpt)                          |
| IVLP & MaPLe | [CVPR 2023](https://arxiv.org/abs/2210.03117)  | [link](https://github.com/muzairkhattak/multimodal-prompt-learning) |
| DAPL         | [TNNLS 2023](https://arxiv.org/abs/2202.06687) | [link](https://github.com/LeapLabTHU/DAPrompt)               |
| PDA          | [AAAI 2024](https://arxiv.org/abs/2312.09553)  | [link](https://github.com/BaiShuanghao/Prompt-based-Distribution-Alignment) |

