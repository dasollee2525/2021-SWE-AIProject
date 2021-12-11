# 2021-2 AI Project_SWE3032_41
## Semi-Supervised Learning for  Detecting Korean Malicious Comments 

### Brief Explanation


This project aims to develop a model that classifies malicious comments by applying semi-supervised learning to natural language processing, and to obtain similar or higher performance than a supervised model using only labeled data. Considering the difficulty of manually labeling malicious comments, we intend to implement a semi-supervised learning model that can be practically applied to malicious comment detection. Additionally, the project compares and evaluates the ratio of labeled and unlabeled data to derive the optimal ratio of data that shows the best performance in semi-supervised learning for classifying malicious Korean comments.


## Installation

Install dependencies:

```
git clone https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git
cd Mecab-ko-for-Google-Colab
bash install_mecab-ko_on_colab190912.sh
```

Note that using different version of required packages can effects the results, especially PyTorch. The implementations are tested on Python 3.7+

## Dataset preparation

### Data sources

Korean malicious comments dataset is from [Korean HateSpeech Dataset](https://github.com/kocohub/korean-hate-speech.git)

### Input of the Model

Due to the large dataset, preprocessing of the data takes a lot of time. Therefore, preprocessed files on spacing and basic spelling were stored as .csv files separately, and the model uses the data in this file as input values.


## Hyperparametrs


```
class Options(object):
    unlabeled_ratio = 4 # labeled : unlabeled ratio
    max_len = 50 # maximum comment length
    min_count = 0 # minimum number to include vocab list
    batch_size = 64
    
    pretrained = True
    num_layers = 1
    hidden_dim = 1024
    embed_dim = 256
    dropout = 0.1
    vat = True
    epsilon = 1e+6 # perturbation size
    alpha = 1.0 # regularization coefficient
    
    lr = 0.001
    factor = 0.5
    patience = 2
    verbose = True
    epoch = 100
```


## How to run

### Pretrained models


### Training

- Korean-handwriting

```
python train.py $NAME cfgs/kor.yaml
```

- Thai-printing

```
python train.py $NAME cfgs/kor.yaml cfgs/thai.yaml
```


### Evaluation

- Korean-handwriting

```
python evaluator.py $NAME $CHECKPOINT_PATH $OUT_DIR cfgs/kor.yaml --mode cv-save
```
