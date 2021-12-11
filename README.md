# 2021-2 AI Project_SWE3032_41
## Semi-Supervised Learning for  Detecting Korean Malicious Comments 

### Abstract


본 연구에서는 준지도 학습을 자연어 처리에 적용하여 악성 댓글을 분류하는 학습 모델을 개발하고, labeled 데이터만을 이용하여 지도 학습한 모델에 비해 유사하거나 높은 성능을 얻는 것을 목표로 한다. 악의적인 내용을 포함하는 댓글을 수기로 레이블링하는 작업의 어려움을 고려하여, 앞으로의 악성 댓글 탐지에 실질적으로 적용할 수 있으면서 유의미한 성능을 내는 준지도 학습 모델을 구현하고자 한다. 추가적으로, labeled, unlabeled 데이터의 비율을 비교 및 평가하여 한국어 악성 댓글 분류를 위한 준지도 학습에서 가장 좋은 성능을 보이는 데이터의 최적 비율을 도출하고자 한다.


## Installation

Install dependencies:

```
pip install -r requirements.txt
```

Note that using different version of required packages can effects the results, especially PyTorch. The implementations are tested on Python 3.7+

## Dataset preparation

### Data sources

Korean malicious comments dataset is from [Korean HateSpeech Dataset](https://github.com/kocohub/korean-hate-speech.git)

데이터 셋의 크기 때문에 데이터 전처리에 많은 시간이 소요된다. 따라서 띄어쓰기와 맞춤법에 대한 사항이 전처리된 파일은 .csv 파일로 각각 저장했으며, 이 파일의 데이터를 모델의 입력값으로 사용한다.


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
