# 2021-2 AI Project_SWE3032_41
## Semi-Supervised Learning for  Detecting Korean Malicious Comments 

### Brief Explanation


This project aims to develop a model that classifies malicious comments by applying semi-supervised learning to natural language processing, and to obtain similar or higher performance than a supervised model using only labeled data. Considering the difficulty of manually labeling malicious comments, we intend to implement a semi-supervised learning model that can be practically applied to malicious comment detection. Additionally, the project compares and evaluates the ratio of labeled and unlabeled data to derive the optimal ratio of data that shows the best performance in semi-supervised learning for classifying malicious Korean comments.


## Installation

Install dependencies:

```
pip3 install konlpy
pip3 install tweepy==3.10.0
bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)
```

Note that using different version of required packages can effects the results, especially PyTorch. The implementations are tested on Python 3.7+


## Dataset

### Data sources

Korean malicious comments dataset is from [Korean HateSpeech Dataset](https://github.com/kocohub/korean-hate-speech.git)

### Input of the Model

Due to the large dataset, preprocessing of the data takes a lot of time. Therefore, preprocessed files on spacing and basic spelling were stored in `data/processed` as .csv files separately, and the model uses the data in this file as input values.
You can use `preprocess.py` if you want to preprocess data yourself.


## Hyperparameters


```python
class Options(object):
    unlabeled_ratio = 1 # labeled : unlabeled ratio
    max_len = 50 # maximum comment length
    min_count = 0 # minimum number to include vocab list
    batch_size = 64
    
    pretrained = False # choose pretrained model or not
    num_layers = 1 
    hidden_dim = 1024 # dimension of hidden layer
    embed_dim = 256 # dimension of embedding layer
    dropout = 0.1 # the rate of dropout
    vat = True
    epsilon = 1e+6 # perturbation size
    alpha = 1.0 # regularization coefficient
    
    lr = 0.001 # learning rate
    factor = 0.5 # the decrease ratio of learning rate
    patience = 2 # Count the number of times the learning rate decreases
    verbose = True
    epoch = 100
```


## How to use

### Input Command
When executing the BiLSTM4VAT model, execute the following code inside the src/ directory. The model will consist of pretrained word embeddings by default.

When executing the SemiPretSeq2Seq model, execute the following code inside the src/utils/pretrain/ directory. This model is only for pretraining the word embeddings for BiLSTM4VAT.
```
python main.py
```

### Output Example
```
[ 1] TRAIN loss: 0.546, acc: 41.414, lr: 0.001000 .... VALID loss: 1.099, acc: 32.484, best_loss: 1.099 .... patience: 0
...  
TEST loss: 1.034, acc: 47.826
```

The `main.py` will be executed with hyperparameters above. You can change the hyperparameter by changing `options.py` in `src`.  
The model created by `main.py` will be saved in model as `your_model_default(pretrained)_base(vat)_(unlabeld_ratio).py`.
