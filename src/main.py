import os
import random
import string
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torch.optim as optim

from konlpy.tag import Mecab

import model as m
import options


# seed
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# paths
train_csv_path = '../data/preprocessed/train.csv'
valid_csv_path = '../data/preprocessed/valid.csv'
unlabeled_train_csv_path = '../data/preprocessed/unlabeled_train.csv'
embedding_path = '../model/wordemb-pretrain124all.pt'


# options
op = options.Options()


# load data
unlabeled_ratio = op.unlabeled_ratio
unlabeled_num = 7000 * unlabeled_ratio - 1

train_df = pd.read_csv(train_csv_path, header=0)
train_df = train_df.loc[:, ['comments', 'hate']]
test_df = pd.DataFrame()
test_df = (train_df.loc[6999:]).copy()
train_df = train_df.loc[:6999]
test_df.reset_index(drop=True, inplace=True)

valid_df = pd.read_csv(valid_csv_path, header=0)
valid_df = valid_df.loc[:, ['comments', 'hate']]

unlabeled_df = pd.read_csv(unlabeled_train_csv_path, header=0)
unlabeled_df = unlabeled_df.loc[:unlabeled_num, ['comments']]
unlabeled_df = unlabeled_df.dropna()


# tokenize
tokenizer = Mecab()


def remove_values_from_list(the_list, val):
    return [value for value in the_list if value != val]


def remove_symbols_from_comments(comments):
    token = []
    for comment in comments:
        try:
            token.append(tokenizer.morphs(comment))
        except KeyError:
            pass
    for i in range(len(token)):
        token[i] = [word.translate(str.maketrans('', '', string.punctuation)) for word in token[i]]
        token[i] = remove_values_from_list(token[i], '')
    return token


train_df[['comments'][0]] = remove_symbols_from_comments(train_df[['comments'][0]])
valid_df[['comments'][0]] = remove_symbols_from_comments(valid_df[['comments'][0]])
test_df[['comments'][0]] = remove_symbols_from_comments(test_df[['comments'][0]])
unlabeled_df[['comments'][0]] = remove_symbols_from_comments(unlabeled_df[['comments'][0]])


# vocab
def get_array_from_index(df, index):
    return np.array([v for v in df[[index][0]]])


def convert_to_vocab_id(vocab, dataset, labels, convert_vocab=True, ignore_unk=False, ign_eos=False):
    dataset_x = []
    dataset_x_len = []
    dataset_y = []

    def conv(words):
        if ignore_unk:
            return [vocab.get(w, 1) for w in words if w in vocab]
        else:
            return [vocab.get(w, 1) for w in words]

    for i, words in enumerate(dataset):
        if convert_vocab:
            if ign_eos:
                conv_words = conv(words)
            else:
                conv_words = conv(words) + [0]
            word_ids = np.array(conv_words, dtype=np.int32)  # EOS
        else:
            word_ids = ' '.join(words)
        dataset_x_len.append(len(word_ids))
        dataset_x.append(word_ids)
        if len(labels) != 0:
            if labels[i] == 'none':
                dataset_y.append(0)
            if labels[i] == 'offensive':
                dataset_y.append(1)
            if labels[i] == 'hate':
                dataset_y.append(2)

    dataset_y = np.array(dataset_y, dtype=np.int32)
    return dataset_x, dataset_x_len, dataset_y


min_count = op.min_count


vocab = {}
vocab['<eos>'] = 0
vocab['<unk>'] = 1

word_cnt = {}
doc_counts = {}

train_labels = get_array_from_index(train_df, 'hate')
train_set = get_array_from_index(train_df, 'comments')
valid_labels = get_array_from_index(valid_df, 'hate')
valid_set = get_array_from_index(valid_df, 'comments')
test_labels = get_array_from_index(test_df, 'hate')
test_set = get_array_from_index(test_df, 'comments')
unlabeled_set = get_array_from_index(unlabeled_df, 'comments')

total_set = np.append(train_set, unlabeled_set)

for words in total_set:
    doc_seen = set()
    for w in words:
        word_cnt[w] = word_cnt.get(w, 0) + 1
        if w not in doc_seen:
            doc_counts[w] = doc_counts.get(w, 0) + 1
            doc_seen.add(w)
        if w not in vocab and doc_counts[w] > min_count:
            vocab[w] = len(vocab)

train_x, train_x_len, train_y = convert_to_vocab_id(vocab, train_set, train_labels)
valid_x, valid_x_len, valid_y = convert_to_vocab_id(vocab, valid_set, valid_labels)
test_x, test_x_len, test_y = convert_to_vocab_id(vocab, test_set, test_labels)
unlabeled_x, _, _ = convert_to_vocab_id(vocab, unlabeled_set, [])


# dataloader
class HateCommentDataset(Dataset): 
    def __init__(self, x_data, y_data, max_len):
        self.x_data = x_data
        self.y_data = y_data
        self.max_len = max_len
    
    def __len__(self): 
        return len(self.x_data)

    def __getitem__(self, idx): 
        x = self.x_data
        y = self.y_data

        for index, comment in enumerate(x):
            if(self.max_len > len(comment)):
                padding = [0] * (self.max_len - len(comment))
                padding = np.array(padding)
                x[index] = np.concatenate((comment, padding), axis=None)
            else:
                x[index] = comment[:self.max_len]

        text = x[idx]
        label = y[idx]
        return text, label


max_len = op.max_len
train_dataset = HateCommentDataset(train_x, train_y, max_len)
valid_dataset = HateCommentDataset(valid_x, valid_y, max_len)
test_dataset = HateCommentDataset(test_x, test_y, max_len)
unlabeled_dataset = HateCommentDataset(unlabeled_x, unlabeled_x, max_len)

batch_size = op.batch_size

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

train_total = len(train_df)
valid_total = len(valid_df)
test_total = len(test_df)
unlabeled_total = len(unlabeled_df)

train_batches = len(train_loader)
valid_batches = len(valid_loader)
test_batches = len(test_df)
unlabeled_batches = len(unlabeled_loader)
total_batches = train_batches + unlabeled_batches


# embedding
if op.pretrained:
    n_emb = []
    with open(embedding_path) as fp:
        for line in fp:
            n_emb.extend([line.split()])
    for i in n_emb:
        i.pop(0)

    n_emb = np.array(n_emb).astype(float)
    n_emb = torch.FloatTensor(n_emb)
    word_embedding = nn.Embedding.from_pretrained(n_emb)

    if op.vat:
        model_path = '../model/your_model_pretrained_vat_%d.pt' % op.unlabeled_ratio
    else:
        model_path = '../model/your_model_pretrained_base.pt'
else:
    word_embedding = None
    if op.vat:
        model_path = '../model/your_model_default_vat_%d.pt' % op.unlabeled_ratio
    else:
        model_path = '../model/your_model_default_base.pt'

# load model
num_vocab = len(vocab)
num_classes = 3

model = m.BiLSTM4VAT(op.hidden_dim,
                   op.num_layers, 
                   num_vocab, 
                   num_classes, 
                   op.embed_dim, 
                   word_embedding,
                   op.dropout, 
                   op.vat, 
                   op.epsilon, 
                   device=device).to(device)
cross_entropy = nn.CrossEntropyLoss()
kl_div_loss = nn.KLDivLoss(reduction='batchmean')
optimizer = optim.Adam(model.parameters(), lr = op.lr, weight_decay=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=op.factor, patience=op.patience, verbose=True)

                    
# load unlabeled data
def get_unlabeled(idx):
    x, _ = next(iter(unlabeled_loader))
    return x

                    
# learning
epochs = op.epochs
num_layers = op.num_layers
hidden_dim = op.hidden_dim
best_valid_loss = 1024
patience = 0

print('START LEARNING')

for epoch in range(epochs):

    # train
    model.train()
    
    train_loss = 0
    train_correct = 0
    
    for i, batch in enumerate(train_loader):
        # supervised learning
        x, y = batch
        x = x.to(device).long()
        y = y.to(device).long()
        h0 = torch.zeros(num_layers * 2, batch_size, hidden_dim).to(device) # Bidirectional=True
        c0 = torch.zeros(num_layers * 2, batch_size, hidden_dim).to(device) # Bidirectional=True

        outputs = model(x, h0, c0)
        loss = cross_entropy(outputs, y)

        # virtaul adversarial training
        if op.vat:
            for j in range(unlabeled_ratio):
                x_unlabeled = get_unlabeled(i).to(device).long()
                h0 = torch.zeros(num_layers * 2, batch_size, hidden_dim).to(device) # Bidirectional=True
                c0 = torch.zeros(num_layers * 2, batch_size, hidden_dim).to(device) # Bidirectional=True

                outputs_orig = model(x_unlabeled, h0, c0)
                outputs_orig = F.softmax(outputs_orig, dim=1)
                outputs_vat_first = model(x_unlabeled, h0, c0, first_step=True)
                outputs_vat_first = F.softmax(outputs_vat_first, dim=1)

                loss_vat_first = kl_div_loss(outputs_orig.log(), outputs_vat_first)
                loss_vat_first.backward(retain_graph=True)
                g = model.r.grad.data

                outputs_vat = model(x_unlabeled, h0, c0, g=g)
                outputs_vat = F.softmax(outputs_vat_first, dim=1)
                loss_vat = kl_div_loss(outputs_orig.log(), outputs_vat)

                loss += op.alpha * loss_vat

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_correct += predicted.eq(y).sum().item()

    train_loss = train_loss / total_batches
    train_acc = train_correct / train_total
    
    # Validate
    model.eval()
    
    valid_loss = 0
    valid_correct = 0
    
    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            x, y = batch
            x = x.to(device).long()
            y = y.to(device).long()
            h0 = torch.zeros(num_layers * 2, batch_size, hidden_dim).to(device) # Bidirectional=True
            c0 = torch.zeros(num_layers * 2, batch_size, hidden_dim).to(device) # Bidirectional=True
            outputs = model(x, h0, c0, train=False)
            loss = cross_entropy(outputs, y)
            
            valid_loss += loss.item()
            _, predicted = outputs.max(1)
            valid_correct += predicted.eq(y).sum().item()
            
    valid_loss = valid_loss / valid_batches
    valid_acc = valid_correct / valid_total
    
    # save best model
    if best_valid_loss > valid_loss:
        torch.save(model.state_dict(), model_path)
        best_valid_loss = valid_loss
        patience = 0
    else:
        patience += 1

    print('[%2d] TRAIN loss: %.3f, acc: %.3f, lr: %f .... VALID loss: %.3F, acc: %.3f, best_loss: %.3f .... patience: %d'
          % (epoch+1, train_loss, train_acc * 100, optimizer.param_groups[0]['lr'], valid_loss, valid_acc * 100, best_valid_loss, patience))

    if patience // (op.patience + 1) == 2:
        break

    scheduler.step(metrics=valid_loss)

              
# test
model = m.BiLSTM4VAT(op.hidden_dim, op.num_layers, num_vocab, num_classes, op.embed_dim, embedding=word_embedding, dropout=op.dropout, vat=op.vat, epsilon=op.epsilon, device=device).to(device)
model.load_state_dict(torch.load(model_path))

test_loss = 0
test_correct = 0

with torch.no_grad():
    for i, batch in enumerate(test_loader):
        x, y = batch
        x = x.to(device).long()
        y = y.to(device).long()
        h0 = torch.zeros(num_layers * 2, batch_size, hidden_dim).to(device) # Bidirectional=True
        c0 = torch.zeros(num_layers * 2, batch_size, hidden_dim).to(device) # Bidirectional=True
        outputs = model(x, h0, c0, train=False)
        loss = cross_entropy(outputs, y)
        
        test_loss += loss.item()
        _, predicted = outputs.max(1)
        test_correct += predicted.eq(y).sum().item()
        
test_loss = test_loss / test_batches
test_acc = test_correct / test_total

print('TEST loss: %.3f, acc: %.3f' % (test_loss, test_acc * 100))
