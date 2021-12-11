import string
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import tweepy
import konlpy
from konlpy.tag import Mecab

import src.options as options

op = options.Options()

unlabeled_ratio = op.unlabeled_ratio
unlabeled_num = 7000 * unlabeled_ratio - 1

# 데이터 경로 설정
train_csv_path = '../../../data/preprocessed/train.csv'
valid_csv_path = '../../../data/preprocessed/valid.csv'
unlabeled_train_csv_path = '../../../data/preprocessed/unlabeled_train.csv'
embedding_path = '../../../model/wordemb-pretrain124all.pt'

unlabeled_ratio = op.unlabeled_ratio
unlabeled_num = 7000 * unlabeled_ratio - 1

# 데이터를 dataframe 으로 load
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

#print(train_df.head())

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

# print(len(train_x))
# print(len(valid_x))
# print(len(test_x))
# print(len(unlabeled_x))


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
            if (self.max_len > len(comment)):
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

# print(len(vocab))
# print(next(iter(train_loader)))

train_total = len(train_df)
valid_total = len(valid_df)
unlabeled_total = len(unlabeled_df)

train_batches = len(train_loader)
valid_batches = len(valid_loader)
unlabeled_batches = len(unlabeled_loader)
total_batches = train_batches + unlabeled_batches


