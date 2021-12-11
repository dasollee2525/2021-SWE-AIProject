import os
import random
import numpy as np
import torch
import torch.nn as nn
import time

import torch.optim as optim

from src import options
from src.utils.pretrain.encoder_decoder import EncoderRNN, DecoderRNN
from src.utils.pretrain.model import SemiPretSeq2Seq
from src.utils.pretrain.data_vocab import vocab, unlabeled_loader, valid_loader, train_loader, test_loader
from src.utils.pretrain.util_functions import init_weights, write_embeddings, epoch_time

# seed
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

op = options.Options()
batch_size = op.batch_size

# path
train_csv_path = '../../../data/preprocessed/train.csv'
valid_csv_path = '../../../data/preprocessed/valid.csv'
unlabeled_train_csv_path = '../../../data/preprocessed/unlabeled_train.csv'
model_path = '../../../model/my_pretrain_model.pt'
emb_path = '../../../model/my_new_word_emb.pt'

options = options.Options()
batch_size = op.batch_size

N_EPOCHS = options.pretrain_epoch
CLIP = options.pretrain_clip
# valid loss 초기값 무한
best_valid_loss = float('inf')

id_to_vocab = {y:x for x,y in vocab.items()}

num_vocab = len(vocab)

word_embedding = nn.Embedding(num_vocab, options.embed_dim)

encoder = EncoderRNN(options.hidden_dim, options.num_layers, options.embed_dim, word_embedding, 0.1)
decoder = DecoderRNN(num_vocab, options.hidden_dim, options.num_layers, options.embed_dim, word_embedding, 0.1)
model = SemiPretSeq2Seq(encoder, decoder, num_vocab, device=device).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=options.lr)

model.apply(init_weights)


def train(model, train_loader, optimizer, criterion, clip):

    # 모델 training mode 설정
    model.train()

    epoch_loss = 0

    # train_loader 을 batch 단위로
    for i, batch in enumerate(train_loader):
        # batch 가져오고 comment id를 long 으로 변환
        # label 필요 없기 때문에 제거
        x, _ = batch
        x = x.to(device).long()

        # SemiPretSeq2Seq 모델에 batch input 넣어주고 output 받음
        # output = [batch size, trg len, vocab len]
        output = model(x)

        # output_dim: vocab len
        output_dim = output.shape[-1]

        # [:, 1:, :] 역할: 각 문장의 첫번째 글자 제거
        # view(-1): loss 계산은 2d input 형태여야 함
        output = output[:, 1:, :]
        x = x[:, 1:]

        output = output.reshape([-1, output_dim])
        x = x.reshape([-1])

        # print("after output: ", output.shape, "x: ", x.shape)

        loss = criterion(output, x)

        loss.backward()

        # gradient clipping 실행
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(train_loader)


# gradient update 없이 진행하는 것 빼고는 train 코드와 거의 유사함
def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            x, _ = batch
            x = x.to(device).long()

            # SemiPretSeq2Seq 모델에 batch input 넣어주고 output 받음
            # output = [batch size, trg len, vocab len]
            output = model(x)

            output_dim = output.shape[-1]

            # [:, 1:, :] 역할: 각 문장의 첫번째 글자 제거
            # view(-1): loss 계산은 2d input 형태여야 함
            output = output[:, 1:, :]
            x = x[:, 1:]

            output = output.reshape([-1, output_dim])
            x = x.reshape([-1])

            loss = criterion(output, x)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


# valid loss 초기값 무한
best_valid_loss = float('inf')

# train 시작
for epoch in range(options.pretrain_epoch):

    print("pretrain train start")

    start_time = time.time()

    # train_loss = train(model, unlabeled_loader, optimizer, criterion, CLIP)
    train_loss = train(model, train_loader, optimizer, criterion, options.pretrain_clip)

    print("pretrain valid start")
    valid_loss = evaluate(model, valid_loader, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), model_path)

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} |  Val. Loss: {valid_loss:.3f}')


# test 시작
start_time = time.time()
print("start test")

test_loss = evaluate(model, test_loader, criterion)

end_time = time.time()

epoch_mins, epoch_secs = epoch_time(start_time, end_time)

print(f'Time: {epoch_mins}m {epoch_secs}s')
print(f'\tTrain Loss: {train_loss:.3f} |  Val. Loss: {valid_loss:.3f}')

write_embeddings(emb_path,
                 word_embedding.weight.data,
                 id_to_vocab)


