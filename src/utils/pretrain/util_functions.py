import torch
import torch.nn as nn
from tqdm import tqdm


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def write_embeddings(path, embeddings, id_to_vocab):
    with open(path, 'w') as f:
        for i, embedding in enumerate(tqdm(embeddings)):
            word = id_to_vocab[i]
            vector = ' '.join(str(i) for i in embedding.tolist())
            f.write(f'{word} {vector}\n')


# epoch 실행 시간 구하는 함수
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

