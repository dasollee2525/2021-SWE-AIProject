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

