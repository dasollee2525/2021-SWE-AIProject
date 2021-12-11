from .encoder_decoder import EncoderRNN, DecoderRNN
from .model import SemiPretSeq2Seq
from .train_eval_functions import train, evaluate
from .util_functions import init_weights

from options import options

import torch
import torch.nn as nn
import torch.optim as optim
import time

N_EPOCHS = options.pretrain_epoch
CLIP = options.pretrain_clip
# valid loss 초기값 무한
best_valid_loss = float('inf')


word_embedding = nn.Embedding(options.num_vocab, options.embed_dim)

encoder = EncoderRNN(options.hidden_dim, options.num_layers, options.num_vocab, options.embed_dim, word_embedding, options.dropout)
decoder = DecoderRNN(options.num_vocab, options.hidden_dim, options.num_layers, options.embed_dim, word_embedding, options.dropout)
model = SemiPretSeq2Seq(encoder, decoder, options.num_vocab, device=device).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=options.lr_min)

model.apply(init_weights)

for epoch in range(N_EPOCHS):

    start_time = time.time()

    # train_loss = train(model, unlabeled_loader, optimizer, criterion, CLIP)
    train_loss = train(model, total_loader, optimizer, criterion, CLIP, device)
    valid_loss = evaluate(model, valid_loader, criterion, device)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), emb_path)

    write_embeddings(emb_path,
                     word_embedding.weight.data,
                     id_to_vocab)

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} |  Val. Loss: {valid_loss:.3f}')




