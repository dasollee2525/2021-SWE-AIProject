class Options(object):
    max_len = 50 # maximum comment length
    min_count = 0 # minimum number to include vocab list
    batch_size = 64
    num_layers = 1
    hidden_dim = 1024
    embed_dim = 256
    dropout = 0.1
    lr = 0.001
    pretrain_epoch = 10
    pretrain_clip = 0.1