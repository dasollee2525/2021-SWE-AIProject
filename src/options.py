class Options(object):
    unlabeled_ratio = 1 # labeled : unlabeled ratio
    max_len = 50 # maximum comment length
    min_count = 0 # minimum number to include vocab list
    batch_size = 64
    pretrained = False
    num_layers = 1
    hidden_dim = 1024
    embed_dim = 256
    dropout = 0.1
    vat = False
    epsilon = 1e+6 # perturbation size
    alpha = 1.0 # regularization coefficient
    lr = 0.001
    factor = 0.5
    patience = 3
    verbose = True
    epochs = 100