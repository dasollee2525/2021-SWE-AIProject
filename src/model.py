import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTM4VAT(nn.Module):
    def __init__(self, hidden_dim, num_layers, num_vocab, num_classes, embed_dim, embedding=None, dropout=0.1, vat=False, epsilon=1.0, device='cpu'):
        super(BiLSTM4VAT, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(num_vocab, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.vat = vat
        self.epsilon = epsilon
        self.device = device
        self.lstm = nn.LSTM(embed_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(self.hidden_dim * 2, num_classes)
        
    def get_normalized_vector(self, d):
        d_term = torch.sqrt(1e-6 + torch.sum((d ** 2),2)) 
        d_new = d.T / d_term.T
        d_new = d_new.T
        return d_new

    def perturb_embedding(self, x, x_size, g=None, first_step=False):
        if first_step:
            d = torch.randn(x_size, device=self.device, requires_grad=True)
            r = self.get_normalized_vector(d)
            r.retain_grad()
            self.r= r
            x += r
        else:
            r = self.epsilon * self.get_normalized_vector(g)
            x += r
        return x

    def forward(self, x, h0, c0, train=True, g=None, first_step=False):
        x = self.embedding(x)

        # VAT
        if self.vat and train:
            if first_step or g is not None:
                x = self.perturb_embedding(x, x.size(), g, first_step)
        
        # LSTM
        x, _ = self.lstm(x, (h0, c0))
        h_t = torch.tanh(x[:,-1,:])
        h_t = self.dropout(h_t)
        outputs = self.fc(h_t)
        return outputs