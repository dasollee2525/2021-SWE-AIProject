import torch
import torch.nn as nn


# 문장 batch을 받아서 처리하는 encoder unit
class EncoderRNN(nn.Module):
    # input_size: vocab 전체 개수(word embedding 생성 위함), hidden_size: hidden state 의 feature 개수
    def __init__(self, hidden_dim, num_layers, embed_dim, embed, dropout=0.1):
        super(EncoderRNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.embed_dim = embed_dim
        self.embedding = embed

        # 1 layer LSTM
        # input 크기: embed_dim, hidden 크기: hidden_dim
        self.lstm = nn.LSTM(self.embed_dim, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=True)

        # 디코더는 one-directional로 처리해줬기 때문에 hidden_dim 사이즈 축소
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, input):
        # 해당 1개 단어의 embedding
        # 3D shape로 형성
        # input 문장을 word embedding으로 각각 전환 후, dropout 적용
        embedded = self.dropout(self.embedding(input))

        # embedding 처리된 전체 문장을 lstm에 넘기면, recurrent 계산 수행
        output, (hidden, cell) = self.lstm(embedded)

        # print("[encoder] hidden shape: ", hidden.shape)

        # forward hidden 과 backward hidden을 합쳐서 linear -> tanh 처리
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

        cell = torch.tanh(self.fc(torch.cat((cell[-2, :, :], cell[-1, :, :]), dim=1)))

        hidden = torch.unsqueeze(hidden, 0)
        cell = torch.unsqueeze(cell, 0)

        return hidden, cell


# 단어 batch을 받아서 처리하는 decoder unit
# output_dim: vocab 전체 크기 (len)
class DecoderRNN(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers, embed_dim, embed, dropout=0.1):
        super(DecoderRNN, self).__init__()
        # hidden state 의 size 정의
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.embed_dim = embed_dim
        self.embedding = embed
        self.output_dim = output_dim

        # 디코더는 one-directional 처리
        self.lstm = nn.LSTM(self.embed_dim, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=False)

        # one-directional 이어서 hidden_dim 으로 그대로 실행
        self.out = nn.Linear(hidden_dim, self.output_dim)

    def forward(self, input, hidden, cell):
        # 토큰 하나씩 디코딩 하므로 sequence length = 1 -> sentence length 차원 추가
        # 단어 하나씩 처리하기 때문에 차원 추가
        input = input.unsqueeze(1)

        # 문장에서 각 단어마다 dropout을 적용한 word embedding 생성
        embedded = self.dropout(self.embedding(input))

        # lstm 에서 단어를 input으로 넣어줌 (sequence length = 1)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))

        # lstm의 output(각 timestep의 마지막 layer hidden 값)을 넣으면
        # nn.Linear에서 정의한 output_dim 차원의 결과를 뱉어낸다
        prediction = self.out(output.squeeze(1))

        # 3D output, 3D hidden
        # hidden, cell 은 one-directional
        return prediction, hidden, cell

