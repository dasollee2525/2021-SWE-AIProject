import torch
import torch.nn as nn


class SemiPretSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, num_vocab, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        # sos 없이 진행
        self.num_vocab = num_vocab

    def forward(self, src):
        # input 형태가 [batch size, sequence length] 이므로
        # input 의 batch size 는 0 번째 dim 값으로 알 수 있다.
        batch_size = src.shape[0]
        # input 문장의 길이는 sequence length 에 해당하므로 1번째 dim 으로 가져옴
        trg_len = src.shape[1]
        # vocab 전체 개수는 decoder 의 output_dim 으로 가져옴
        trg_vocab_size = self.decoder.output_dim

        # decoder 에서 return 하는 prediction 값을 저장하는 tensor
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        # encoder 에 input 문장 batch 전달하고 hidden, cell return 값 받아옴
        hidden, cell = self.encoder(src)

        # decoder에 처음 넣어주는 input 값 (원래는 sos여야 함)
        # 문장의 첫 토큰들
        input = src[:, 0]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)

            output = torch.unsqueeze(output, dim=1)

            # decoder 에서 생성한 prediction을 저장
            outputs[:, t:, :] = output
            outputs[:, t + 1:, :] = 0

            # 원래/실제 input 값을 넣어준다
            input = src[:, t]

        return outputs
