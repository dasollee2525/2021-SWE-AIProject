import torch
import torch.nn as nn


def train(model, train_loader, optimizer, criterion, clip, device):
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
def evaluate(model, iterator, criterion, device):
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