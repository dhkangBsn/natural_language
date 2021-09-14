"""
training RNN for simple sentiment analysis.
"""

from typing import List, Tuple
import pandas as pd
import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import logging
from sys import stdout
logging.basicConfig(stream=stdout, level=logging.INFO)


class SimpleRNN(torch.nn.Module):
    """
    Many-to-one RNN, to be used for sentiment analysis.
    """
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int):
        super().__init__()
        # hyper parameters
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        # parameters to optimize
        self.Embeddings = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.W_hh = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.W_xh = torch.nn.Linear(in_features=embed_size, out_features=hidden_size)
        self.W_hy = torch.nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: (N, L) =  정수인코딩된 토큰들.
        :return:
        """
        # always start with zeros.
        H_t = torch.zeros(size=(X.shape[0], self.hidden_size))
        # the forward step of RNN is sequential.
        for t in range(X.shape[1]):
                # 현재 시간대의 입력을 다 가져오기.
                X_t = X[:, t]  # (N, L) -> (N,)
                H_t_m_1 = H_t  # 과거 시간대의 hidden state.
                Embeds = self.Embeddings(X_t)  # (N, L) -> (N, E)
                Xh = self.W_xh(Embeds)  # (N, E) * (E, H) -> (N, H)
                Hh = self.W_hh(H_t_m_1)  # (N, H) * (H, H) -> (N, H)
                # update the current hidden state.
                H_t = torch.tanh(Xh + Hh)
        # return the latest hidden state.
        return H_t

    def training_step(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        compute the loss here.
        :param X: (N, L)
        :param y: (N,) - 0 (부정) 또는 1 (긍정)의 나열.
        :return: The loss
        """
        H_t = self.forward(X)  # (N, L) -> (N, H)
        Hy = self.W_hy(H_t)  # (N, H) * (H, 1) -> (N, 1)
        Hy_normalized = torch.sigmoid(Hy)  # (N, 1) -> (N, 1). 값이  [0-1] 사이로 정규화된다!
        Hy_normalized = torch.reshape(Hy_normalized, y.shape)
        loss = F.binary_cross_entropy(Hy_normalized, y)  # (N,) -> (N,)
        loss = loss.sum()   # (N,1) -> (1). 배치 속 모든 로스를 계산해준다!
        return loss

    def predict(self, X: torch.Tensor) -> float:
        H_t = self.forward(X)
        Hy = self.W_hy(H_t)
        return torch.sigmoid(Hy).item()


class SimpleSentimentAnalyser:
    def __init__(self, rnn: SimpleRNN, tokenizer: Tokenizer):
        # rnn과, 학습된 토크나이저를 입력으로 받는다.
        self.rnn = rnn
        self.tokenizer = tokenizer

    def __call__(self, text: str) -> float:
        X = build_X(sents=[text], tokenizer=self.tokenizer)
        return self.rnn.predict(X)


class SimpleDataset(Dataset):
    """
    DataLoader를 사용하기 위한.. Dataset.
    """
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        """
        Returning the size of the dataset
        :return:
        """
        return self.y.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.LongTensor]:
        """
        Returns features & the label
        :param idx:
        :return:
        """
        return self.X[idx], self.y[idx]


# hyper parameters
EPOCHS = 20
EMBED_SIZE = 16
HIDDEN_SIZE = 256
BATCH_SIZE = 32
LR = 0.001


# --- builders: for X and y--- #
def build_X(sents: List[str], tokenizer: Tokenizer) -> torch.LongTensor:
    seqs = tokenizer.texts_to_sequences(sents)  # 토크나이즈 & 정수인코딩 된것들.
    seqs = pad_sequences(seqs, padding="post", value=0)  # 가장 길이가 긴것에 맞추어서 패딩을 진행.
    X = torch.LongTensor(seqs)
    return X


def build_y(labels: List[int]) -> torch.FloatTensor:
    y = torch.FloatTensor(labels)
    return y


DATA = [
    ("유빈이는 애플을 좋아해", 1),
    ("혁이는 삼성을 좋아해", 1),
    ("나는 널 좋아해", 1),
    ("유빈이는 애플을 싫어해", 0),
    ("혁이는 삼성을 싫어해", 0),
    ("나는 널 싫어해", 0),
]


def main():
    global EMBED_SIZE, BATCH_SIZE, EPOCHS
    sents = [sent for sent, _ in DATA]
    labels = [label for _, label in DATA]
    tokenizer = Tokenizer(char_level=False)
    tokenizer.fit_on_texts(sents)
    # building X
    X = build_X(sents, tokenizer)
    # building y
    y = build_y(labels)
    # construct a dataset
    dataset = SimpleDataset(X, y)
    # calculate the size of the vocab
    vocab_size = len(tokenizer.word_index) + 1
    rnn = SimpleRNN(vocab_size=vocab_size, embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE)
    # we are using an Adam Optimizer.
    optimizer = optim.RMSprop(params=rnn.parameters(), lr=LR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for e_idx, epoch in enumerate(range(EPOCHS)):
        losses = list()
        for b_idx, batch in enumerate(dataloader):
            X, y = batch
            loss = rnn.training_step(X, y)
            # what is zero_grad for?
            # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
            optimizer.zero_grad()  # resetting the gradients.
            loss.backward()  # backprop the loss
            optimizer.step()  # gradient step
            losses.append(loss.item())
        avg_loss = (sum(losses) / len(losses))
        print("-----epoch={}, avg_loss={}----".format(e_idx, avg_loss))

    # then, we do some inference
    rnn.eval()
    print("--- test ---")
    analyser = SimpleSentimentAnalyser(rnn, tokenizer)
    for sent, label in DATA:
        print(analyser(sent), label)
    print(analyser("나는 오이를 좋아해"))
    # 결국 학습하게 되는것은 ... 좋아해! 싫어해! 뿐.
    print(analyser("라라라라라라 좋아해"))
    print(analyser("라라라라라라 싫어해"))


if __name__ == '__main__':
    main()
