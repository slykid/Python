import torch
import torch.nn as nn


def get_device() -> torch.device:
    if torch.backends.mps.is_available():   # Apple Silicon
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_batch(batch_size: int, seq_len: int, vocab: int = 10):
    """무작위 시퀀스를 만들고, 정답은 시퀀스의 첫 번째 값으로 둔다."""
    x = torch.randint(0, vocab, (batch_size, seq_len))
    y = x[:, 0].clone()
    return x, y


class FirstTokenRNN(nn.Module):
    def __init__(self, vocab: int = 10, emb: int = 32, hidden: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab, emb)
        self.rnn = nn.RNN(emb, hidden, batch_first=True)  # 기본 tanh
        self.fc = nn.Linear(hidden, vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.embedding(x)              # (B, T, emb)
        out, h_last = self.rnn(e)          # out: (B, T, H), h_last: (1, B, H)
        return self.fc(h_last.squeeze(0))  # 마지막 은닉 상태만 사용


def run(seq_len: int, steps: int = 1500, batch_size: int = 128) -> float:
    device = get_device()
    torch.manual_seed(42)

    model = FirstTokenRNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for step in range(steps):
        x, y = make_batch(batch_size, seq_len)
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 폭발 방지
        optimizer.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(20):
            x, y = make_batch(256, seq_len)
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return correct / total


if __name__ == "__main__":
    print(f"device: {get_device()}")
    for seq_len in (5, 10, 20, 40, 80):
        acc = run(seq_len)
        print(f"seq_len={seq_len:3d} | test acc {acc:.2%}")