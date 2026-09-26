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


class FirstTokenModel(nn.Module):
    """cell_type만 바꿔 RNN / GRU / LSTM을 같은 조건에서 비교한다."""

    def __init__(self, cell_type: str = "lstm", vocab: int = 10,
                 emb: int = 32, hidden: int = 64):
        super().__init__()
        self.cell_type = cell_type
        self.embedding = nn.Embedding(vocab, emb)
        rnn_cls = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[cell_type]
        self.rnn = rnn_cls(emb, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, vocab)

        if cell_type == "lstm":          # 망각 게이트 편향을 1로 초기화
            for name, param in self.rnn.named_parameters():
                if "bias" in name:
                    param.data[hidden:2 * hidden].fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.embedding(x)
        out, state = self.rnn(e)
        # LSTM은 (h, c) 튜플을, RNN/GRU는 h 하나를 돌려준다
        h_last = state[0] if self.cell_type == "lstm" else state
        return self.fc(h_last[-1])       # 마지막 층의 마지막 은닉 상태


def run(cell_type: str, seq_len: int,
        steps: int = 1500, batch_size: int = 128) -> float:
    device = get_device()
    torch.manual_seed(42)

    model = FirstTokenModel(cell_type).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for _ in range(steps):
        x, y = make_batch(batch_size, seq_len)
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
    lengths = (5, 10, 20, 40, 80)

    print(f"{'cell':>5} | " + " | ".join(f"T={t:<4}" for t in lengths))
    for cell_type in ("rnn", "gru", "lstm"):
        accs = [f"{run(cell_type, t):.1%}".rjust(6) for t in lengths]
        print(f"{cell_type:>5} | " + " | ".join(accs))

    for cell_type in ("rnn", "gru", "lstm"):
        n = sum(p.numel() for p in FirstTokenModel(cell_type).parameters())
        print(f"{cell_type:>5} params: {n:,}")