import math
import torch
import torch.nn as nn


def get_device() -> torch.device:
    if torch.backends.mps.is_available():   # Apple Silicon
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_batch(batch_size: int, seq_len: int, vocab: int = 10):
    x = torch.randint(0, vocab, (batch_size, seq_len))
    y = x[:, 0].clone()          # 정답은 맨 앞 숫자
    return x, y


class PositionalEncoding(nn.Module):
    """사인·코사인 위치 인코딩 (학습 파라미터 없음)"""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class FirstTokenTransformer(nn.Module):
    def __init__(self, use_pos: bool, vocab: int = 10,
                 d_model: int = 64, nhead: int = 4, layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab, d_model)
        self.pos = PositionalEncoding(d_model) if use_pos else None
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,
            batch_first=True, norm_first=True,      # Pre-LN
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.fc = nn.Linear(d_model, vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embedding(x)
        if self.pos is not None:
            h = self.pos(h)
        h = self.encoder(h)              # (B, T, d)
        return self.fc(h.mean(dim=1))    # 평균 풀링 후 분류


def run(use_pos: bool, seq_len: int, steps: int = 1500, batch_size: int = 128):
    device = get_device()
    torch.manual_seed(42)

    model = FirstTokenTransformer(use_pos).to(device)
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

    print(f"{'model':>18} | " + " | ".join(f"T={t:<4}" for t in lengths))
    for use_pos in (False, True):
        name = "Transformer+위치" if use_pos else "Transformer(위치 없음)"
        accs = [f"{run(use_pos, t):.1%}".rjust(6) for t in lengths]
        print(f"{name:>18} | " + " | ".join(accs))

    n = sum(p.numel() for p in FirstTokenTransformer(True).parameters())
    print(f"\n파라미터 수: {n:,}")
