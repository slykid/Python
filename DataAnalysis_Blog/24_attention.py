import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    """학습되는 Query 하나로 시퀀스 전체를 가중합한다."""

    def __init__(self, hidden: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(hidden) * 0.1)
        self.scale = hidden ** 0.5

    def forward(self, h: torch.Tensor):
        # h: (B, T, H) — 인코더의 모든 시점 상태
        scores = (h @ self.query) / self.scale       # (B, T)   1단계: 점수
        weights = torch.softmax(scores, dim=1)       # (B, T)   2단계: 비율
        context = torch.bmm(weights.unsqueeze(1), h).squeeze(1)   # 3단계: 섞기
        return context, weights                      # (B, H), (B, T)

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


class FirstTokenModel(nn.Module):
    """use_attention=False면 29편의 RNN과 동일한 구조."""

    def __init__(self, use_attention: bool, vocab: int = 10,
                 emb: int = 32, hidden: int = 64):
        super().__init__()
        self.use_attention = use_attention
        self.embedding = nn.Embedding(vocab, emb)
        self.rnn = nn.RNN(emb, hidden, batch_first=True)
        self.attn = AttentionPooling(hidden) if use_attention else None
        self.fc = nn.Linear(hidden, vocab)

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        out, h_last = self.rnn(self.embedding(x))   # out: (B, T, H)
        if self.use_attention:
            context, weights = self.attn(out)       # 모든 시점을 본다
        else:
            context, weights = h_last[-1], None     # 마지막 상태만 본다
        logits = self.fc(context)
        return (logits, weights) if return_weights else logits

def run(use_attention: bool, seq_len: int,
        steps: int = 1500, batch_size: int = 128):
    device = get_device()
    torch.manual_seed(42)

    model = FirstTokenModel(use_attention).to(device)
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
    return correct / total, model


if __name__ == "__main__":
    print(f"device: {get_device()}")
    lengths = (5, 10, 20, 40, 80)

    print(f"{'model':>10} | " + " | ".join(f"T={t:<4}" for t in lengths))
    for use_attn in (False, True):
        name = "RNN+어텐션" if use_attn else "RNN"
        accs = [f"{run(use_attn, t)[0]:.1%}".rjust(6) for t in lengths]
        print(f"{name:>10} | " + " | ".join(accs))

    # 학습된 어텐션이 어디를 보는지 확인
    _, model = run(True, 40)
    model.eval()
    with torch.no_grad():
        x, _ = make_batch(512, 40)
        _, w = model(x.to(get_device()), return_weights=True)
    avg = w.mean(dim=0).cpu()
    print("\n위치별 평균 어텐션 가중치")
    print(f"  0번(정답 위치): {avg[0]:.3f}")
    print(f"  1~9번 평균    : {avg[1:10].mean():.3f}")
    print(f"  나머지 평균   : {avg[10:].mean():.3f}")