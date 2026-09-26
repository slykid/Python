import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image


def get_device() -> torch.device:
    if torch.backends.mps.is_available():   # Apple Silicon
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class AutoEncoder(nn.Module):
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),              # 1x28x28 -> 784
            nn.Linear(784, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 64), nn.ReLU(inplace=True),
            nn.Linear(64, latent_dim),  # 병목: 활성화 없이 그대로
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 784),
            nn.Sigmoid(),              # 픽셀값 0~1에 맞춤
            nn.Unflatten(1, (1, 28, 28)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

def main():
    device = get_device()
    print(f"device: {device}")

    tf = transforms.ToTensor()   # 0~1로 스케일만 맞춤 (정규화는 하지 않음)
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=tf)
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=tf)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)

    model = AutoEncoder(latent_dim=32).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, 21):
        model.train()
        train_loss = 0.0
        for images, _ in train_loader:          # 레이블은 쓰지 않습니다
            images = images.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), images)   # 정답이 입력 자신
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                test_loss += criterion(model(images), images).item() * images.size(0)

        print(f"epoch {epoch:2d} | train {train_loss / len(train_ds):.4f} "
              f"| test {test_loss / len(test_ds):.4f}")

    # 원본과 복원 결과를 위아래로 붙여 저장
    model.eval()
    with torch.no_grad():
        sample = next(iter(test_loader))[0][:8].to(device)
        pair = torch.cat([sample, model(sample)])
        save_image(pair.cpu(), "reconstruction.png", nrow=8)


if __name__ == "__main__":   # macOS에서 num_workers > 0 쓸 때 필수
    main()