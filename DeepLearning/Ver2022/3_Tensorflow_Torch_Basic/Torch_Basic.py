import os
import sys
import datetime
import numpy as np
import pandas as pd

import matplotlib
from matplotlib import pyplot as plt
from PIL import Image


import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose

matplotlib.use('MacOSX')

print(torch.__version__)

# MNIST Dataset Load
train_data = datasets.MNIST(root='Dataset', train=True, download=True, transform=ToTensor())
test_data = datasets.MNIST(root='Dataset', train=False, download=True, transform=ToTensor())

batch_size = 64

# DataLoader 생성
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

for X, y in test_loader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

# Apple Silcon Mac 전용
device = "mps" if torch.mps.is_available() else "cpu"
print("Using {} device.".format(device))

# CUDA 전용
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print("Using {} device.".format(device))

# Model
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)

        return logits

model = NeuralNet().to(device)
print(model)

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train(dataloader, model, loss_func, optimizer):
    size = len(dataloader.dataset)

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Loss 계산
        pred = model(X)
        loss = loss_func(pred, y)

        # Back Propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:5d}]")

def test(dataloader, model, loss_func):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_func(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size

        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f} %, Avg loss: {test_loss:>8f} \n")

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n------------------------------------------")
    train(train_loader, model, loss_func, optimizer)
    test(test_loader, model, loss_func)
print("Done")
# 최종 Accuray: 97.8%, Avg.Loss: 0.07


# image.png 업로드!
# 업로드 완료 후 아래 코드 실행
current_dir = os.getcwd()
img_path = os.path.join(current_dir, "image.png")

# image 파일 읽기
current_img = Image.open(img_path)

# 이미지 크기 조정
current_img = current_img.resize((28, 28))
image = np.array(current_img)

# color image일 경우 RGB 평균값의 GrayScale로 조정
try:
    image = np.mean(image, axis=2)
except:
    pass

image = np.abs(255 - image)
image = image.astype(np.float32) / 255.0

plt.imshow(image, cmap="gray")
plt.show()


# 이미지 to Tensor
image = torch.as_tensor(image).to(device).reshape(1, 1, 28, 28)
model.eval()

pred = model(image)
print("Model이 예측한 값은 {} 입니다.".format(pred.argmax(1).item()))

# Tensor
data = [[1, 2], [3, 4]]
x_data = torch.Tensor(data)
print(x_data)

## numpy array to tensor
np_array = np.array(data)
x_np_1 = torch.tensor(np_array)
print(x_np_1)

x_np_2 = torch.as_tensor(np_array)
print(x_np_2)

x_np_3 = torch.from_numpy(np_array)
print(x_np_3)

x_np_1[0, 0] = 5
print(x_np_1)
print(np_array)

x_np_2[0, 0] = 6
print(x_np_2)
print(np_array)

x_np_3[0, 0] = 7
print(x_np_3)
print(np_array)
# torch.tensor() 를 사용할 경우에는 np_array 아예 카피를 하여 새로운 텐서 객체를 만드는 반면,
# torch.as_tensor() 또는 torch.from_numpy() 의 경우에는 객체의 복사가 아닌 주소를 복사해서 생성하는 과정이므로, 값이 변경될 경우 원본도 같이 변경됨

np_again = x_np_1.numpy()
print(np_again, type(np_again))

# ones(), zeros(), full(), empty()
a = torch.ones(2, 3)
b = torch.zeros(2, 3)
c = torch.full((2, 3), 2)
d = torch.empty(2, 3)

print(a)
print(b)
print(c)
print(d)

# zeros_like(), ones_like(), full_like(), empty_like()
e = torch.zeros_like(c)
f = torch.ones_like(c)
g = torch.full_like(c, 3)
h = torch.empty_like(c)

print(e)
print(f)
print(g)
print(h)

i = torch.eye(3)  # 단위행렬 생성
print(i)

j = torch.arange(10)
print(j)

k = torch.rand(2, 2)
l = torch.randn(2, 2)
print(k)
print(l)


# Tensor의 속성
tensor = torch.rand(3, 4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {type(tensor)}")
print(f"Device of tensor: {tensor.device}")

# 속성 변경하기
tensor = tensor.reshape(4, 3)
tensor = tensor.int()
if torch.mps.is_available():
    tensor = tensor.to('mps')

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {type(tensor)}")
print(f"Device of tensor: {tensor.device}")


# indexing & slicing
a = torch.arange(1, 13).reshape(3, 4)
print(a)

print(a[1])
print(a[0, -1])

# transpose
a = torch.arange(16).reshape(2, 2, 4)
print(a, a.shape)

b = a.transpose(1, 2)
print(b, b.shape)

c = a.permute((2, 0, 1))  # Tensorflow의 transpose와 동일
print(c, c.shape)

# concat & stack
a = torch.arange(24).reshape(4, 6)
b = a.clone().detach()  # torch에서만 행렬을 복사하는 방법임!
print(a, a.shape)
print(b, b.shape)

c = torch.cat([a, b], axis=0)
print(c, c.shape)

c = torch.cat([a, b], axis=1)
print(c, c.shape)

d = torch.stack([a, b], axis=0)
print(d, d.shape)

d = torch.stack([a, b], axis=-1)
print(d, d.shape)


# Dataset / DataLoader
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tr

training_data = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=ToTensor()
)

labels_map = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    image, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)

    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(image.squeeze(), cmap="gray")

plt.show()

training_data[0]

# DataLoader
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

train_features, train_labels = next(iter(train_dataloader))
print(f"Features batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

image = train_features[0].squeeze()
label = train_labels[0]

plt.imshow(image, cmap="gray")
plt.show()
print(f"Label: {label}")


# Custom Dataset, Custom DataLoader 만들기
class CustomDataset(Dataset):
    def __init__(self, np_data, transform=None):
        self.data = np_data
        self.transform = transform
        self.len = np_data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample

def square(sample):
    return sample**2

trans = tr.Compose([square])

np_data = np.arange(10)
custom_dataset = CustomDataset(np_data, transform=trans)

custom_dataloader = DataLoader(custom_dataset, batch_size=2, shuffle=True)
for _ in range(3):
    for data in custom_dataloader:
        print(data)
    print("=" * 20)

# Set up Modeling
device = 'mps' if torch.mps.is_available() else 'cpu'
print('Using device:', device)

# Model Class 생성하기
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

X = torch.rand(1, 28, 28, device=device)
logits = model(X)
y_prob = nn.Softmax(dim=1)(logits)
y_pred = y_prob.argmax(1)

print(f"Predicted class: {y_pred}")

# Training
## Loss Function
loss_func = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        # Back Prop.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}   [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg. loss: {test_loss:>8f} \n")

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-----------------------------")
    train_loop(train_dataloader, model, loss_func, optimizer)
    test_loop(test_dataloader, model, loss_func)

print("Done!")


# Model save & load
## Parameter만 저장
torch.save(model.state_dict(), "model_weights.pth")

model2 = NeuralNetwork().to(device)
print(model2)

model2.eval()  # 해당 모델이 학습 상태가 아님을 표시하는 함수
test_loop(test_dataloader, model2, loss_func)
# ReinforcementLearning Error:
#  Accuracy: 5.5%, Avg. loss: 2.324286


model2.load_state_dict(torch.load("model_weights.pth"))
model2.eval()
test_loop(test_dataloader, model2, loss_func)
# ReinforcementLearning Error:
#  Accuracy: 88.2%, Avg. loss: 0.328638

## Model 전체 저장
torch.save(model, 'model.pth')
model3 = torch.load('model.pth', weights_only=False)  # 2.6 버전 이상 부터는 weights_only=False로 해야 모델 전체를 불러옴
model3.eval()
test_loop(test_dataloader, model3, loss_func)
