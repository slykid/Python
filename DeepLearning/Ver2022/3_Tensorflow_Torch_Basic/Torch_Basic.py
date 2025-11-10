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
if torch.cuda.is_available():
    tensor = tensor.to('cuda')

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {type(tensor)}")
print(f"Device of tensor: {tensor.device}")


# indexing & slicing
a = torch.arange(1, 13).reshape(3, 4)
print(a)