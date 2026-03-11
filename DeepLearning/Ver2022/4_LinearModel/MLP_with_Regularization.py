import os
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split

from torchvision.datasets import FashionMNIST
from torchvision import transforms

from sklearn.metrics import roc_curve, roc_auc_score

from tqdm import tqdm

import matplotlib
from matplotlib import pyplot as plt

matplotlib.use("MacOSX")
plt.style.use("seaborn-v0_8")

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS 장치를 사용할 수 있습니다.")
else:
    device = torch.device("cpu")
    print("MPS 장치를 찾을 수 없어 CPU로 설정합니다.")

data_root = os.path.join(os.getcwd(), "Dataset/")

# 전처리 부분 & 데이터 셋 정의
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

# Data Load
fashion_mnist_dataset = FashionMNIST(data_root, download=True, train=True, transform=transform)

# check data
fashion_mnist_dataset[0]

# Data Split
dataset = random_split(
    fashion_mnist_dataset,
    [int(len(fashion_mnist_dataset) * 0.7), len(fashion_mnist_dataset) - int(len(fashion_mnist_dataset) * 0.7)],
)

train_dataset = dataset[0]
valid_dataset = dataset[1]

train_batch_size = 100
valid_batch_size = 10

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=1
)

valid_dataloader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=valid_batch_size, shuffle=True, num_workers=1
)

# check Dataloader
for sample_batch in train_dataloader:
    print(sample_batch)
    print(sample_batch[0].shape, sample_batch[1].shape)
    break

# Modeling
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden1_dim: int, hidden2_dim: int, output_dim: int):
        super().__init__()

        self.linear1 = nn.Linear(input_dim, hidden1_dim)
        self.linear2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.linear3 = nn.Linear(hidden2_dim, output_dim)
        self.relu = F.relu

    def forward(self, input):
        x = torch.flatten(input, start_dim=1)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        output = self.linear3(x)

        return output

class MLPWithDropout(MLP):
    def __init__(self, input_dim: int, hidden1_dim: int, hidden2_dim: int, output_dim: int,  dropout_prob: float):
        super().__init__(input_dim, hidden1_dim, hidden2_dim, output_dim)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)

    def forward(self, input):
        x = torch.flatten(input, start_dim=1)
        x = self.relu(self.linear1(x))
        x = self.dropout1(x)
        x = self.relu(self.linear2(x))
        x = self.dropout2(x)
        output = self.linear3(x)

        return output

# model = MLP(28*28, 128, 64, 10)
model = MLPWithDropout(28*28, 128, 64, 10, 0.3)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
max_epochs = 15

writer = SummaryWriter()
log_interval = 100
step = 0
for epoch in range(1, max_epochs + 1):
    # Validation Step
    # Random Network 일 때 디버깅의 역할을 수행해주기 때문에 validation을 먼저 해보는 것을 추천
    # 단, optimizer 는 업데이트 하지 않는 것을 고정!
    with torch.no_grad():
        val_loss = 0.0
        val_correct = 0
        model.eval()

        for val_batch_idx, (val_images, val_labels) in enumerate(tqdm(valid_dataloader, position=0, leave=True, desc="Validation")):
            # forward
            val_outputs = model(val_images)
            _, val_preds = torch.max(val_outputs, 1)

            # loss & acc
            val_loss = loss_function(val_outputs, val_labels) / val_outputs.shape[0]
            val_correct = torch.sum(val_preds == val_labels.data) / val_outputs.shape[0]

    # Validation Logging
    val_epoch_loss = val_loss / len(valid_dataloader)
    val_epoch_acc = val_correct / len(valid_dataloader)

    print(
        f"\n{epoch} epoch, {step} step: val loss: {val_epoch_loss:.4f}, val acc: {val_epoch_acc:.4f}\n"
    )

    writer.add_scalar("Loss/val", val_epoch_loss, step)
    writer.add_scalar("Accuracy/val", val_epoch_acc, step)
    writer.add_images("Images/val", val_images, step)

    current_loss = 0
    current_corrects = 0
    model.train()

    # Train step
    for batch_idx, (images, labels) in enumerate(tqdm(train_dataloader, position=0, leave=True, desc="Train")):
        current_loss = 0.0
        current_corrects = 0

        # get prediction
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        # get loss
        loss = loss_function(outputs, labels)

        # Backpropagation
        # optimizer initialize
        optimizer.zero_grad()

        # Perform backward pass
        loss.backward()

        # Perform Optimization
        optimizer.step()

        current_loss += loss.item()
        current_corrects += torch.sum(preds == labels.data)

        if step % log_interval == 0:
            train_loss = current_loss / log_interval
            train_acc = current_corrects / log_interval

            print(
                f"\n{step}: train loss: {train_loss:.4f}, train acc: {train_acc:.4f}"
            )

            writer.add_scalar("Loss/train", train_loss, step)
            writer.add_scalar("Accuracy/train", train_acc, step)
            writer.add_images("Images/train", images, step)
            writer.add_graph(model, images)
            current_loss = 0
            current_corrects = 0

        step += 1

# save model
os.makedirs("./logs/models", exist_ok=True)
torch.save(model, "./logs/models/mlp.chkpt")

# load model
loaded_model = torch.load("./logs/models/mlp.chkpt", weights_only=False)
loaded_model.eval()
print(loaded_model)


def softmax(x, axis=0):
    "numpy softmax"
    max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - max)
    sum = np.sum(e_x, axis=axis, keepdims=True)
    f_x = e_x / sum

    return f_x

test_batch_size = 100
test_dataset = FashionMNIST(data_root, download=True, train=False, transform=transforms.ToTensor())
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=1)

test_labels_list = []
test_preds_list = []
test_outputs_list = []

for i, (test_images, test_labels) in enumerate(tqdm(test_dataloader, position=0, leave=True, desc="ReinforcementLearning")):
    # forward
    test_outputs = loaded_model(test_images)
    _, test_preds = torch.max(test_outputs, 1)

    final_outputs = softmax(test_outputs.detach().numpy(), axis=1)
    test_outputs_list.extend(final_outputs)
    test_preds_list.extend(test_preds.detach().numpy())
    test_labels_list.extend(test_labels.detach().numpy())

test_preds_list = np.array(test_preds_list)
test_labels_list = np.array(test_labels_list)

print(f"\nacc: {np.mean(test_labels_list == test_preds_list)*100:.4f}%")

# ROC Curve
fpr = {}
tpr = {}
thresholds = {}

n_class = 10

for i in range(n_class):
    fpr[i], tpr[i], thresholds[i] = roc_curve(test_labels_list, np.array(test_outputs_list)[:, i], pos_label=i)

for i in range(n_class):
    plt.plot(fpr[i], tpr[i], linestyle='--', label=f"Class {i} vs. Rest")
plt.title("Multi-class ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="best")
plt.show()

print("auc_score", roc_auc_score(test_labels_list, test_outputs_list, multi_class='ovo', average='macro'))