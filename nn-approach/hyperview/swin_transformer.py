# %% [markdown]
# ## Imports

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import pandas as pd
import numpy as np

from torchvision import models
from torchvision import transforms as T
import random
from random import random as rnd

from glob import glob
import os

from sklearn.cluster import KMeans

# %% [markdown]
# ## Constants

# %%
SEED = 42
VL_SPLIT  = 0.2
LR = 1e-4
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

WIDTH = 128
HEIGHT = 128
device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed) # seed the global NumPy random number generator(RNG)
    torch.manual_seed(seed) # seed the RNG for all devices(both CPU and CUDA) 

set_random_seed(seed = SEED)

base_path = "/data/"

gt_path = base_path + 'train_gt.csv'
wavelength_path = base_path + 'wavelengths.csv'


# %% [markdown]
# ## Data imports

# %%
class ReduceChannels(nn.Module):
    def __init__(self, in_channels=150, out_channels=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class GaussianNoise(torch.nn.Module):
    def __init__(self, mean=0.0, std=0.005, p=0.5):
        super().__init__()
        self.std = std
        self.mean = mean
        self.p = p

    def forward(self, img):
        if random.random() < self.p:
            noise = torch.randn_like(img) * self.std + self.mean
            return img + noise
        return img

train_transform = T.Compose([
    ReduceChannels(),  # Resize allinea con Albumentations
    GaussianNoise(std=0.005, p=0.5),  # GaussNoise simulato
    T.RandomRotation(90),  # RandomRotate90
    T.RandomResizedCrop((WIDTH, HEIGHT), scale=(0.95, 1.05), ratio=(0.75, 1.33)),  # RandomResizedCrop
    T.RandomHorizontalFlip(p=0.5),  # Flip orizzontale casuale
    T.RandomVerticalFlip(p=0.5),  # Flip verticale casuale (equivalente a Flip generico)
    T.RandomAffine(degrees=90, translate=(0.05, 0.05)),  # ShiftScaleRotate (senza scaling)
])

eval_transform = T.Compose([
    ReduceChannels(),
    T.Resize((WIDTH, HEIGHT)),
])

# %%
gt_df = pd.read_csv(gt_path)
wavelength_df = pd.read_csv(wavelength_path)


def load_data(directory: str, tr = None):
    data = []
    sizes = set()
    all_files = np.array(
        sorted(
            glob(os.path.join(directory, "*.npz")),
            key=lambda x: int(os.path.basename(x).replace(".npz", "")),
        )
    )
    for file_name in all_files:
        with np.load(file_name) as npz:
            
            arr = npz['data']
            mask = npz["mask"]
            
            arr = torch.tensor(arr, dtype=torch.float32)
            mask = torch.tensor(~mask, dtype=torch.float32)
            
            arr = arr * mask
            
            if tr:
                arr = tr(arr)

        sizes.add((arr.shape[1], arr.shape[2]))
        data.append(arr)
    return data, sizes


def load_gt(file_path: str):
    gt_file = pd.read_csv(file_path)
    labels = gt_file[["P", "K", "Mg", "pH"]].values
    return labels

# X_train_grouped, sizes = load_data(base_path + "train_data")
X_train_base, _ = load_data(base_path + "train_data", tr = train_transform)
# X_test_grouped, _ = load_data(base_path + "test_data")
X_test_base, _ = load_data(base_path + "test_data", tr = eval_transform)
y_train_base = load_gt(base_path + "train_gt.csv")

# %% [markdown]
# ## Model Definition

# %%
non_pretrained_model = models.swin_v2_b()
pretrained_model = models.swin_v2_b(weights= models.Swin_V2_B_Weights.DEFAULT)

num_features = pretrained_model.head.in_features
pretrained_model.head = torch.nn.Linear(num_features, 4)
non_pretrained_model.head = torch.nn.Linear(num_features, 4)

non_pretrained_model = non_pretrained_model.to(device)
pretrained_model = pretrained_model.to(device)

# %%
# values = list(sizes)

# kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
# kmeans.fit(values)

# centroids = kmeans.cluster_centers_.astype(int)

# def assegna_gruppo(tupla, centroids):
#     distanze = [np.linalg.norm(tupla - centroide) for centroide in centroids]
#     return np.argmin(distanze)

# transformers = [
#     transforms.Resize(size=(centroids[0][0], centroids[0][1])),
#     transforms.Resize(size=(centroids[1][0], centroids[1][1])),
#     transforms.Resize(size=(centroids[2][0], centroids[2][1]))
# ]

# gruppi = {i: [] for i in range(len(centroids))}
# target_gruppi = {i: [] for i in range(len(centroids))}
# for idx, t in enumerate(X_train_grouped):
#     gruppo = assegna_gruppo(list(t.shape[1:]), centroids)
#     gruppi[gruppo].append(transformers[gruppo](t))
#     target_gruppi[gruppo].append(y_train_base[idx]) 

# X_train_sml = torch.stack(gruppi[1])
# X_train_mid = torch.stack(gruppi[0])
# X_train_big = torch.stack(gruppi[2])

# y_train_sml = torch.tensor(target_gruppi[1], dtype=torch.float32)
# y_train_mid = torch.tensor(target_gruppi[0], dtype=torch.float32)
# y_train_big = torch.tensor(target_gruppi[2], dtype=torch.float32)

# gruppi = {i: [] for i in range(len(centroids))}
# for idx, t in enumerate(X_test_grouped):
#     gruppo = assegna_gruppo(list(t.shape[1:]), centroids)
#     gruppi[gruppo].append(transformers[gruppo](t))
    
# X_test_sml = torch.stack(gruppi[1])
# X_test_mid = torch.stack(gruppi[0])
# X_test_big = torch.stack(gruppi[2])

X_train = torch.stack(X_train_base)
X_test = torch.stack(X_test_base)
y_train = torch.Tensor(y_train_base)

# print(X_train_sml.shape)
# print(X_train_mid.shape)
# print(X_train_big.shape)

# print(X_test_sml.shape)
# print(X_test_mid.shape)
# print(X_test_big.shape)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)

# %%
dataset=TensorDataset(X_train,y_train)

n = int(len(X_train) * VL_SPLIT)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n, len(dataset) - n])

train_dataloader = DataLoader(train_dataset,batch_size=32,shuffle=True)
val_dataloader = DataLoader(val_dataset,batch_size=32,shuffle=True) 

testset=TensorDataset(X_test)

test_dataloader= DataLoader(testset,shuffle=False)

# %% [markdown]
# ## Training

# %%
criterion = nn.MSELoss()

pretrained_optimizer = torch.optim.SGD(pretrained_model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
non_pretrained_optimizer = torch.optim.SGD(non_pretrained_model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

def train_one_epoch(m,o):
    running_loss = 0.
    num_samples = 0

    for _, data in enumerate(train_dataloader):
        inputs, labels = data

        o.zero_grad()

        outputs = m(inputs.to(device))
        
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        o.step()

        running_loss += loss.item()
        num_samples += inputs.size(0)
        
    train_loss = running_loss / len(train_dataloader)

    return train_loss

# %%
def train(m,o,path = ""):
    epoch_number = 0
    tr_acc = 0.0
    best_vloss = 1_000_000.

    for epoch in range(100):
        print('============= EPOCH {} ============='.format(epoch_number + 1))

        m.train(True)
        avg_loss = train_one_epoch(m, o)

        running_vloss = 0.0
        val_acc = 0.0
        num_vcorrect = 0
        num_vsamples = 0

        m.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(val_dataloader):
                vinputs, vlabels = vdata
                voutputs = m(vinputs.to(device))
                vloss = criterion(voutputs, vlabels.to(device))
                running_vloss += vloss
                _, vpredictions = voutputs.max(dim=-1)

        avg_vloss = running_vloss / len(val_dataloader)
        print('LOSS : train {} | valid {}'.format(round(avg_loss, 4), round(avg_vloss.item(), 4)))

        # Track best performance, and save the model's state
        # if avg_vloss < best_vloss:
        #     best_vloss = avg_vloss
        #     torch.save(m.state_dict(), path)

        epoch_number += 1

# %%
train(pretrained_model, pretrained_optimizer)

# %%
train(non_pretrained_model, non_pretrained_optimizer)


# %%
def predict(model, path):
    model.eval()
    predictions = []  # Inizializza una lista per memorizzare le predizioni
    with torch.no_grad():
        for _, data in enumerate(test_dataloader):
            inputs = data[0].to(device)
            # Effettua le previsioni utilizzando il modello
            outputs = model(inputs)
            # Aggiungi le predizioni alla lista delle predizioni
            predictions.append(outputs.cpu().numpy())

    predictions_array = np.concatenate(predictions)
    submission_df = pd.DataFrame(data=predictions_array, columns=["P", "K", "Mg", "pH"])
    submission_df.to_csv(path, index_label="sample_index")

# %%
predict(pretrained_model, "pretrained_model_submission.csv")
predict(non_pretrained_model, "non_pretrained_model_submission.csv")

# %% [markdown]
# 


