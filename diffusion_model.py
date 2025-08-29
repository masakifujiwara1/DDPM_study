import math
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from model.utils.unet import UNet, UNetCond
from model.utils.diffuser import Diffuser

img_size = 28
b_size = 128
num_timeseteps = 1000
epochs = 10
lr = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

def show_images(imgs, rows=2, cols=10, labels=None):
    fig = plt.figure(figsize=(cols, rows))
    i = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, i + 1)
            plt.imshow(imgs[i], cmap='gray')
            plt.title(f' {labels[i].item()}' if labels is not None else '')
            plt.axis('off')
            i += 1
    plt.show()

preprocess = transforms.ToTensor()
dataset = torchvision.datasets.MNIST(root='./data', transform=preprocess, download=True)
dataloader = DataLoader(dataset, batch_size=b_size, shuffle=True)

diffuser = Diffuser(num_timesteps=num_timeseteps, device=device)
# model = UNet()
model = UNetCond(num_labels=10)
model = model.to(device)
optimizer = Adam(model.parameters(), lr=lr)
losses = []

for epoch in range(epochs):
    loss_sum = 0.0
    cnt = 0

    for imgs, labels in tqdm(dataloader):
        optimizer.zero_grad()
        x = imgs.to(device)
        labels = labels.to(device)
        t = torch.randint(1, num_timeseteps+1, (len(x), ), device=device)

        x_noisy, noise = diffuser.add_noise(x, t)
        noise_pred = model(x_noisy, t, labels)
        loss = F.mse_loss(noise, noise_pred)

        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        cnt += 1

    loss_avg = loss_sum / cnt
    losses.append(loss_avg)
    print(f"Epoch {epoch}, Loss: {loss_avg}")

plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

imgs, labels = diffuser.sample(model)
show_images(imgs, labels=labels)