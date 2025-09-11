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
from model.utils.unet import UNet, UNetCond, UNetCondDeep
from model.utils.diffuser import Diffuser

img_size = 32
b_size = 16
num_timeseteps = 1000
epochs = 100
lr = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

def show_images(imgs, rows=2, cols=10, labels=None):
    # imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()  # Convert to (N, H, W, C) and move to CPU
    fig = plt.figure(figsize=(cols, rows))
    i = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, i + 1)
            plt.imshow(imgs[i])
            plt.title(f'{labels[i].item()}' if labels is not None else '')
            plt.axis('off')
            i += 1
    plt.show()

preprocess = transforms.ToTensor()
dataset = torchvision.datasets.CIFAR10(root='./data', transform=preprocess, download=True)
dataloader = DataLoader(dataset, batch_size=b_size, shuffle=True)

diffuser = Diffuser(num_timesteps=num_timeseteps, device=device)
# model = UNet()
model = UNetCondDeep(in_ch=3, num_labels=10)
model = model.to(device)
optimizer = Adam(model.parameters(), lr=lr)
losses = []

model.load_state_dict(torch.load('checkpoint/norm_lr_bsize/model_cifar10_epoch80.pth', map_location=device))

# for epoch in range(epochs):
#     loss_sum = 0.0
#     cnt = 0

#     for imgs, labels in tqdm(dataloader):

#         # x = imgs.clone()
#         # imgs = [diffuser.reverse2img(x[i]) for i in range(128)]
#         # show_images(imgs, labels=labels)

#         optimizer.zero_grad()
#         x = imgs.to(device)
#         labels = labels.to(device)
#         t = torch.randint(1, num_timeseteps+1, (len(x), ), device=device)

#         x_noisy, noise = diffuser.add_noise(x, t)
#         noise_pred = model(x_noisy, t, labels)
#         loss = F.mse_loss(noise, noise_pred)

#         loss.backward()
#         optimizer.step()

#         loss_sum += loss.item()
#         cnt += 1

#     loss_avg = loss_sum / cnt
#     losses.append(loss_avg)
#     print(f"Epoch {epoch}, Loss: {loss_avg}")
#     if (epoch + 1) % 10 == 0:
#         torch.save(model.state_dict(), f'checkpoint/model_cifar10_epoch{epoch+1}.pth')

# plt.plot(losses)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.show()

imgs, labels = diffuser.sample(model, x_shape=(40, 3, 32, 32))
show_images(imgs, labels=labels, rows=4, cols=10)