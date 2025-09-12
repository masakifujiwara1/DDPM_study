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
import wandb
# from wandb import Alertlevel

img_size = 32
b_size = 64
num_timeseteps = 1000
epochs = 100
lr = 1e-4
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

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
dataset = torchvision.datasets.CIFAR10(root='./data', transform=preprocess, download=True)
dataloader = DataLoader(dataset, batch_size=b_size, shuffle=True, num_workers=2)

diffuser = Diffuser(num_timesteps=num_timeseteps, device=device)
# model = UNet()
model = UNetCondDeep(in_ch=3, num_labels=10)
model = model.to(device)
optimizer = Adam(model.parameters(), lr=lr)
losses = []

config_dict = {
    "dataset": "CIFAR10",
    "model": model,
    "epochs": epochs,
    "batch_size": b_size,
    "learning_rate": lr,
    "num_timesteps": num_timeseteps,
    "optimizer": optimizer,
    "loss_function": "MSELoss",
    "dataloader": dataloader,
}
with wandb.init(project="DDPM_study", group="cifar10", name="deepU_ver", config=config_dict):

    for epoch in range(epochs):
        loss_sum = 0.0
        cnt = 0

        for imgs, labels in tqdm(dataloader):

            # x = imgs.clone()
            # imgs = [diffuser.reverse2img(x[i]) for i in range(128)]
            # show_images(imgs, labels=labels)

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
        wandb.log({"loss": loss_avg}, step=epoch)
        losses.append(loss_avg)
        print(f"Epoch {epoch}, Loss: {loss_avg}")
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'checkpoint/deepU/model_cifar10_epoch{epoch+1}.pth')

        if (epoch + 1) % 5 == 0:
            imgs_sampled, sample_labels = diffuser.sample(model)
            wandb_images = [wandb.Image(img, caption=str(lbl.item())) for img, lbl in zip(imgs_sampled, sample_labels)]
            wandb.log({"samples": wandb_images}, step=epoch)

wandb.finish()

plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

imgs, labels = diffuser.sample(model)
show_images(imgs, labels=labels)