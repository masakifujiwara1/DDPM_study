import math
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from model.unet import UNet, UNetCond, UNetCondDeep
from model.utils.diffuser import Diffuser
from model.utils.decorrelation import DecorrelationLoss
import wandb
import copy
# from wandb import Alertlevel

img_size = 32
b_size = 512
num_timeseteps = 1000
epochs = 100
lr = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

is_deccorrelation = True  # decorrelation lossを使うかどうか
deccorrelation_beta = 0.02

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
    fig.tight_layout()
    return fig

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
dataset = torchvision.datasets.CIFAR10(root='./data', transform=preprocess, download=True)
dataloader = DataLoader(dataset, batch_size=b_size, shuffle=True, num_workers=4, pin_memory=True)

diffuser = Diffuser(num_timesteps=num_timeseteps, device=device)
# model = UNet()
model = UNetCondDeep(in_ch=3, num_labels=10)
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(dataloader), epochs=epochs)
calc_loss_reg = DecorrelationLoss().to(device)
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
    "decorrelation": is_deccorrelation,
    "deccorrelation_beta": deccorrelation_beta,
}

# ema initialization
ema_decay = 0.995
ema_model = copy.deepcopy(model).eval()
for p in ema_model.parameters():
    p.requires_grad_(False)

with wandb.init(project="DDPM_study", group="cifar10", name="deepU3-adamw-noema-cos_ver", config=config_dict):

    for epoch in range(epochs):
        loss_sum = 0.0
        S_top100_sum = None
        cnt = 0

        for imgs, labels in tqdm(dataloader):

            # x = imgs.clone()
            # imgs = [diffuser.reverse2img(x[i]) for i in range(128)]
            # show_images(imgs, labels=labels)

            optimizer.zero_grad()
            x = imgs.to(device)
            labels = labels.to(device)
            t = torch.randint(1, num_timeseteps, (len(x), ), device=device)
            # print(f"t min: {t.min()}, t max: {t.max()}")
            x_noisy, noise = diffuser.add_noise(x, t)
            noise_pred, feature = model(x_noisy, t, labels)
            loss = F.mse_loss(noise, noise_pred)

            # loss_regと相関行列の計算
            loss_reg, corr = calc_loss_reg(feature)
            if is_deccorrelation:
                # decorrelation lossを加算
                loss = loss + deccorrelation_beta * loss_reg

            # 特異値分解で相関行列の確認
            U, S, V = torch.svd(corr)
            # print(f"Singular values of correlation matrix: {S}")
            topk_vals, topk_idx = torch.topk(S, k=100, largest=True, sorted=True)
            S_top100 = topk_vals

            loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                msd = model.state_dict()
                for k, ema_v in ema_model.state_dict().items():
                    model_v = msd[k].detach()
                    ema_v.copy_(ema_v * ema_decay + (1.0 - ema_decay) * model_v)

            loss_sum += loss.item()
            cnt += 1

            if S_top100_sum is None:
                S_top100_sum = S_top100
            else:
                S_top100_sum += S_top100

        loss_avg = loss_sum / cnt
        S_top100_avg = S_top100_sum / cnt

        # 上位100特異値の折れ線グラフを作成し、wandbに保存
        try:
            s_vals = S_top100_avg.detach().cpu().numpy()
            fig_s = plt.figure(figsize=(8, 4))
            plt.plot(range(1, len(s_vals) + 1), s_vals, marker='o', linewidth=1)
            plt.title(f'Top-100 singular values (epoch {epoch+1})')
            plt.xlabel('rank (1-100)')
            plt.ylabel('singular value')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            wandb.log({"top100_singular_values": wandb.Image(fig_s, caption=f"Epoch {epoch+1} S top-100")}, step=epoch)
            plt.close(fig_s)
        except Exception as e:
            # 例外が出ても学習は止めない
            print(f"[warn] plotting top-100 S failed: {e}")

        wandb.log({"loss": loss_avg, "learning_rate": scheduler.get_last_lr()[0]}, step=epoch)
        losses.append(loss_avg)
        print(f"Epoch {epoch}, Loss: {loss_avg}")
        # if (epoch + 1) % 10 == 0:
        #     torch.save(ema_model.state_dict(), f'checkpoint/deepU-mlp-ema-scheduler/model_cifar10_epoch{epoch+1}.pth')

        if (epoch + 1) % 5 == 0:
            # 0~9を2回繰り返したラベルを作成
            sample_labels = torch.tensor([i for i in range(10)] * 2, device=device)
            imgs_sampled, sample_labels = diffuser.sample(model, labels=sample_labels)
            # 画像グリッドを作成し、wandbに登録
            fig = show_images(imgs_sampled, labels=sample_labels)
            wandb.log({"samples_grid": wandb.Image(fig, caption=f"Epoch {epoch+1} grid")}, step=epoch+1)
            plt.close(fig)
            # 個別画像も登録したい場合は下記を有効化
            # wandb_images = [wandb.Image(img, caption=str(lbl.item())) for img, lbl in zip(imgs_sampled, sample_labels)]
            # wandb.log({"samples": wandb_images}, step=epoch)

wandb.finish()

# plt.plot(losses)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.show()

# imgs, labels = diffuser.sample(model)
# fig = show_images(imgs, labels=labels)
# plt.show()