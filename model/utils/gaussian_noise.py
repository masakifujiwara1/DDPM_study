import os
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir + '/../../imgs', 'hoge.png')
img = plt.imread(file_path)
print(img.shape)

T = 1000
beta_s = 1e-4
beta_e = 0.02
betas = torch.linspace(beta_s, beta_e, T)

preprocess = transforms.ToTensor()
x = preprocess(img)
x_hat = x.clone()
print(x.shape)

def reverse2img(x):
    x = x * 255
    x = x.clamp(0, 255)
    x = x.to(torch.uint8)
    to_pil = transforms.ToPILImage()
    return to_pil(x)

def add_noise(x_0, t, betas):
    T = len(betas)
    assert t >= 0 and t <= T

    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    t_idx = t - 1
    alpha_bar = alpha_bars[t_idx]

    eps = torch.randn_like(x_0)
    x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * eps

    return x_t

imgs = []
cnt = 0

for t in range(T):
    beta = betas[t]
    eps = torch.randn_like(x_hat)
    x_hat = torch.sqrt(1 - beta) * x_hat + torch.sqrt(beta) * eps
    cnt += 1

# print(cnt)
imgs.append(reverse2img(x_hat))

t = T
x_t = add_noise(x, t, betas)
imgs.append(reverse2img(x_t))

plt.figure(figsize=(15, 6))
for i, img in enumerate(imgs[:10]):
    plt.subplot(1, 2, i + 1)
    plt.imshow(img)
    plt.title(f'Step {i}')
    plt.axis('off')

plt.show()