import torch
from torch import nn
from tqdm import tqdm
from torchvision import transforms

import math

class Diffuser:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.num_timesteps = num_timesteps
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        # self.betas = self.cosine_schedule(lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def cosine_schedule(self, alpha_bar, max_beta=0.999):
        betas = []
        for i in range(self.num_timesteps):
            t1 = i / self.num_timesteps
            t2 = (i + 1) / self.num_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return torch.tensor(betas)


    def add_noise(self, x_0, t):
        T = self.num_timesteps
        t = t.to(torch.long)
        assert (t >= 1).all() and (t <= T).all()
        t_idx = t - 1

        alpha_bar = self.alpha_bars[t_idx]
        N = alpha_bar.size(0)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)

        noise = torch.randn_like(x_0, device=self.device)
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise
        return x_t, noise

    def denoise(self, model, x, t, labels=None):
        T = self.num_timesteps
        t = t.to(torch.long)
        assert (t >= 1).all() and (t <= T).all()

        t_idx = t - 1
        alpha = self.alphas[t_idx]
        alpha_bar = self.alpha_bars[t_idx]
        alpha_bar_prev = self.alpha_bars[t_idx-1]

        N = alpha.size(0)
        alpha = alpha.view(N, 1, 1, 1)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)
        alpha_bar_prev = alpha_bar_prev.view(N, 1, 1, 1)

        model.eval()
        with torch.no_grad():
            eps = model(x, t, labels)
        
        model.train()

        noise = torch.randn_like(x, device=self.device)
        noise[t == 1] = 0

        mu = (x - ((1-alpha) / torch.sqrt(1-alpha_bar)) * eps) / torch.sqrt(alpha)
        std = torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha))
        return mu + noise * std

    def sample(self, model, x_shape=(20, 3, 32, 32), labels=None):
        b_size = x_shape[0]
        x = torch.randn(x_shape, device=self.device)

        if labels is None:
            labels = torch.randint(0, 10, (len(x), ), device=self.device)

        for i in tqdm(range(self.num_timesteps, 0, -1)):
            t = torch.tensor([i] * b_size, device=self.device, dtype=torch.long)
            x = self.denoise(model, x, t, labels)

        imgs = [self.reverse2img(x[i]) for i in range(b_size)]
        return imgs, labels

    def reverse2img(self, x):
        x = (x * 0.5) + 0.5
        x = x * 255
        x = x.clamp(0, 255)
        x = x.to(torch.uint8)
        x = x.cpu()
        to_pil = transforms.ToPILImage()
        return to_pil(x)