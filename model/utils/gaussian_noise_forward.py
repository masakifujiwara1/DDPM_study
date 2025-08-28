import os
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

T = 1000
betas = torch.linspace(1e-4, 0.02, T)

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir + '/../../imgs', 'hoge.png')
image = plt.imread(file_path)
print(image.shape)

preprocess = transforms.ToTensor()
x = preprocess(image)
print(x.shape)

def reverse2img(x):
    x = x * 255
    x = x.clamp(0, 255)
    x = x.to(torch.uint8)
    to_pil = transforms.ToPILImage()
    return to_pil(x)

imgs = []

for t in range(T):
    if t % 100 == 0:
        img = reverse2img(x)
        imgs.append(img)
    
    beta = betas[t]
    eps = torch.randn_like(x)
    x = torch.sqrt(1 - beta) * x + torch.sqrt(beta) * eps

imgs.append(reverse2img(x))

plt.figure(figsize=(15, 6))
for i, img in enumerate(imgs[:11]):
    plt.subplot(2, 6, i + 1)
    plt.imshow(img)
    plt.title(f'Step {i * 100}')
    plt.axis('off')

plt.show()