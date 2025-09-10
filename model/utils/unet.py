import torch
from torch import nn
from model.utils.pos_encoding import pos_encoding

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.mlp = nn.Sequential(
            nn.Linear(time_emb_dim, in_ch),
            nn.ReLU(),
            nn.Linear(in_ch, in_ch)
        )

    def forward(self, x, v):
        N , C, _, _ = x.shape
        v = self.mlp(v)
        v = v.view(N, C, 1, 1)
        y = self.convs(x + v)
        return y

class UNet(nn.Module):
    def __init__(self, in_ch=1, time_emb_dim = 100):
        super().__init__()
        self.time_emb_dim = time_emb_dim

        self.down1 = ConvBlock(in_ch, 64, time_emb_dim)
        self.down2 = ConvBlock(64, 128, time_emb_dim)
        self.bot1 = ConvBlock(128, 256, time_emb_dim)
        self.up2 = ConvBlock(128 + 256, 128, time_emb_dim)
        self.up1 = ConvBlock(128 + 64, 64, time_emb_dim)
        self.out = nn.Conv2d(64, in_ch, 1)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor = 2, mode = 'bilinear')

    def forward(self, x, timesteps):
        v = pos_encoding(timesteps, self.time_emb_dim, device=x.device)

        x1 = self.down1(x, v)
        x = self.maxpool(x1)
        x2 = self.down2(x, v)
        x = self.maxpool(x2)

        x = self.bot1(x, v)
        
        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x, v)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up1(x, v)
        x = self.out(x)
        return x

class UNetCond(nn.Module):
    def __init__(self, in_ch=1, time_emb_dim=100, num_labels=None):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        
        self.down1 = ConvBlock(in_ch, 64, time_emb_dim)
        self.down2 = ConvBlock(64, 128, time_emb_dim)
        self.bot1 = ConvBlock(128, 256, time_emb_dim)
        self.up2 = ConvBlock(128 + 256, 128, time_emb_dim)
        self.up1 = ConvBlock(128 + 64, 64, time_emb_dim)
        self.out = nn.Conv2d(64, in_ch, 1)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor = 2, mode = 'bilinear')

        if num_labels is not None:
            self.label_emb = nn.Embedding(num_labels, time_emb_dim)

    def forward(self, x, timesteps, labels):
        v = pos_encoding(timesteps, self.time_emb_dim, device=x.device)

        if labels is not None:
            v += self.label_emb(labels)

        x1 = self.down1(x, v)
        x = self.maxpool(x1)
        x2 = self.down2(x, v)
        x = self.maxpool(x2)

        x = self.bot1(x, v)
        
        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x, v)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up1(x, v)
        x = self.out(x)
        return x

if __name__ == "__main__":
    model = UNet()
    x = torch.randn(10, 1, 28, 28)
    y = model(x)
    print(y.shape)  # Should output: torch.Size([10, 1, 28, 28])