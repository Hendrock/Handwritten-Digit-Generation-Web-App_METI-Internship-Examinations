import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, n_classes=10, img_shape=(1,28,28)):
        super().__init__()
        self.label_emb = nn.Embedding(n_classes, latent_dim)
        self.init_size = img_shape[1] // 4  # e.g., 28/4 = 7
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, img_shape[0], 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)
        gen_input = torch.mul(z, c)
        out = self.l1(gen_input)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img
