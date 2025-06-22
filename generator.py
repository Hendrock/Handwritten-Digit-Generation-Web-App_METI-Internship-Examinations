import torch
import torch.nn as nn

class CVAEDecoder(nn.Module):
    def __init__(self, latent_dim=20, n_classes=10, img_shape=(1, 28, 28)):
        super().__init__()
        self.label_emb = nn.Embedding(n_classes, 10)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 10, 400),
            nn.ReLU(),
            nn.Linear(400, 28 * 28),
            nn.Sigmoid()
        )
        self.img_shape = img_shape

    def forward(self, z, labels):
        label = self.label_emb(labels)
        z = torch.cat([z, label], dim=1)
        out = self.decoder(z)
        out = out.view(-1, *self.img_shape)
        return out
