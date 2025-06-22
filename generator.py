# generator.py
import torch
import torch.nn as nn

class CVAE(nn.Module):
    def __init__(self, latent_dim=20, n_classes=10, img_shape=(1, 28, 28)):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.n_classes = n_classes

        self.label_emb = nn.Embedding(n_classes, 10)

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28 + 10, 400),
            nn.ReLU()
        )

        self.mu_layer = nn.Linear(400, latent_dim)
        self.logvar_layer = nn.Linear(400, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 10, 400),
            nn.ReLU(),
            nn.Linear(400, 28 * 28),
            nn.Sigmoid()
        )

    def encode(self, x, labels):
        label = self.label_emb(labels)
        x = torch.cat([x, label], dim=1)
        h = self.encoder(x)
        return self.mu_layer(h), self.logvar_layer(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, labels):
        label = self.label_emb(labels)
        z = torch.cat([z, label], dim=1)
        out = self.decoder(z)
        return out.view(-1, *self.img_shape)

    def forward(self, x, labels):
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, labels), mu, logvar
