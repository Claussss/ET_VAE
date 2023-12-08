import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), 
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 2 * latent_dim), 
            nn.BatchNorm1d(2 * latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.BatchNorm1d(input_dim),
        )
        
    def reparameterize(self, mu, std):
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def encode(self, x):
        h = self.encoder(x)
        mu, std = torch.chunk(h, 2, dim=1)
        return mu, std 
    
    def decode(self, z):
        x_hat = self.decoder(z)
        return x_hat

    def forward(self, x):
        mu, std = self.encode(x)
        z = self.reparameterize(mu, std)
        x_hat = self.decode(z)
        return x_hat, mu, std