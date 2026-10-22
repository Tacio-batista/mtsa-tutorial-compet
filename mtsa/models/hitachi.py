import torch
import torch.nn as nn


class Hitachi(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.input_dim),
        )

    def forward(self, X):
        X_encoded = self.encoder(X)
        X_decoded = self.decoder(X_encoded)
        return X_decoded


class WindowedHitachi(nn.Module):

    def __init__(self, window_size, n_features):
        super().__init__()
        self.autoencoder = Hitachi(input_dim=window_size * n_features)

    def forward(self, X):
        shape = X.shape
        X_flat = X.float().reshape(shape[0], -1)
        return self.autoencoder(X_flat).reshape(shape)


class ReconstructionError(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, X):
        X = X.float()
        return torch.mean((X - self.model(X)) ** 2)
