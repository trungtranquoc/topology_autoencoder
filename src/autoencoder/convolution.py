import torch.nn as nn

class CNNAutoencoder(nn.Module):
    def __init__(self):
        # Input size: N x 1 x 28 x 28
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),   # N x 16 x 14 x 14
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # N x 32 x 7 x 7
            nn.ReLU(),
            nn.Conv2d(32, 64, 7),   # N x 64 x 1 x 1
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),  # N x 32 x 7 x 7
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), # N x 16 x 14 x 14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  # N x 1 x 28 x 28
            nn.Sigmoid(),
        )

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded