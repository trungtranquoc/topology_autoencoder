import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc_1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc_2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, X):
        X = self.relu(self.fc_1(X))
        X = self.relu(self.fc_2(X))
        return X
    
class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc_1 = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.fc_2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, X):
        X = self.relu(self.fc_1(X))
        X = self.sigmoid(self.fc_2(X))
        return X
    
class FNNAutoEncoder(nn.Module):
    def __init__(self, input_size, latent_size, hidden_size=10):
        super().__init__()
        self.encoder = Encoder(input_size, hidden_size, latent_size)
        self.decoder = Decoder(latent_size, hidden_size, input_size)
    
    def forward(self, X):
        return self.decoder(self.encoder(X))
    
    def encode(self, X):
        return self.encoder(X)
    
    def decode(self, X):
        return self.decoder(X)    