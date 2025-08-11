from .model_base import BaseModel
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch
import numpy as np

from ..data_processing import ClassificationDataset

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, activation=nn.ReLU):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [64, 32]  # Default hidden layers
        layers = []
        in_features = input_size
        for out_features in hidden_layers:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            in_features = out_features
        layers.append(nn.Linear(in_features, output_size))
        layers.append(nn.Softmax(dim=1))            # Softmax for classification
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class NeuralNetworkModel(BaseModel):
    def __init__(self, input_size, output_size, hidden_layers=[32,32], device="cpu", seed=2025):
        self.model = NeuralNetwork(input_size, output_size, hidden_layers)
        self.device = device
        torch.manual_seed(seed=seed)

    def fit(self, X_train, y_train, num_epochs=10, lr=1e-2, batch_size=64):
        """
        Trains the neural network model on the provided dataset.

        Args:
            X_train (np.ndarray): Input feature matrix of shape (n_samples, n_features).
            y_train (np.ndarray): Target labels of shape (n_samples,).
            num_epochs (int): Number of training epochs.
            lr (float): Learning rate for the optimizer.
            batch_size (int): Size of each training batch.
        """
        train_loader = ClassificationDataset(X_train, y_train).preprare_batch(batch_size=batch_size)
        
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        progress_bar = tqdm(range(num_epochs), desc="Training Neural Network")
        for _ in progress_bar:
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch.float())
                loss = loss_fn(outputs, y_batch.long())
                loss.backward()
                optimizer.step()
            
            progress_bar.set_postfix({"Loss": loss.item() / len(train_loader)})

    def predict(self, X_test):
        """
        Generates class predictions for the input data.

        Args:
            X_test (np.ndarray): Input feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels of shape (n_samples,).
        """
        test_loader = ClassificationDataset(X_test).preprare_batch(batch_size=64)

        output_list = []
        self.model.eval()
        with torch.no_grad():
            for X_batch in test_loader:
                outputs = self.model(X_batch.float())
                preds = torch.argmax(outputs, dim=1)
                output_list.append(preds.detach().cpu().numpy())

        return np.concatenate(output_list, axis=0)