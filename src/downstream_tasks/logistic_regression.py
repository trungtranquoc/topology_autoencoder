from .model_base import BaseModel
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from ..data_processing import ClassificationDataset

class LogisticReg(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))

class LogisticRegModel(BaseModel):
    def __init__(self, input_size, output_size, device="cpu", seed=2025):
        super().__init__()
        self.device = device
        torch.manual_seed(seed=seed)
        self.model = LogisticReg(input_size, output_size)
        self.model.to(device)
    
    def fit(self, X, y, num_epochs=10, lr=1e-2):
        """
        Trains the model on the provided dataset.

        Args:
            X (np.ndarray): Input feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Target labels of shape (n_samples,).
            device (str, optional): Device to train on, e.g., "cpu" or "cuda". Defaults to "cpu".
            lr (float, optional): Learning rate for the optimizer. Defaults to 1e-2.

        Returns:
            list[float]: A list of training loss values for each epoch.
        """
        train_loader = ClassificationDataset(X, y).preprare_batch(batch_size=64)
    
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        log_loss = []
        progress_bar = tqdm(range(num_epochs), desc="Training Logistic Regression")
        for _ in progress_bar:
            self.model.train()
            running_loss = 0.0
            for X_batch, y_batch in train_loader:
                images, labels = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                # outputs = outputs.squeeze(1)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
        
            log_loss.append(running_loss)
            progress_bar.set_postfix({"Loss": running_loss / len(train_loader)})

        return log_loss
    
    def predict(self, X):
        """
        Generates class predictions for the input data.

        Args:
            X (np.ndarray or torch.Tensor): Input feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels of shape (n_samples,).
        """
        dataset = ClassificationDataset(X)
        test_loader = dataset.preprare_batch(batch_size=64)

        output_list = []
        self.model.eval()
        for X_batch in test_loader:
            with torch.no_grad():
                images = X_batch.to(self.device)    # .unsqueeze(1)
                outputs = self.model(images)            # shape: 

                # outputs = outputs.squeeze(1)
                
                probs = torch.softmax(outputs, dim=1)  # shape: [batch_size, 10]
                preds = torch.argmax(probs, dim=1)     # predicted classes
                
                output_list.append(preds.detach().cpu().numpy())
    
        return np.concatenate(output_list, axis=0)