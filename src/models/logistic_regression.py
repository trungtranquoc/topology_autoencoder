from .model_base import BaseModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm

class LogisticReg(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))
    
class ClassificationDataset(Dataset):
    def __init__(self, X, y = None):
        self.X = X.astype(np.float32)  # convert once, but stay in numpy
        if y is not None:
            self.y = y.astype(np.int64)
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])  # convert per sample on-the-fly
        else:
            return torch.from_numpy(self.X[idx])

class LogisticRegModel(BaseModel):
    def __init__(self, input_size, output_size, device="cpu", seed=2025):
        super().__init__()
        self.device = device
        torch.manual_seed(seed=seed)
        self.model = LogisticReg(input_size, output_size)
        self.model.to(device)

    def prepare_batch(self, X, y=None, batch_size=64):
        return DataLoader(ClassificationDataset(X, y), batch_size=batch_size)
    
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
        train_loader = self.prepare_batch(X, y)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        log_loss = []
        
        for _ in tqdm(range(num_epochs), desc="Training"):
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

        return log_loss
    
    def predict(self, X):
        """
        Generates class predictions for the input data.

        Args:
            X (np.ndarray or torch.Tensor): Input feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels of shape (n_samples,).
        """
        test_loader = self.prepare_batch(X)

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

    def eval(self, y_true, y_preds, metric, **kwargs):
        """
        Evaluates prediction results using a specified metric function with optional keyword arguments.

        Args:
            y_true (np.ndarray): Ground truth labels, shape (n_samples,).
            y_preds (np.ndarray): Predicted labels, shape (n_samples,).
            metric (Callable): A scoring function that takes (y_true, y_preds) and optional kwargs, e.g.,
                `sklearn.metrics.accuracy_score`, `f1_score`, etc.
            **kwargs: Additional keyword arguments to pass to the metric function (e.g., average='micro').

        Returns:
            float: Evaluation score computed by the given metric.
        """

        return metric(y_true.astype(np.int32), y_preds.astype(np.int32), **kwargs)