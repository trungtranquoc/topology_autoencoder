import matplotlib.pyplot as plt
from torch import nn
from typing import List, Tuple
import os
import torch
import json

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models'))

class ModelState:
    """
    This class is used to store the state of the model.
    """

    def __init__(self, model: nn.Module, model_name: str = "Default", loss_description: List[str] = ["MSELoss"], **extra_args):
        """
        Initialize the ModelState class.
        """
        self.model = model
        self.model_name = model_name
        self.args = extra_args
        self.loss_history = []              # Type: List[Tuple[float, float] | float]
        self.loss_description = loss_description

    def reset_training(self):
        self.loss_history = []

    def loss_record(self, loss_item):
        self.loss_history.append(loss_item)

    def visualize_loss(self, title: str = "Loss vs. epoch"):
        loss_ids = list(range(1, len(self.loss_history)+1))

        loss_lsts = [[loss_item[idx] for loss_item in self.loss_history] for idx in range(len(self.loss_history[0]))]
        for loss_description, loss_lst in zip(self.loss_description, loss_lsts):
            plt.plot(loss_ids, loss_lst, label=loss_description)

        plt.xlabel("Epoch")
        plt.ylabel("loss")

        plt.title(title)
        plt.legend()

        plt.show()

    def log(self):
        print("-"*50)
        for key, value in self.__dict__.items():
            print(f"{key}: {value}")
        print("-"*50)

    def save_model(self):
        # Create directory if not exists
        model_dir = os.path.join(MODELS_DIR, self.model_name)
        os.makedirs(model_dir, exist_ok=True)

        metadata_file = os.path.join(model_dir, "metadata.json")
        torch.save(self.model.state_dict(), f"{model_dir}/model.pth")
        
        data_to_save = {k: v for k, v in self.__dict__.items() if k not in ["model"]}
        with open(metadata_file, 'w') as f:
            json.dump(data_to_save, f, indent=4) # indent for readability

        print(f"Model saved to {model_dir}")
    
    def load_model(self, model_name: str):
        model_dir = os.path.join(MODELS_DIR, model_name)
        metadata_file = os.path.join(model_dir, "metadata.json")
        
        self.model.load_state_dict(torch.load(os.path.join(model_dir, "model.pth"), map_location="cpu"))
        
        with open(metadata_file, 'r') as f:
            data = json.load(f)
            self.__dict__.update(data)

        print(f"Model loaded from {model_dir}")