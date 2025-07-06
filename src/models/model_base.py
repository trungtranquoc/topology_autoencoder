from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self):
        pass