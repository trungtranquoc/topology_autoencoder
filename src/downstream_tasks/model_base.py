from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass

    def eval(self, X_test, y_test, metric, **kwargs):
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
        y_pred = self.predict(X_test)
        return metric(y_test.astype(np.int32), y_pred.astype(np.int32), **kwargs)