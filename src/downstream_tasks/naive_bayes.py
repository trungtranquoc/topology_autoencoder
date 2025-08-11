from .model_base import BaseModel
from sklearn.naive_bayes import GaussianNB

class NaiveBayesModel(BaseModel):
    def __init__(self):
        self.model = GaussianNB()

    """
    A simple Naive Bayes classifier.
    """
    def fit(self, X_train, y_train):
        """
        Fit the Naive Bayes model to the training data.
        
        Args:
            X_train (np.ndarray): Training feature matrix.
            y_train (np.ndarray): Training labels.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Predict labels for the test data.
        
        Args:
            X_test (np.ndarray): Test feature matrix.
        
        Returns:
            np.ndarray: Predicted labels.
        """
        return self.model.predict(X_test)