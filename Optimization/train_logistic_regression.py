import numpy as np


def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # stablizing the inputs
    X = np.asarray(X)
    y = np.asarray(y)

    # Rows and columns
    n_samples, n_features = X.shape
    # Initializing weights and bias with zeros
    w = np.zeros(n_features)
    b = 0.0

    for _ in range(steps):
        # Linear equation
        z = X @ w + b
        # Sigmoid (Logistic regression)
        y_pred = _sigmoid(z)

        # Error calculation
        errors = y_pred - y

        # Calculating gradients
        dw = (X.T @ errors) / n_samples  # how much each feature contribute to erros
        db = (
            np.sum(errors) / n_samples
        )  # average error as bias is global for all features

        # updating weights and bias
        w -= lr * dw
        b -= lr * db
    return (w, b)
