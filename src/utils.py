import json
import pickle

def save_model(model, file_path="model.pkl"):
    """
    Save a trained model to a file.
    """
    with open(file_path, "wb") as f:
        pickle.dump(model, f)

def load_model(file_path="model.pkl"):
    """
    Load a model from a file.
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)

def save_metrics(metrics, file_path="metrics.json"):
    """
    Save evaluation metrics to a JSON file.
    """
    with open(file_path, "w") as f:
        json.dump(metrics, f)

def load_metrics(file_path="metrics.json"):
    """
    Load evaluation metrics from a JSON file.
    """
    with open(file_path, "r") as f:
        return json.load(f)

