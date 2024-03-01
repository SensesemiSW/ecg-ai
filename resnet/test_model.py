# test_model.py
from data_utils import load_and_preprocess_data
from train_model import train_model

def test_model(trained_model, X_test, Y_test):
    results = trained_model.evaluate(X_test, Y_test)
    print("Loss on Unseen Data:", results[0])
    print("Accuracy on Unseen Data:", results[1])

# Example usage:
if __name__ == "__main__":
    X_train, X_test, Y_train, Y_test, label_encoder = load_and_preprocess_data()
    trained_model = train_model(X_train, Y_train, input_shape=(1000, 12), num_classes=5)
    test_model(trained_model, X_test, Y_test)
