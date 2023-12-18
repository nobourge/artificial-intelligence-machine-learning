import numpy as np
import matplotlib.pyplot as plt


def load_mnist_data(file_path):
    """
    Loads the MNIST dataset
    from a given file path.

    Parameters:
    file_path (str): Path to the dataset file.

    Returns:
    tuple: Tuple containing
    the normalized images and
    their one-hot encoded labels.
    one-hot encoding: https://en.wikipedia.org/wiki/One-hot
    """
    data = np.loadtxt(file_path, delimiter=",", skiprows=1)

    # ...
    # Extract labels (first column)
    labels = data[:, 0].astype(int)

    # Extract images (all columns except the first)
    images = data[:, 1:]

    # Normalize images by dividing by 255
    images = images / 255.0

    # Number of classes (digits 0-9)
    num_classes = 10
    # One-hot encode the labels
    labels_one_hot = np.eye(num_classes)[labels]
    # Qu'est-ce qu'un encodage one-hot ?
    # https://fr.wikipedia.org/wiki/One-hot:
    # "En apprentissage automatique,
    # un encodage one-hot est un vecteur
    # qui contient des valeurs binaires.

    return images, labels_one_hot


class NeuralNetwork:
    """neural network with 1 hidden layer"""

    def __init__(
        self,
        input_size: int,  # input neurons quantity
        hidden_size: int,  # hidden neurons quantity
        output_size: int,  # output neurons quantity
    ):
        """initializes the weights
        of the neural network
        with a normal distribution
        """

        # input hidden is the output of the input layer
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        # hidden output is the output of the hidden layer
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)

    def tanh(self, x):
        """tanh is the
        activation function
        Parameters:
        x (ndarray): Input array.

        Returns:
        ndarray: Output after applying the tanh function.
        """
        return np.tanh(x)

    def softmax(self, x):
        """
        Softmax activation function.

        Parameters:
        x (ndarray): Input array.

        Returns:
        ndarray: Output after applying the softmax function.
        """
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    def mse_loss(self, y_true, y_pred):
        """
        Calculates the Mean Squared Error loss.

        Parameters:
        y_true (ndarray): True labels.
        y_pred (ndarray): Predicted labels.

        Returns:
        float: Computed MSE loss.
        """
        return ((y_true - y_pred) ** 2).mean()

    def forward(self, X: np.ndarray) -> np.ndarray:  # input data
        """
        Performs the forward pass
        of the neural network.
        """
        # Input to Hidden Layer
        # Weighted sum of inputs
        # np.dot() is the matrix multiplication between X and self.weights_input_hidden
        self.hidden_input = np.dot(X, self.weights_input_hidden)
        # Apply tanh activation function
        self.hidden_output = self.tanh(self.hidden_input)

        # Hidden Layer to Output Layer
        # Weighted sum of hidden outputs
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output)
        # Apply softmax activation function
        self.model_output = self.softmax(self.output_input)
        # print("self.model_output", self.model_output)
        # print("self.model_output.shape", self.model_output.shape)

        return self.model_output

    def backward(
        self,
        X: np.ndarray,  # input data
        y_one_hot: np.ndarray,  # one-hot encoded labels
        learning_rate=0.01,
    ):
        """
        Performs the backward pass
        (backpropagation) and
        updates the weights.
        Returns:
        float: Computed loss.
        """
        # # Calcul de la MSE
        # # ...
        # loss = None

        # # Rétropropagation
        # # ...
        # output_error = None
        # hidden_error = None

        # # Mise à jour des poids
        # # ...
        # self.weights_hidden_output = None
        # self.weights_input_hidden = None

        # return loss
        # Calculate the Mean Squared Error loss

        loss = self.mse_loss(y_one_hot, self.model_output)

        # Rétropropagation
        # Output layer error is the difference between predicted and true values
        output_error = y_one_hot - self.model_output

        # gradient is the derivative of the loss function (MSE) and serves to update the weights
        # Calculate gradient for weights between hidden and output layer
        d_weights_hidden_output = self.hidden_output.T.dot(output_error)

        # Calculate hidden layer error (backpropagated error)
        hidden_error = output_error.dot(self.weights_hidden_output.T) * (
            1 - self.hidden_output**2
        )

        # Calculating gradient for weights between input and hidden layer
        d_weights_input_hidden = X.T.dot(hidden_error)

        # Update the weights with the derivatives (gradient descent)
        self.weights_hidden_output += learning_rate * d_weights_hidden_output
        self.weights_input_hidden += learning_rate * d_weights_input_hidden

        return loss

    def train(self, X, y_one_hot, epochs=100, learning_rate=0.01):
        """
        Trains the neural network.

        Parameters:
        X (ndarray): Training data.
        y_one_hot (ndarray): One-hot encoded labels.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate.
        """
        for epoch in range(epochs):
            self.forward(X)
            loss = self.backward(X, y_one_hot, learning_rate)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        """
        Predicts labels for given data.

        Parameters:
        X (ndarray): Data for prediction.

        Returns:
        ndarray: Predicted labels.
        """
        self.forward(X)
        return np.argmax(self.model_output, axis=1)

    def visualize_prediction(self, X, y_true, index):
        """
        Visualizes the prediction for a single data point.

        Parameters:
        X (ndarray): Data for prediction.
        y_true (ndarray): True labels.
        index (int): Index of the data point to visualize.
        """
        input_data = X[index, :].reshape(28, 28)

        predicted_label = self.predict(X[index : index + 1])[0]

        plt.imshow(input_data, cmap="gray")
        plt.title(
            f"Prediction: {predicted_label}, True Label: {np.argmax(y_true[index])}"
        )
        plt.show()

    def confusion_matrix(self, X, y_true):
        y_pred = self.predict(X)

        num_classes = 10
        cm = np.zeros((num_classes, num_classes), dtype=int)

        for true_label, pred_label in zip(np.argmax(y_true, axis=1), y_pred):
            cm[true_label, pred_label] += 1

        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, range(num_classes))
        plt.yticks(tick_marks, range(num_classes))
        plt.xlabel("Predicted")
        plt.ylabel("True")

        for i in range(num_classes):
            for j in range(num_classes):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

        plt.show()


if __name__ == "__main__":
    X, y = load_mnist_data("mnist_train.csv/mnist_train.csv")

    input_size = X.shape[1]  # shape[0] = number of rows, shape[1] = number of columns
    hidden_size = 128
    output_size = 10
    e = 100
    mu = 0.01

    nn = NeuralNetwork(input_size, hidden_size, output_size)
    nn.train(X, y, epochs=e, learning_rate=mu)

    X_test, y_test = load_mnist_data("mnist_test.csv/mnist_test.csv")
    nn.confusion_matrix(X_test, y_test)
    nn.visualize_prediction(X_test, y_test, 10)
