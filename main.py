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

    def softmax(self, x: np.ndarray, epsilon=1e-12) -> np.ndarray:
        """
        Softmax activation function.

        Returns:
        ndarray: Output after applying the softmax function.
        """
        # subtract the maximum value in the input array x
        # before taking the exponential.
        # This prevents overflow by ensuring
        # the argument of np.exp is always non-positive
        exp_x = np.exp(x - np.max(x))
        # but underflow can still occur
        # if x is a large negative number,
        # resulting in exp_x becoming zero.

        # One potential issue is that
        # if all values in a row of x are very large
        # negative numbers,
        # exp_x will be an array of zeros,
        # and summing these will result in
        # zero - leading to a division by zero and hence
        # a NaN in the output.

        sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
        # return exp_x / exp_x.sum(axis=1, keepdims=True)
        return exp_x / np.maximum(
            sum_exp_x, epsilon
        )  # Use maximum to prevent division by zero

    def mse_loss(
        self, y_true: np.ndarray, y_pred: np.ndarray  # true labels  # predicted labels
    ) -> float:
        """
        Calculates the Mean Squared Error loss.
        prediction error between the probability vector ytrue and the predicted vector y_pred

        Returns:
        float: Computed MSE loss.
        """
        # return (np.subtract(y_true,y_pred) ** 2).mean()
        return np.divide(np.sum((np.subtract(y_true, y_pred) ** 2)), y_true.shape[0])

    def get_output_error(
        self,
        #  y_one_hot : np.ndarray # one-hot encoded labels,
        #  model_output : np.ndarray # predicted labels
        y_true: np.ndarray,  # true labels
        y_pred: np.ndarray,  # predicted labels
    ) -> np.ndarray:
        """
        Calculates the output error.

        Returns:
        ndarray: Output error.
        """
        # Output layer error is the difference between predicted and true values
        # y_one_hot.shape[0] is the number of rows in y_one_hot
        # output_error = np.substract(y_one_hot, self.model_output) / y_one_hot.shape[0]
        output_error = np.divide(np.subtract(y_true, y_pred), y_true.shape[0])
        return output_error

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
    ) -> float:
        """
        Performs the backward pass
        (backpropagation) and
        updates the weights.
        Returns:
        float: Computed loss.
        """
        # Calculate the Mean Squared Error loss

        loss = self.mse_loss(y_one_hot, self.model_output)
        print("loss : ", loss)

        # Rétropropagation
        output_error = self.get_output_error(y_one_hot, self.model_output)

        # Calculate hidden layer error (backpropagated error)
        # hidden_output_ =  1 - self.hidden_output**2
        # L’erreur de la couche intermédiaire est donnée par 𝑒ℎ = (𝑒_𝑜 × 𝑊_𝑜^𝑇 ) ∗ 𝑦ℎ ∗ (1 − 𝑦ℎ)
        # weights_hidden_output_transpose = self.weights_hidden_output.T
        weights_hidden_output_transpose = np.transpose(self.weights_hidden_output)

        hidden_error = (
            np.dot(output_error, weights_hidden_output_transpose)
            * self.hidden_output
            * (1 - self.hidden_output)
        )
        # gradient is the derivative of the loss function (MSE) and serves to update the weights

        # Calculating gradient for weights between input and hidden layer
        x_transpose = np.transpose(X)
        d_weights_input_hidden = np.dot(x_transpose, hidden_error)
        # Calculate gradient for weights between hidden and output layer
        # d_weights_hidden_output = np.dot(self.model_output,output_error)
        d_weights_hidden_output = np.dot(self.hidden_output.T,output_error)

        # Update the weights with the derivatives (gradient descent)
        # 𝑊ℎ = 𝑊ℎ − 𝜇(𝑥𝑇 × 𝑒ℎ)
        self.weights_input_hidden -= learning_rate * d_weights_input_hidden
        # 𝑊𝑜 = 𝑊𝑜 − 𝜇(𝑦𝑜 × 𝑒𝑜)
        self.weights_hidden_output -= learning_rate * d_weights_hidden_output

        return loss

    def train(
        self,
        X: np.ndarray,  # input data
        y_one_hot: np.ndarray,  # one-hot encoded labels
        epochs=100,  # number of training epochs
        learning_rate=0.01,  # learning rate
    ):
        """
        Trains the neural network.
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

    def visualize_prediction(
        self,
        X: np.ndarray,  # input data
        y_true: np.ndarray,  # one-hot encoded labels
        index: int,  # index of the data point to visualize
    ):
        """
        Visualizes the prediction for a single data point.
        """
        input_data = X[index, :].reshape(28, 28)  # 28x28 image

        predicted_label = self.predict(X[index : index + 1])[0]

        plt.imshow(input_data, cmap="gray")
        plt.title(
            f"Prediction: {predicted_label}, True Label: {np.argmax(y_true[index])}"
        )
        plt.show()

    def visualize_predictions(self, X, y_true, num_predictions=10):
        """
        Visualizes the predictions for the first
        num_predictions data points
        into one plot.

        Parameters:
        X (ndarray): Data for prediction.
        y_true (ndarray): True labels.
        num_predictions (int): Number of predictions to visualize.
        """
        predictions = self.predict(X[:num_predictions])
        images = X[:num_predictions, :].reshape(num_predictions, 28, 28)
        true_labels = np.argmax(y_true[:num_predictions], axis=1)

        plt.figure(figsize=(12, 12))

        for i in range(num_predictions):
            plt.subplot(5, 2, i + 1)
            plt.imshow(images[i], cmap="gray")
            plt.title(f"Prediction: {predictions[i]}, True Label: {true_labels[i]}")
            plt.axis("off")

        plt.tight_layout()
        plt.show()

    def confusion_matrix(
        self,
        X: np.ndarray,  # input data
        y_true: np.ndarray,  # one-hot encoded true labels
    ):
        # Predict the labels for the given input data using the model
        y_pred = self.predict(X)

        # Number of classes in the dataset, here assumed to be 10
        num_classes = 10

        # Initialize a confusion matrix with zeros, of size num_classes x num_classes
        cm = np.zeros((num_classes, num_classes), dtype=int)

        # Populate the confusion matrix by comparing actual and predicted labels
        for true_label, pred_label in zip(np.argmax(y_true, axis=1), y_pred):
            cm[true_label, pred_label] += 1

        # Create a plot to visualize the confusion matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()

        # Set the tick marks for the x and y axes
        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, range(num_classes))
        plt.yticks(tick_marks, range(num_classes))

        # Label the axes with appropriate names
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")

        # Annotate each cell of the matrix with the count of occurrences
        for i in range(num_classes):
            for j in range(num_classes):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

        # Display the plot
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
    # nn.visualize_prediction(X_test, y_test, 10)
    nn.visualize_predictions(X_test, y_test, 10)
