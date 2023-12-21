import numpy as np
import matplotlib.pyplot as plt


def sort_csv_file(file_path, sort_by_label=False):
    """
    Sorts a CSV file by a given criterion.

    """
    print("sort_csv_file")
    print("sort_by_label : ", sort_by_label)
    print("file_path : ", file_path)
    # Step 1: Read the CSV file
    data = np.loadtxt(file_path, delimiter=",", skiprows=1, dtype=int)
    # s

    # Step 2: Sort the data
    if sort_by_label:
        # Assuming the label is in the first column
        sorted_data = data[data[:, 0].argsort()]
    else:
        # If not sorting by label, you can choose another sorting criterion
        # For example, sort by the second column
        sorted_data = data[data[:, 1].argsort()]

    # Step 3: Save the sorted data back to a CSV file
    sorted_file_path = file_path.rsplit(".", 1)[0] + "_sorted.csv"
    np.savetxt(sorted_file_path, sorted_data, delimiter=",", fmt="%d")

    print("sorted_file_path : ", sorted_file_path)
    return sorted_file_path  # Return the path of the sorted file for reference


def sort_by_labels(images, labels):
    """
    Orders the images and labels by the labels.

    Parameters:
    images (ndarray): Array of images.
    labels (ndarray): Array of labels.

    Returns:
    tuple: Tuple containing the sorted images and their corresponding labels.
    """
    # Convert labels to their numeric values if they are one-hot encoded
    if labels.ndim > 1:
        labels = np.argmax(labels, axis=1)
        print("labels[0] : ", labels[0])

    # Get the sorted indices based on labels
    sorted_indices = np.argsort(labels)

    # Sort the images and labels using the sorted indices
    sorted_images = images[sorted_indices]
    sorted_labels = labels[sorted_indices]
    print("sorted_images.shape : ", sorted_images.shape)
    print("sorted_labels.shape : ", sorted_labels.shape)
    print("sorted_labels[0] : ", sorted_labels[0])
    # verify type of values in sorted_labels
    print("type(sorted_labels[0]) : ", type(sorted_labels[0]))
    print("type(sorted_labels[0]) : ", type(sorted_labels[0].item()))
    print("sorted_labels[0].item() : ", sorted_labels[0].item())
    # verify type of values in sorted_images
    print("type(sorted_images[0]) : ", type(sorted_images[0]))
    print("type(sorted_images[0][0]) : ", type(sorted_images[0][0]))
    print("type(sorted_images[0]) : ", type(sorted_images[0].item()))
    print("sorted_images[0].item() : ", sorted_images[0].item())

    return sorted_images, sorted_labels


def load_mnist_data(file_path, from_save=False, sort_by_label=False):
    """
    Loads the MNIST dataset from a given file path.

    Parameters:
    file_path (str): Path to the dataset file.
    from_save (bool): Whether to load from a saved NumPy file.

    Returns:
    tuple: Tuple containing the normalized images and their one-hot encoded labels.
    """
    print("file_path : ", file_path)

    if from_save:
        # Assuming the saved file is a .npz file
        save_file = file_path.rsplit(".", 1)[0] + ".npz"
        if sort_by_labels:
            save_file = save_file.rsplit(".", 1)[0] + "_label_sorted.npz"
        print("save_file : ", save_file)
        try:
            with np.load(save_file, allow_pickle=True) as data:
                images = data["images"]
                labels_one_hot = data["labels_one_hot"]
        except IOError:
            print(f"Error loading file: {save_file}. File may not exist.")
            from_save = False
            try:
                if sort_by_label:
                    print("sort_by_label : ", sort_by_label)
                    file_path = sort_csv_file(file_path, sort_by_label)
                    print("file_path : ", file_path)
                data = np.loadtxt(file_path, delimiter=",", skiprows=1)
                labels = data[:, 0].astype(int)
                images = data[:, 1:] / 255.0
                num_classes = 10
                labels_one_hot = np.eye(num_classes)[labels]

            except IOError:
                print(f"Error loading file: {file_path}. File may not exist.")
                return None, None
    print("images and labels loaded from", save_file if from_save else file_path)

    if not from_save:
        # if sort_by_label:
        #     images, labels = sort_by_labels(images, labels_one_hot)
        #     # create a new csv file with the sorted images and labels
        #     print("labels[0] as int: ", labels[0])
        #     np.savetxt(
        #         save_file.rsplit(".", 1)[0] + "_label_sorted.csv",
        #         # np.c_[labels, images],
        #         np.c_[labels, images],
        #         delimiter=",",
        #     )
        #     labels_one_hot = np.eye(num_classes)[labels]
        np.savez(save_file, images=images, labels_one_hot=labels_one_hot)
        print("images and labels saved to", save_file)
    return images, labels_one_hot


class NeuralNetwork:
    """neural network with 1 hidden layer"""

    def __init__(
        self,
        input_size: int,  # input neurons quantity
        hidden_size: int,  # hidden neurons quantity
        output_size: int,  # output neurons quantity
        hidden_activation_function: str,  # activation function
        output_activation_function: str = "softmax",  # activation function
    ):
        """initializes the weights
        of the neural network
        with a normal distribution
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_activation_function = hidden_activation_function
        self.output_activation_function = output_activation_function

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
        # return np.tanh(-x)

    def rectified_linear_unit(self, x):
        """ReLU is the
        activation function
        Parameters:
        x (ndarray): Input array.

        Returns:
        ndarray: Output after applying the ReLU function.
        """
        return np.maximum(0, x)

    def rectified_linear_unit_leaky(self, x):
        """LeakyReLU is the
        activation function
        Parameters:
        x (ndarray): Input array.

        Returns:
        ndarray: Output after applying the LeakyReLU function.
        """
        return np.maximum(0.01 * x, x)

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
        return exp_x / exp_x.sum(axis=1, keepdims=True)
        # return exp_x / np.maximum(
            # sum_exp_x, epsilon
        # )  # Use maximum to prevent division by zero

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
        # Apply input-hidden activation function
        if self.hidden_activation_function == "tanh":
            self.hidden_output = self.tanh(self.hidden_input)
        elif self.hidden_activation_function == "ReLU":
            self.hidden_output = self.rectified_linear_unit(self.hidden_input)
        elif self.hidden_activation_function == "LeakyReLU":
            self.hidden_output = self.rectified_linear_unit_leaky(self.hidden_input)
        else:
            raise ValueError(
                f"Unknown activation function: {self.hidden_activation_function}"
            )

        # Hidden Layer to Output Layer
        # Weighted sum of hidden outputs
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output)
        # Apply hidden-output activation function
        if self.output_activation_function == "softmax":
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

        self.loss = self.mse_loss(y_one_hot, self.model_output)
        # print("loss : ", loss)

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
        d_weights_hidden_output = np.dot(self.hidden_output.T, output_error)

        # Update the weights with the derivatives (gradient descent)
        # 𝑊ℎ = 𝑊ℎ − 𝜇(𝑥𝑇 × 𝑒ℎ)
        self.weights_input_hidden -= learning_rate * d_weights_input_hidden
        # 𝑊𝑜 = 𝑊𝑜 − 𝜇(𝑦𝑜 × 𝑒𝑜)
        self.weights_hidden_output -= learning_rate * d_weights_hidden_output

        self.accuracy = np.sum(
            np.argmax(y_one_hot, axis=1) == np.argmax(self.model_output, axis=1)
        ) / y_one_hot.shape[0]
        return self.loss

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
        print("hidden_activation_function : ", self.hidden_activation_function)
        print("output_activation_function : ", self.output_activation_function)
        print("epochs : ", epochs)
        print("learning_rate : ", learning_rate)
        self.epochs = epochs
        self.learning_rate = learning_rate
        for epoch in range(epochs):
            self.forward(X)
            loss = self.backward(X, y_one_hot, learning_rate)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X: np.ndarray):  # input data
        """
        Predicts labels for given data.

        Returns:
        ndarray: Predicted labels.
        """
        self.forward(X)
        # return max value of each row
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
        plt.title(
            f"Confusion Matrix ; accuracy = {self.accuracy}\n activation functions = {self.hidden_activation_function}, {self.output_activation_function} ; hidden layer size = {self.hidden_size}\nepochs = {self.epochs} ; learning rate = {self.learning_rate} ; loss = {self.loss}"
        )

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

        # save
        plt.savefig(
            f"confusion/confusion_matrix_activation_functions_{self.hidden_activation_function}_{self.output_activation_function}_hidden_layer_size_{self.hidden_size}_epochs_{self.epochs}_learning_rate_{self.learning_rate}_loss_{self.loss}.png",
            bbox_inches="tight", # prevents the labels from being cut off
        )
        # Display the plot
        plt.show()



if __name__ == "__main__":
    X, y = load_mnist_data(
        "mnist_train.csv/mnist_train.csv", from_save=True, sort_by_label=True
    )

    # X.shape[0] = number of rows, X.shape[1] = number of columns
    input_size = X.shape[1]  # 784 = 28 * 28
    print("input_size : ", input_size)
    # hidden_size = 512
    # hidden_size = 256
    hidden_size = 128
    output_size = 10
    hidden_activation_function = "tanh"
    # hidden_activation_function = "ReLU"
    # hidden_activation_function = "LeakyReLU"
    output_activation_function = "softmax"
    # e = 1000
    e = 100
    # e = 1
    # mu = 0.1
    # mu = 0.05
    mu = 0.04
    # mu = 0.03
    # mu = 0.01
    # mu = 0.001

    nn = NeuralNetwork(
        input_size,
        hidden_size,
        output_size,
        hidden_activation_function,
        output_activation_function,
    )
    nn.train(
        X,
        y,
        epochs=e,
        learning_rate=mu,
    )

    X_test, y_test = load_mnist_data("mnist_test.csv/mnist_test.csv", from_save=True)
    nn.confusion_matrix(
        X_test,
        y_test,
    )
    # nn.visualize_prediction(X_test, y_test, 10)
    nn.visualize_predictions(X_test, y_test, 10)
