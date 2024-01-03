import csv
from datetime import date
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import sparse
from scipy.interpolate import griddata


def sort_csv_file(file_path, sort_by_label=False):
    """
    Sorts a CSV file by a given criterion.

    """
    print("sort_csv_file")
    print("sort_by_label : ", sort_by_label)
    print("file_path : ", file_path)
    data = np.loadtxt(file_path, delimiter=",", skiprows=1, dtype=int)
    if sort_by_label:
        # Assuming the label is in the first column
        sorted_data = data[data[:, 0].argsort()]
    sorted_file_path = file_path.rsplit(".", 1)[0] + "_sorted.csv"
    np.savetxt(sorted_file_path, sorted_data, delimiter=",", fmt="%d")

    print("sorted_file_path : ", sorted_file_path)
    return sorted_file_path  # Return the path of the sorted file for reference


def load_mnist_data(
    file_path: str,  # Path to the dataset file.
    from_save=False,  # Whether to load from a saved NumPy file.
    sort_by_label=False,  # Whether to sort the data by label.
    sparse=False,  # Whether to load the data as sparse matrices.
) -> tuple:  # Tuple containing the normalized images and their one-hot encoded labels.
    """
    Loads the MNIST dataset from a given file path.
    """
    print("file_path : ", file_path)
    if from_save:
        # Assuming the saved file is a .npz file
        save_file = file_path.rsplit(".", 1)[0] + ".npz"
        if sort_by_label:
            save_file = save_file.rsplit(".", 1)[0] + "_label_sorted.npz"
        print("save_file : ", save_file)
        try:
            with np.load(save_file, allow_pickle=True) as data:
                images = data["images"]
                labels_one_hot = data["labels_one_hot"]
        except IOError:
            print(f"Error loading file: {save_file}. File may not exist.")
            from_save = False

    if not from_save:
        try:
            print("trying with file_path : ", file_path)
            if sort_by_label:
                print("sort_by_label : ", sort_by_label)
                file_path = sort_csv_file(file_path, sort_by_label)
                print("file_path : ", file_path)
            data = np.loadtxt(file_path, delimiter=",", skiprows=1)
            labels = data[:, 0].astype(int)
            images = data[:, 1:] / 255.0
            # images = sparse.csr_matrix(images)

            num_classes = 10
            labels_one_hot = np.eye(num_classes)[labels]

        except IOError:
            print(f"Error loading file: {file_path}. File may not exist.")
            return None, None
        print("images and labels loaded from", save_file if from_save else file_path)

        if sparse:
            # np.savez(save_file, images_sparse=images_sparse, labels_one_hot=labels_one_hot)
            save_file = file_path.rsplit(".", 1)[0] + "_sparse.npz"
        np.savez(save_file, images=images, labels_one_hot=labels_one_hot)
        print("images and labels saved to", save_file)
    # return images_sparse, labels_one_hot
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
        weights_file_path=None,  # path to the weights file
    ):
        """initializes the weights
        of the neural network
        with a normal distribution
        """
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation_function = hidden_activation_function
        self.output_activation_function = output_activation_function

        if weights_file_path:
            self.load_weights(weights_file_path)
            self.hidden_size = self.weights_input_hidden.shape[1]
        else:
            self.hidden_size = hidden_size
            # input hidden is the output of the input layer
            # np.random.randn() returns a sample (or samples) from the “standard normal” distribution.
            # possible values are between -1 and 1
            self.weights_input_hidden = np.random.randn(input_size, hidden_size)
            # print("self.weights_input_hidden.shape : ", self.weights_input_hidden.shape)
            # print("self.weights_input_hidden : ", self.weights_input_hidden)
            # hidden output is the output of the hidden layer
            self.weights_hidden_output = np.random.randn(hidden_size, output_size)

    def print_weights(self, weights):
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                print(weights[i][j])

    def save_weights(self, file_path):
        """
        Saves the current weights of the neural network to a file.

        Parameters:
        file_path (str): The file path where the weights will be saved.
        """
        np.savez(
            file_path,
            weights_input_hidden=self.weights_input_hidden,
            weights_hidden_output=self.weights_hidden_output,
        )
        print(f"Weights saved to {file_path}")

    def load_weights(self, file_path):
        """
        Loads weights from a file and initializes the neural network with them.

        Parameters:
        file_path (str): The file path from which the weights will be loaded.
        """
        data = np.load(file_path)
        self.weights_input_hidden = data["weights_input_hidden"]
        self.weights_hidden_output = data["weights_hidden_output"]
        print(f"Weights loaded from {file_path}")

    def tanh(
        self, x: np.ndarray  # input data
    ) -> np.ndarray:  # Output after applying the tanh function.
        """
        tanh activation function
        """
        return np.tanh(x)

    def rectified_linear_unit(self, x: np.ndarray) -> np.ndarray:
        """
        ReLU activation function
        """
        return np.maximum(0, x)

    def rectified_linear_unit_leaky(self, x):
        """
        LeakyReLU activation function
        """
        return np.maximum(0.01 * x, x)

    def softmax(self, x: np.ndarray, epsilon=1e-12) -> np.ndarray:
        """
        Softmax activation function.
        """
        # to avoid large exponentials and possible overflows:
        # Shift each row of x by subtracting its max value.

        x_shifted = x - np.max(x, axis=1, keepdims=True)

        # Calculate the softmax with the shifted values.
        exp_x_shifted = np.exp(x_shifted)
        sum_exp_x_shifted = np.sum(exp_x_shifted, axis=1, keepdims=True)

        # Calculate softmax and prevent division by zero.
        softmax_output = np.divide(
            exp_x_shifted, np.maximum(sum_exp_x_shifted, epsilon)
        )

        return softmax_output

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
        d_weights_hidden_output = np.dot(self.hidden_output.T, output_error)

        # Update the weights with the derivatives (gradient descent)
        # 𝑊ℎ = 𝑊ℎ − 𝜇(𝑥𝑇 × 𝑒ℎ)
        self.weights_input_hidden -= learning_rate * d_weights_input_hidden
        # 𝑊𝑜 = 𝑊𝑜 − 𝜇(𝑦𝑜 × 𝑒𝑜)
        self.weights_hidden_output -= learning_rate * d_weights_hidden_output

        self.fit = (
            np.sum(np.argmax(y_one_hot, axis=1) == np.argmax(self.model_output, axis=1))
            / y_one_hot.shape[0]
        )
        return self.loss

    def exponential_decay_lr(
        self, initial_lr, epoch, total_epochs, decay_rate=0.1, end_lr=0.0001
    ):
        """
        Calculates the exponentially decaying learning rate.

        Parameters:
        initial_lr (float): Initial learning rate.
        epoch (int): Current epoch.
        total_epochs (int): Total number of epochs.
        decay_rate (float): Decay rate.

        Returns:
        float: Adjusted learning rate.
        """
        return initial_lr * np.exp(-decay_rate * epoch / total_epochs)
        # lr_decay = (initial_lr - end_lr) / total_epochs
        # return initial_lr - lr_decay * epoch

    def train(
        self,
        X: np.ndarray,  # input data
        y_one_hot: np.ndarray,  # one-hot encoded labels
        epochs=100,  # number of training epochs
        learning_rate=0.01,  # learning rate
        learning_rate_adaptative=False,
        batch_size=32,  # batch size
        # batch_size=1000,  # batch size
        weights_save_path=None,  # path to save the weights
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
        learning_rate_decay = learning_rate / epochs
        # self.accuracy = 1 - learning_rate
        self.fit = 0.0

        best_fit = 0.0
        best_weights_input_hidden = None
        best_weights_hidden_output = None

        for epoch in range(epochs):
            if learning_rate_adaptative:
                # reducing the learning rate to reach near 0
                learning_rate = self.learning_rate * ((1 - self.fit) ** (100 + epoch))
                # learning_rate = self.learning_rate * (1 - self.fit) ** self.hidden_size
            print(
                f"epoch : {epoch} ; learning_rate : {learning_rate} ; fit : {self.fit}"
            )
            # Shuffle the dataset
            permutation = np.random.permutation(X.shape[0])
            x_shuffled = X[permutation]
            y_shuffled = y[permutation]

            for i in range(0, X.shape[0], batch_size):
                x_batch = x_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]

                # Forward and backward pass for the batch
                self.forward(x_batch)
                loss = self.backward(x_batch, y_batch, learning_rate)

            print(f"Epoch {epoch}, Loss: {loss}, fit: {self.fit}")
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
            if self.fit > best_fit:
                best_fit = self.fit
                best_weights_input_hidden = self.weights_input_hidden.copy()
                best_weights_hidden_output = self.weights_hidden_output.copy()
            if loss < 0.000001:
                print("fit : ", self.fit)

                weights_save_path = (
                    weights_save_path
                    + "_loss:"
                    + str(loss)
                    + "_fit:"
                    + str(self.fit)
                    + str(date.today())
                    + str(time.strftime("%H%M%S"))
                )
                self.save_weights(weights_save_path)
                break

        # Restore the best state of the network
        self.weights_input_hidden = best_weights_input_hidden
        self.weights_hidden_output = best_weights_hidden_output
        self.fit = best_fit

        print(f"Best fit: {self.fit}")
        if weights_save_path:
            weights_save_path = (
                weights_save_path + self.fit + date.today() + time.strftime("%H%M%S")
            )
            self.save_weights(weights_save_path)

    def predict(self, X: np.ndarray):  # input data
        """
        Predicts labels for given data.

        Returns:
        ndarray: Predicted labels.
        """
        self.forward(X)
        # return max value of each row
        return np.argmax(self.model_output, axis=1)

    def calculate_accuracy(self, X: np.ndarray, y_true: np.ndarray) -> float:
        """
        Calculates the accuracy of the neural network on a given dataset.

        Parameters:
        X (np.ndarray): The input data.
        y_true (np.ndarray): The true labels, expected to be one-hot encoded.

        Returns:
        float: The accuracy of the model.
        """
        predictions = self.predict(X)
        correct_predictions = np.sum(predictions == np.argmax(y_true, axis=1))
        accuracy = correct_predictions / X.shape[0]
        return accuracy

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
            bbox_inches="tight",  # prevents the labels from being cut off
        )
        plt.show()

    def save_results_to_csv(self, results, file_path):
        """
        Saves the results dictionary to a CSV file.

        Parameters:
        results (dict): The results dictionary.
        file_path (str): Path to the CSV file.
        """
        with open(file_path, "w", newline="") as file:
            writer = csv.writer(file)
            for key, value in results.items():
                lr, epochs, hidden_size = key
                accuracy = value
                writer.writerow([lr, epochs, hidden_size, accuracy])

    def test_combinations(
        self,
        X: np.ndarray,  # Input data
        y: np.ndarray,  # Target labels
        X_test: np.ndarray,  # Input data for testing
        y_test: np.ndarray,  # Target labels for testing
        learning_rates: list,  # List of learning rates to test
        learning_rate_adaptative=False,
        epochs_list: list = [1],  # List of numbers of epochs to test
        hidden_sizes: list = [784],  # List of hidden layer sizes to test
        batch_size=32,  # Batch size
        weights_random_samples=10,  # Number of random samples of weights to test
    ) -> dict:
        """
        Tests different combinations of learning rates, epochs, and hidden layer sizes.

        Returns:
        dict: Dictionary containing accuracies for each combination.
        """

        results = {}
        epochs_max = max(epochs_list)  # todo

        for lr in learning_rates:
            print(f"LR: {lr}")
            for epochs in epochs_list:
                print(f"LR: {lr}, Epochs: {epochs}")
                for hidden_size in hidden_sizes:
                    print(f"LR: {lr}, Epochs: {epochs}, Hidden Size: {hidden_size}")
                    accuracy_mean = 0.0
                    for sample in range(weights_random_samples):
                        print(
                            f"LR: {lr}, Epochs: {epochs}, Hidden Size: {hidden_size}, Sample: {sample}/{weights_random_samples}"
                        )
                        weights_input_hidden = np.random.randn(
                            self.input_size, hidden_size
                        )
                        weights_hidden_output = np.random.randn(
                            hidden_size, self.output_size
                        )
                        # Reinitialize the network with the new hidden layer size
                        self.hidden_size = hidden_size
                        self.weights_input_hidden = weights_input_hidden
                        self.weights_hidden_output = weights_hidden_output

                        weights_save_path = (
                            "weights/weights_"
                            + self.hidden_activation_function
                            + "_"
                            + self.output_activation_function
                            + "_"
                            + str(lr)
                            # + "_"
                            # + str(epochs)
                            + "_"
                            + str(hidden_size)
                            # + "_"
                            # + str(sample)
                            # + "_"
                        )
                        # Train the network
                        self.train(
                            X,
                            y,
                            epochs=epochs,
                            learning_rate=lr,
                            learning_rate_adaptative=learning_rate_adaptative,
                            batch_size=32,
                            weights_save_path=weights_save_path,
                            #    keep_best_weights=False
                        )

                        # Test the network
                        self.accuracy = self.calculate_accuracy(X_test, y_test)

                        # Record the accuracy
                        current_accuracy = self.accuracy
                        print(f"current_accuracy : {current_accuracy}")
                        accuracy_mean += current_accuracy / weights_random_samples

                    results[(lr, epochs, hidden_size)] = accuracy_mean

                    print(
                        f"LR: {lr},"
                        f" Epochs: {epochs},"
                        f" Hidden Size: {hidden_size},"
                        f" Samples: {weights_random_samples},"
                        f" Accuracy mean: {accuracy_mean}"
                    )
        save_file = (
            "test_combinations_results"
            + date.today().strftime("%Y%m%d")
            + time.strftime("%H%M%S")
        )
        try:
            # # Convert the dictionary to a 2D array
            # results_array = np.array(list(results.items()))

            # # Save the array
            self.save_results_to_csv(results, save_file)
            print(f"Results saved to {save_file}")
        except IOError:
            print(f"Error saving file: {save_file}.")
        return results

    def plot_results(self, results):
        """
        Plots the results of the test_combinations method.

        """
        # sort the results by keys ascending values
        results = dict(
            sorted(results.items(), key=lambda item: item[0])
        )  # item[0] is the key of the dictionary
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection="3d")

        # Extract data for plotting
        learning_rates = np.array([k[0] for k in results.keys()])
        epochs = np.array([k[1] for k in results.keys()])
        hidden_sizes = np.array([k[2] for k in results.keys()])
        accuracies = np.array(list(results.values()))

        # Calculate point sizes (e.g., normalize accuracies and scale them)
        normalized_accuracies = (accuracies - accuracies.min()) / (
            accuracies.max() - accuracies.min()
        )
        print("normalized_accuracies : ", normalized_accuracies)

        base_size = 50  # Base size for the smallest points
        # scaled_point_sizes = base_size + 2000 * normalized_accuracies  # Scale up the normalized accuracies
        scaled_point_sizes = (
            base_size + 1000 * normalized_accuracies
        )  # Scale up the normalized accuracies

        # Scatter plot with dynamic point sizes
        img = ax.scatter(
            learning_rates,
            epochs,
            hidden_sizes,
            c=accuracies,
            s=scaled_point_sizes,  # Apply calculated sizes here
            # cmap=plt.viridis()
            # cmap='RdYlGn'
            cmap="gist_rainbow",
        )
        fig.colorbar(img)

        # Add labels and title
        ax.set_xlabel("Learning Rate")
        ax.set_ylabel("Epochs")
        ax.set_zlabel("Hidden Layer Size")
        ax.set_title(
            f"Activation Functions: {self.hidden_activation_function}, {self.output_activation_function}; accuracy max: {accuracies.max()}"
        )

        # Set axis limits
        ax.set_xlim([learning_rates.min(), learning_rates.max()])
        ax.set_ylim([epochs.min(), epochs.max()])
        ax.set_zlim([hidden_sizes.min(), hidden_sizes.max()])

        # Draw lines
        for lr, ep, hs, acc in zip(learning_rates, epochs, hidden_sizes, accuracies):
            color = plt.cm.viridis(acc)
            ax.plot(
                [lr, lr],
                [ep, ep],
                [hidden_sizes.min(), hs],
                color=color,
                linestyle="--",
                linewidth=0.5,
            )  # vertical line
            ax.plot(
                [lr, lr],
                [epochs.min(), ep],
                [hs, hs],
                color=color,
                linestyle="--",
                linewidth=0.5,
            )  # learning rate to epochs
            ax.plot(
                [learning_rates.min(), lr],
                [ep, ep],
                [hs, hs],
                color=color,
                linestyle="--",
                linewidth=0.5,
            )  # epochs to hidden size

        # Save and show
        plt.savefig(
            f"results_plot_activation_functions_{self.hidden_activation_function}_{self.output_activation_function}_{date.today()}{time.strftime('%H%M%S')}.png",
            bbox_inches="tight",
        )
        plt.show()


if __name__ == "__main__":
    X, y = load_mnist_data(
        "src/mnist_train.csv",
        from_save=True,
        #    sort_by_label=True
        sort_by_label=False,
        sparse=False,
    )
    X_test, y_test = load_mnist_data("src/mnist_test.csv", from_save=True)

    # # X.shape[0] = number of rows, X.shape[1] = number of columns
    input_size = X.shape[1]  # 784 = 28 * 28
    hidden_size = 21952
    hidden_activation_function = "tanh"
    # # hidden_activation_function = "ReLU"
    # # hidden_activation_function = "LeakyReLU"
    output_activation_function = "softmax"
    OUTPUT_SIZE = 10

    nn = NeuralNetwork(
        input_size,
        hidden_size,
        OUTPUT_SIZE,
        hidden_activation_function,
        output_activation_function,
        # weights_file_path="weights",
    )
    e = 10
    mu = 0.03
    # nn.train(
    #     X,
    #     y,
    #     epochs=e,
    #     learning_rate=mu,
    #     batch_size=32,
    #     weights_save_path="weights/weights_",
    # )

    # nn.confusion_matrix(
    #     X_test,
    #     y_test,
    # )
    # nn.visualize_predictions(X_test, y_test, 10)

    # learning_rates = [0.001, 0.01, 1]
    # learning_rates = [0.0001, 0.001, 0.01, 0.05, 0.1]
    # learning_rates = [0.0000001, 0.0001, 0.02, 0.025, 0.03, 0.1]
    # learning_rates = [0.02, 0.025, 0.03]
    learning_rates = reversed([0.02, 0.025, 0.03])
    # epochs_list = [10, 100, 1000]
    # epochs_list = [1, 10, 100]
    epochs_list = [100, 10, 1]
    # epochs_list = [10, 1]
    # epochs_list = [1]
    # epochs_list = [1, 2, 3]
    # epochs_list = (value for value in np.arange(100, 1000, 100))
    # hidden_sizes = [28, 784, 21952]
    # hidden_sizes = [1, 2, 3]
    # hidden_sizes = [1, 28, 21952]
    # hidden_sizes = [21952]
    # hidden_sizes = [21952, 28, 1]
    # hidden_sizes = [784, 56, 28]
    hidden_sizes = [784]
    # hidden_sizes = [14, 28, 784]
    batch_size = 32

    results = nn.test_combinations(
        X,
        y,
        X_test,
        y_test,
        learning_rates,
        learning_rate_adaptative=True,
        epochs_list=epochs_list,
        hidden_sizes=hidden_sizes,
        batch_size=batch_size,
        # weights_random_samples=10,
        weights_random_samples=1,
    )
    nn.plot_results(results)
