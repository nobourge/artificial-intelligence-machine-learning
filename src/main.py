import csv
from datetime import date
from math import sqrt
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import sparse
from scipy.interpolate import griddata

# print("sys.path : ", sys.path)
# print current working directory
print("current working directory : ", os.getcwd())
# print current working directory content
print("current working directory content : ", os.listdir(os.getcwd()))
from auto_indent import AutoIndent

# from src.auto_indent import AutoIndent
# from .auto_indent import AutoIndent

sys.stdout = AutoIndent(sys.stdout)


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
        self.weights_file_path = weights_file_path
        self.reset(hidden_size, weights_file_path)

    def reset(
        self,
        hidden_size=None,
        weights_file_path=None,
    ):
        self.loss = float("inf")
        self.fit = 0.0
        self.loss_history = []
        self.fit_history = []
        self.accuracy = 0.0
        self.epochs = 0
        self.learning_rate = 0.0
        if weights_file_path:
            self.load_weights(weights_file_path)
            self.hidden_size = self.weights_input_hidden.shape[1]
        else:
            if hidden_size:
                self.hidden_size = hidden_size
            # input hidden is the output of the input layer
            self.weights_input_hidden = np.random.randn(
                self.input_size, self.hidden_size
            )
            # hidden output is the output of the hidden layer
            self.weights_hidden_output = np.random.randn(
                self.hidden_size, self.output_size
            )

    def get_details(
        self,
        mode="plot",
        include_hidden_size=True,
        include_hidden_activation=True,
        include_output_activation=True,
        include_batch_rate=True,
        include_learning_rate=True,
        include_epochs=True,
        include_loss=True,
        include_fit=True,
    ):
        """
        Returns a string containing details about the network.
        Parameters:
        mode (str): 'plot' for plot annotation, 'filename' for file naming.
        include_* (bool): Flags to control the inclusion of each attribute.
        """
        newline_char = "\n" if mode == "plot" else "_"
        colon_char = ":" if mode == "plot" else "_"

        details = ""

        if self.hidden_size is not None and include_hidden_size:
            details += f"{newline_char}hidden{colon_char}{self.hidden_size}"

        if self.hidden_activation_function is not None and include_hidden_activation:
            details += f"{newline_char}Activations{colon_char}{newline_char}hidden{colon_char}{self.hidden_activation_function}"

        if self.output_activation_function is not None and include_output_activation:
            details += (
                f"{newline_char}output{colon_char}{self.output_activation_function}"
            )

        if self.batch_rate is not None and include_batch_rate:
            details += f"{newline_char}Batch_rate{colon_char}{self.batch_rate}"

        if self.learning_rate is not None and include_learning_rate:
            details += f"{newline_char}Learning_rate{colon_char}{self.learning_rate}"
        if self.epochs is not None and include_epochs:
            details += f"{newline_char}Epochs{colon_char}{self.epochs}"
        if self.loss is not None and include_loss:
            details += f"{newline_char}Loss{colon_char}{self.loss}"

        if self.fit is not None and include_fit:
            details += f"{newline_char}Fit{colon_char}{self.fit}"

        return details

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

    def exponential_decay_lr(self, initial_lr, epoch, total_epochs, decay_rate=0.1):
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

    def adapt_learning_rate(
        self,
        epoch_factor=None,
        epoch=None,
        neuron_quantity_factor=True,
        batch_rate_factor=True,
        loss_factor=1.0,
        learning_rate_adaptation=1.0,
    ):
        learning_rate = self.learning_rate
        if epoch_factor:
            # epoch reduces exponentially learning rate
            learning_rate = self.exponential_decay_lr(learning_rate, epoch, self.epochs)
            print("exponential decayed learning_rate : ", learning_rate)
        if neuron_quantity_factor:
            # neuron quantity augments learning rate
            learning_rate = learning_rate * (
                self.hidden_size**learning_rate_adaptation
            )
            print("neuron quantity adapted learning_rate : ", learning_rate)
        # if batch_rate_factor:
        #     # batchize augments learning rate
        #     learning_rate = learning_rate ** (batch_rate)
        #     print("batch rate adapted learning_rate : ", learning_rate)
        # fitness reduces exponentially learning rate
        # if self.fit < 1:
        #     learning_rate = learning_rate * ((1 - self.fit) ** 2)
        # print("fit adapted learning_rate : ", learning_rate)
        # loss reduces exponentially learning rate
        if loss_factor:
            # reducing the learning rate to reach near 0
            if self.loss == float("inf"):
                loss_factor = 1.0
            else:
                # ensure loss is under 1
                print("loss : ", self.loss)
                loss_below_0 = self.loss / 10
                loss_factor = loss_below_0 ** (loss_factor**loss_factor)
            print("loss_factor : ", loss_factor)
            learning_rate = learning_rate * loss_factor
            print("loss adapted learning_rate : ", learning_rate)
        return learning_rate

    def train(
        self,
        X: np.ndarray,  # input data
        y_one_hot: np.ndarray,  # one-hot encoded labels
        x_test: np.ndarray = None,
        y_test: np.ndarray = None,
        epochs=100,  # number of training epochs
        learning_rate=0.01,  # learning rate
        learning_rate_adaptation=1.0,
        loss_factor=1.0,
        batch_rate=0.1,  # batch size
        show_training_progress=False,  # whether to show the training progress
        weights_save_path=None,  # path to save the weights
        keep_best_weights=True,
    ):
        """
        Trains the neural network.
        """
        print("train")
        print("hidden_activation_function : ", self.hidden_activation_function)
        print("output_activation_function : ", self.output_activation_function)
        print("X.shape : ", X.shape)
        print("y_one_hot.shape : ", y_one_hot.shape)
        print("epochs : ", epochs)
        print("learning_rate : ", learning_rate)
        print("batch_rate : ", batch_rate)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_rate = batch_rate
        initial_lr = learning_rate
        # self.fit = 0.0

        best_fit = 0.0
        best_loss = float("inf")
        if keep_best_weights:
            best_weights_input_hidden = None
            best_weights_hidden_output = None

        for epoch in range(epochs):
            print(f"epoch : {epoch}")
            if learning_rate_adaptation:
                learning_rate = self.adapt_learning_rate(
                    epoch=epoch,
                    epoch_factor=False,
                    neuron_quantity_factor=False,
                    batch_rate_factor=False,
                    loss_factor=loss_factor,
                    learning_rate_adaptation=learning_rate_adaptation,
                )
            print(
                f"learning_rate : {learning_rate}"
                # ; fit : {self.fit}"
            )
            # Shuffle the dataset
            permutation = np.random.permutation(X.shape[0])
            x_shuffled = X[permutation]
            y_shuffled = y[permutation]

            loss = 0.0
            batch_size = int(batch_rate * X.shape[0])
            print("batch_size : ", batch_size)
            best_batch_loss = float("inf")
            for i in range(0, X.shape[0], batch_size):
                x_batch = x_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]

                # Forward and backward pass for the batch
                self.forward(x_batch)
                batch_loss = self.backward(x_batch, y_batch, learning_rate)
                # print(f"batch_loss {ii}: {batch_loss}")
                if batch_loss < best_batch_loss:
                    best_batch_loss = batch_loss

                loss += batch_loss / len(range(0, X.shape[0], batch_size))
            print(f"best_batch_loss : {best_batch_loss}")
            self.loss_history.append(self.loss)
            self.fit_history.append(self.fit)
            if x_test is not None and y_test is not None:
                predictions = self.predict(x_test)
                self.accuracy = self.calculate_accuracy(predictions, y_test)
                self.accuracy_history.append(self.accuracy)
            print(f"Epoch {epoch}, Loss: {loss}, fit: {self.fit}")
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
            if best_fit < self.fit:
                best_fit = self.fit
            if loss < best_loss:
                best_loss = loss
                if keep_best_weights:
                    best_weights_input_hidden = self.weights_input_hidden.copy()
                    best_weights_hidden_output = self.weights_hidden_output.copy()
            if loss < 0.00000001:
                print("fit : ", self.fit)
                if not weights_save_path:
                    weights_save_path = "src/weights_"
                break

        print(f"Best fit: {best_fit}")
        print(f"Best loss: {best_loss}")
        if keep_best_weights:
            # Restore the best state of the network
            self.weights_input_hidden = best_weights_input_hidden
            self.weights_hidden_output = best_weights_hidden_output
            self.loss = best_loss
            self.fit = best_fit

        if weights_save_path:
            weights_save_path = (
                weights_save_path
                + f"_fit_max_{self.fit:.2f}"
                + "_"
                + self.get_details(mode="filename")
                + str(date.today())
                + "_"
                + str(time.strftime("%H%M%S"))
            )
            self.save_weights(weights_save_path)
        if show_training_progress:
            self.plot_training_progress()

    def plot_training_progress(self):
        plt.figure(figsize=(8, 6))

        # Plotting both loss and fit on the same subplot
        plt.plot(self.loss_history, label="Loss", color="blue")

        plt.title("Training Progress")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.legend(loc="upper left")

        # Secondary y-axis for Fit (if scales are very different)
        ax2 = plt.gca().twinx()
        ax2.plot(self.fit_history, label="Fit", color="red")
        ax2.set_ylabel("Fit", color="red")
        ax2.tick_params(axis="y", labelcolor="red")

        # Third y-axis for accuracy (if scales are very different)
        ax3 = plt.gca().twinx()
        ax3.plot(self.accuracy_history, label="Accuracy", color="green")
        ax3.set_ylabel("Accuracy", color="green")
        ax3.tick_params(axis="y", labelcolor="green")

        # Global title with network properties
        global_title = "Neural Network Training Progress"
        plt.suptitle(global_title)

        # Annotation with network properties

        plt.annotate(
            self.get_details(),
            xy=(0.5, -0.15),
            xycoords="axes fraction",
            ha="left",
            va="top",
            fontsize=10,
            color="gray",
        )

        # Save the plot
        file_path = "doc/out/progress/" + self.get_details(mode="filename") + ".png"
        plt.savefig(
            file_path,
            bbox_inches="tight",  # prevents the labels from being cut off
        )

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show(block=False)

        plt.pause(0.1)

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

    def calculate_accuracy(self, predictions: np.ndarray, y_true: np.ndarray) -> float:
        """
        Calculates the accuracy of the neural network on a given dataset.

        Parameters:
        X (np.ndarray): The input data.
        y_true (np.ndarray): The true labels, expected to be one-hot encoded.

        Returns:
        float: The accuracy of the model.
        """
        correct_predictions = np.sum(predictions == np.argmax(y_true, axis=1))
        accuracy = correct_predictions / predictions.shape[0]
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
        plt.show(block=False)
        plt.pause(0.1)

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
        plt.show(block=False)
        plt.pause(0.1)

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

        self.accuracy = self.calculate_accuracy(y_pred, y_true)

        # Create a plot to visualize the confusion matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(
            f"Confusion Matrix ; loss = {self.loss}; fit = {self.fit}; accuracy = {self.accuracy}\n activation functions = {self.hidden_activation_function}, {self.output_activation_function} ; hidden layer size = {self.hidden_size}\nepochs = {self.epochs} ; learning rate = {self.learning_rate} ; loss = {self.loss}"
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
        file_path = "doc/out/confusion/" + self.get_details(mode="filename") + ".png"
        plt.savefig(
            file_path,
            bbox_inches="tight",  # prevents the labels from being cut off
        )
        plt.show(block=False)
        plt.pause(0.1)

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
        learning_rates: list[float] = [0.01],  # List of learning rates to test
        learning_rate_adaptation: float = 1.0,
        loss_factor: float = 1.0,
        epochs_list: list = [1],  # List of numbers of epochs to test
        hidden_sizes: list = [784],  # List of hidden layer sizes to test
        batch_rate=0.1,  # Batch size
        weights_random_samples=10,  # Number of random samples of weights to test
        show_training_progress=False,  # Whether to show the training progress
        weights_save_path=None,  # Path to save the weights
    ) -> dict:
        """
        Tests different combinations of learning rates, epochs, and hidden layer sizes.

        Returns:
        dict: Dictionary containing accuracies for each combination.
        """
        results = {}
        epochs_list = sorted(epochs_list)

        for lr in learning_rates:
            print(f"LR: {lr}")
            for hidden_size in hidden_sizes:
                print(f"Hidden Size: {hidden_size}")
                # accuracy_mean = 0.0
                accuracy_mean_list = [0.0] * len(epochs_list)
                print(f"accuracy_mean_list : {accuracy_mean_list}")
                for sample in range(weights_random_samples):
                    print(f"Sample: {sample+1}/{weights_random_samples}")
                    # Reinitialize the network with the new hidden layer size
                    self.reset(hidden_size=hidden_size)
                    epochs_done = 0
                    for epochs in epochs_list:
                        print(f"Epochs: {epochs}")
                        # Train the network
                        self.train(
                            X,
                            y,
                            epochs=epochs - epochs_done,
                            learning_rate=lr,
                            learning_rate_adaptation=learning_rate_adaptation,
                            loss_factor=loss_factor,
                            batch_rate=batch_rate,
                            show_training_progress=show_training_progress,
                            weights_save_path=weights_save_path,
                            keep_best_weights=False,
                        )
                        epochs_done += epochs
                        print(f"self.fit : {self.fit}")
                        print(f"self.loss : {self.loss}")

                        # Test the network
                        predictions = self.predict(X_test)
                        self.accuracy = self.calculate_accuracy(predictions, y_test)

                        # Record the accuracy
                        current_accuracy = self.accuracy
                        print(f"current_accuracy : {current_accuracy}")
                        # accuracy_mean += current_accuracy / weights_random_samples
                        accuracy_mean_list[epochs_list.index(epochs)] += (
                            current_accuracy / weights_random_samples
                        )
                print(f"accuracy_mean_list : {accuracy_mean_list}")
                for epochs in epochs_list:
                    print(f"Epochs: {epochs}")
                    epoch_accuracy_mean = accuracy_mean_list[epochs_list.index(epochs)]
                    results[(lr, epochs, hidden_size)] = epoch_accuracy_mean

                    print(f" Accuracy mean: {epoch_accuracy_mean}")
        save_file = (
            "doc/out/test_combinations_results/"
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
        # Logarithmic transformation
        log_learning_rates = np.log10(learning_rates)
        epochs = np.array([k[1] for k in results.keys()])
        epochs_log = np.log10(epochs)
        hidden_sizes = np.array([k[2] for k in results.keys()])
        hidden_sizes_log = np.log10(hidden_sizes)
        accuracies = np.array(list(results.values()))
        accuracy_max = accuracies.max()
        accuracy_min = accuracies.min()

        # Calculate point sizes (e.g., normalize accuracies and scale them)
        # Check if all accuracies are the same
        if accuracy_max == accuracy_min:
            # Handle the case where all accuracies are the same
            # For example, you might choose to set normalized_accuracies to zero or some other default value
            normalized_accuracies = np.zeros_like(accuracies)
        else:
            # Calculate normalized accuracies
            normalized_accuracies = (accuracies - accuracies.min()) / (
                accuracy_max - accuracy_min + np.finfo(float).eps
            )

        print("normalized_accuracies : ", normalized_accuracies)

        base_size = 10  # smallest point
        scaled_point_sizes = (
            base_size + (normalized_accuracies * 10) ** 2
        )  # Scale up the normalized accuracies

        # Scatter plot with dynamic point sizes
        img = ax.scatter(
            # learning_rates,
            log_learning_rates,
            epochs_log,
            hidden_sizes_log,
            c=accuracies,
            s=scaled_point_sizes,  # Apply calculated sizes here
            # cmap=plt.viridis()
            # cmap='RdYlGn'
            cmap="gist_rainbow",
        )
        fig.colorbar(img)

        # Add labels and title
        ax.set_xlabel("Learning Rate log")
        # Set the tick labels for the x-axis to represent the original learning rates
        # ax.set_xticks(log_learning_rates)
        # ax.set_xticklabels(learning_rates)
        ax.set_ylabel("Epochs log")
        ax.set_zlabel("Hidden Layer Size log")
        ax.set_title("Parameters combinations Accuracies")

        # Set axis limits
        # ax.set_xlim([learning_rates.min(), learning_rates.max()])
        ax.set_xlim([log_learning_rates.min(), log_learning_rates.max()])
        # ax.set_xscale("log")
        ax.set_ylim([epochs_log.min(), epochs_log.max()])
        # ax.set_yscale("normal")
        ax.set_zlim([hidden_sizes_log.min(), hidden_sizes_log.max()])

        # Draw lines
        # for lr, ep, hs, acc in zip(learning_rates, epochs, hidden_sizes, accuracies):
        for lr, ep, hs, acc in zip(
            log_learning_rates, epochs_log, hidden_sizes_log, accuracies
        ):
            # color = plt.cm.viridis(acc)
            color = plt.cm.gist_rainbow(acc)
            ax.plot(
                [lr, lr],
                [ep, ep],
                [hidden_sizes_log.min(), hs],
                color=color,
                linestyle="--",
                linewidth=0.5,
            )  # vertical line
            ax.plot(
                [lr, lr],
                [epochs_log.min(), ep],
                [hs, hs],
                color=color,
                linestyle="--",
                linewidth=0.5,
            )  # learning rate to epochs
            ax.plot(
                # [learning_rates.min(), lr],
                [log_learning_rates.min(), lr],
                [ep, ep],
                [hs, hs],
                color=color,
                linestyle="--",
                linewidth=0.5,
            )  # epochs to hidden size

            # lines to the axes
            # Draw lines to the x-axis
            ax.plot(
                [lr, lr],
                [ep, ep],
                [ax.get_zlim()[0], hs],
                color="gray",
                linewidth=0.5,
                linestyle="--",
            )
            # Draw lines to the y-axis
            ax.plot(
                [lr, lr],
                [ax.get_ylim()[0], ep],
                [hs, hs],
                color="gray",
                linewidth=0.5,
                linestyle="--",
            )
            # Draw lines to the z-axis
            ax.plot(
                [ax.get_xlim()[0], lr],
                [ep, ep],
                [hs, hs],
                color="gray",
                linewidth=0.5,
                linestyle="--",
            )

            # Add accuracies
            ax.text(
                lr,
                ep,
                hs,
                f"{acc:.2f}",
                color="black",
            )
        ax.annotate(
            f"accuracy max: {accuracy_max}\n{self.get_details(include_hidden_size=False, include_learning_rate=False, include_epochs=False, include_loss=False, include_fit=False)}",
            xy=(0.5, -0.15),
            xycoords="axes fraction",
            ha="left",
            va="top",
            fontsize=10,
            color="gray",
        )
        plt.tight_layout()
        # Save and show
        details = self.get_details(
            mode="filename",
            include_hidden_size=False,
            include_learning_rate=False,
            include_loss=False,
            include_fit=False,
        )
        plt.savefig(
            "doc/out/accuracy/"
            + f"accuracy_max_{accuracy_max:.2f}_min_{accuracy_min:.2f}_"
            + f"{details}"
            # +f"_{date.today()}{time.strftime('%H%M%S')}
            + ".png",
            bbox_inches="tight",
        )
        plt.show(block=False)
        plt.pause(0.1)


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
    # hidden_size = 21952
    hidden_size = 784
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
        # weights_file_path="src/weights_.npz",
    )
    e = 100
    mu = 0.03
    # nn.train(
    #     X,
    #     y,
    #     X_test,
    #     y_test,
    #     epochs=e,
    #     learning_rate=mu,
    #     batch_size=32,
    #     weights_save_path="src/weights_",
    # )

    # nn.confusion_matrix(
    #     X_test,
    #     y_test,
    # )
    # nn.visualize_predictions(X_test, y_test, 10)

    learning_rates = [
        # 0.0000000000000000000000000000000000000000000000000000000000000001,
        # 0.000000000000000000000000000000001,
        # 0.00000000000000001,
        # 0.0000000001,
        # 0.0000001,
        # 0.0001,
        # 0.001,
        # 0.02,
        # 0.025,
        0.03,
        # 0.1,
        # 0.9,
    ]
    epochs_list = [
        1,
        10,
        100,
        #    1000
    ]
    hidden_sizes = [
        # 1,
        # 28,
        784,
        # 21952,
    ]

    batch_rates = [
        # 0.0001,
        0.0005,
        #0.001,
        #    0.01,
        # 0.1,
        #    0.2,
        # 0.5,
        # 1.0,
    ]

    loss_factor_exponents = [
        #1.1,
         1.5,
        # 1.8,
        # 2.0,
        # 3.0,
        # 0.1,
        # 0.5,
        # 1.0,
        # 4.0,
        # 5.0,
        # 10.0,
        # 100.0,
    ]

    # for learning_rate_adaptation in learning_rate_adaptations:
    #     print("learning_rate_adaptation : ", learning_rate_adaptation)
    for loss_factor_exponent in loss_factor_exponents:
        print("loss_factor_exponent : ", loss_factor_exponent)
        nn.loss_factor = loss_factor_exponent
        for batch_rate in batch_rates:
            print("batch_rate : ", batch_rate)
            results = nn.test_combinations(
                X,
                y,
                X_test,
                y_test,
                learning_rates=learning_rates,
                # learning_rate_adaptation=learning_rate_adaptation,
                loss_factor=loss_factor_exponent,
                epochs_list=epochs_list,
                hidden_sizes=hidden_sizes,
                # batch_rates=batch_rates,
                batch_rate=batch_rate,
                # weights_random_samples=10,
                weights_random_samples=1,
                show_training_progress=False,
                # weights_save_path="src/weights/weights_",
            )
            nn.plot_results(results)
        wait = input("close results ? (y/n) : ")
