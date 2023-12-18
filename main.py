import numpy as np
import matplotlib.pyplot as plt

def load_mnist_data(file_path):
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    
    #... 
    labels = None
    num_classes = 10
    labels_one_hot = np.eye(num_classes)[labels] # Qu'est-ce qu'un encodage one-hot ?
    
    images = None
    images = images / 255.0

    return images, labels_one_hot

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)

    def tanh(self, x):
        # ...
        return None

    def softmax(self, x):
        # ...
        return None

    def mse_loss(self, y_true, y_pred):
        # ...
        return None

    def forward(self, X):
        # ...
        
        self.hidden_input = None
        self.hidden_output = None

        self.output_input = None
        self.model_output = None

    def backward(self, X, y_one_hot, learning_rate=0.01):
        # Calcul de la MSE
        # ...
        loss = None

        # Rétropropagation
        # ...
        output_error = None
        hidden_error = None

        # Mise à jour des poids
        # ...
        self.weights_hidden_output = None
        self.weights_input_hidden = None

        return loss

    def train(self, X, y_one_hot, epochs=100, learning_rate=0.01):
        for epoch in range(epochs):
            self.forward(X)
            loss = self.backward(X, y_one_hot, learning_rate)
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

    def predict(self, X):
        self.forward(X)
        return np.argmax(self.model_output, axis=1)
    
    def visualize_prediction(self, X, y_true, index):
        input_data = X[index, :].reshape(28, 28)

        predicted_label = self.predict(X[index:index + 1])[0]

        plt.imshow(input_data, cmap='gray')
        plt.title(f"Prediction: {predicted_label}, True Label: {np.argmax(y_true[index])}")
        plt.show()
        
    def confusion_matrix(self, X, y_true):
        y_pred = self.predict(X)

        num_classes = 10
        cm = np.zeros((num_classes, num_classes), dtype=int)

        for true_label, pred_label in zip(np.argmax(y_true, axis=1), y_pred):
            cm[true_label, pred_label] += 1

        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, range(num_classes))
        plt.yticks(tick_marks, range(num_classes))
        plt.xlabel('Predicted')
        plt.ylabel('True')

        for i in range(num_classes):
            for j in range(num_classes):
                plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')

        plt.show()

if __name__=="__main__":
    X, y = load_mnist_data('train.csv')

    input_size = X.shape[1]
    hidden_size = None # ...
    output_size = 10
    e = None # ...
    mu = None # ...

    nn = NeuralNetwork(input_size, hidden_size, output_size)
    nn.train(X, y, epochs=e, learning_rate=mu)

    X_test, y_test = load_mnist_data('test.csv')
    nn.confusion_matrix(X_test, y_test)
    nn.visualize_prediction(X_test, y_test, 10)