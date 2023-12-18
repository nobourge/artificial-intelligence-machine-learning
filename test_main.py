import numpy as np
import pytest
from main_completed import NeuralNetwork 

@pytest.fixture
def neural_network():
    return NeuralNetwork(input_size=4, hidden_size=3, output_size=2)

@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = np.random.rand(10, 4)
    y_one_hot = np.eye(2)[np.random.choice(2, 10)]
    return X, y_one_hot

def test_tanh():
    nn = NeuralNetwork(1, 1, 1)
    assert nn.tanh(0) == 0.0

def test_softmax():
    nn = NeuralNetwork(1, 1, 2)
    assert np.allclose(nn.softmax(np.array([[1, 2]])), np.array([[0.26894142, 0.73105858]]))

def test_mse_loss():
    nn = NeuralNetwork(1, 1, 1)
    assert nn.mse_loss(np.array([1, 1]), np.array([0.5, 0.5])) == 0.25

def test_forward(neural_network, sample_data):
    X, _ = sample_data
    neural_network.forward(X)
    assert neural_network.model_output.shape == (10, 2)

def test_backward(neural_network, sample_data):
    X, y_one_hot = sample_data
    neural_network.forward(X)
    loss = neural_network.backward(X, y_one_hot)
    assert isinstance(loss, float)

def test_predict(neural_network, sample_data):
    X, _ = sample_data
    predictions = neural_network.predict(X)
    assert predictions.shape == (10,)