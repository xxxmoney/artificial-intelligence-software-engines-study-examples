import copy
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

#
# Neural networks are essentially interconnected networks of neurons
# Key feature of neural networks in backpropagation - its a way of learning from errors
# - while when having a single neuron, its training was rather simple - it just took its error from result and adjusted weights
# - now for thw neural network its more advanced - each neuron has to be capable to tell the delta (the "margin" from error)
# - this is then used throughout the network
#   - the process start from the start - it gets input, goes through the network, and to the output (in our case 1 output neuron)
#   - the error from the end has to go through the whole network - the backpropagation - each neuron has to adjust its own weights (thus why we need the delta calculation for each neuron)
# We have an "orchestrator" for the neurons - the Neural Network
# The overall neural network has benefits over single neuron - for example, neural network is capable of being trained to "think" about XOR operations (not possible with only singular neuron)
# Graphical representation:
# DATA (Input)        HIDDEN LAYER/LAYERS           OUTPUT LAYER      TARGET
#    [0.5] -----------> (Neuron/Neurons) -----------> (Neuron) ---------> [1.0]
#      |                   |                     |
#      |                   |                     |
# (just number)     (has weights and bias)   (has weights and bias)     (just number)
# (does not change)  (Learns from next layer)  (learns from target)     (does not change)
#

LEARNING_RATE = 0.01

@dataclass
class TrainingDataItem:
    inputs: list[float]
    targets: list[float]

class Neuron(ABC):
    weights: list[float]
    bias: float
    next_layer_neurons: list[float]
    last_delta: Optional[float]
    last_inputs: Optional[list[float]]
    last_output: Optional[float]

    def __init__(self, num_inputs: int):
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        self.bias = random.uniform(-1, 1)
        self.last_delta = None
        self.last_inputs = None
        self.last_output = None

    @abstractmethod
    def _activate(self, value: float) -> float:
        pass

    @abstractmethod
    def _derivative(self, value: float) -> float:
        pass

    def calculate_output_delta(self, target: float):
        # Calculates delta based on target directly - for the output layer
        error = target - self.last_output
        self.last_delta = error * self._derivative(self.last_output)

    def calculate_hidden_delta(self, next_layer_neurons: list['Neuron'], my_index: int):
        # Calculates delta based on the next layer
        error_sum = 0.0
        for neuron in next_layer_neurons:
            error_sum += neuron.last_delta * neuron.weights[my_index]

        self.last_delta = error_sum * self._derivative(self.last_output)

    def think(self, inputs: list[float]) -> float:
        if len(inputs) != len(self.weights):
            raise ValueError(f"Number of inputs ({len(inputs)}) does not match number of weights ({len(self.weights)})")

        sum_value = 0
        # Calculate value for each input and weight and get the sum
        for input, weight in zip(inputs, self.weights):
            sum_value += input * weight

        # Include bias
        sum_value += self.bias

        output = self._activate(sum_value)

        # Preserve last inputs and output for training
        self.last_inputs = copy.copy(inputs) # List - copy, we don't want to preserve the reference
        self.last_output = output # Float - we don't need to copy

        return output

    def update_weights(self):
        # Adjust weights based on last delta
        for i in range(len(self.weights)):
            change = self.last_inputs[i] * self.last_delta * LEARNING_RATE
            self.weights[i] += change

        # And also adjust bias
        self.bias += self.last_delta * LEARNING_RATE

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: weights: {self.weights}, bias: {self.bias}"


class SigmoidNeuron(Neuron):
    def _activate(self, value: float) -> float:
        # Sigmoid: 1 / (1 + e^-x)
        # Handles overflow for very large negative numbers
        try:
            return 1 / (1 + math.exp(-value))
        except OverflowError:
            return 0 if value < 0 else 1

    def _derivative(self, value: float) -> float:
        # Derivative of Sigmoid is: Sigmoid(x) * (1 - Sigmoid(x))
        # Since we already have the output (which IS Sigmoid(x)), it's very fast:
        return value * (1 - value)


class NeuralNetwork:
    layers: list[list[Neuron]]

    def __init__(self, layers_structure: list[int]):
        """
        Creates neural network based on given structure
        Example: [2, 2, 1]:
        - 2 inputs
        - 2 neurons in hidden layer
        - 1 neuron in output layer
        """
        self.layers = []

        # Creating layers - starting from index 1 (first value is just number of inputs)
        for i in range(1, len(layers_structure)):
            num_neurons = layers_structure[i]
            num_inputs = layers_structure[i - 1]

            layer = [SigmoidNeuron(num_inputs) for _ in range(num_neurons)]
            self.layers.append(layer)

    def think(self, inputs: list[float]) -> list[float]:
        current_inputs = inputs # First output is taken as input

        # Goes through layer by layer - basically each layer takes input from last output, gets output and that gets used as input for next (or output)
        for layer in self.layers:
            current_inputs = [neuron.think(current_inputs) for neuron in layer]

        return current_inputs

    def train(self, training_data: list[TrainingDataItem], iteration_count: int) -> None:
        print(f"[Network before training]")
        print(self)

        print(f"[Untrained inputs and outputs]")
        for item in training_data:
            print(f"Inputs: {item.inputs}, Outputs: {self.think(item.inputs)}, Expected Outputs: {item.targets}")

        for _ in range(iteration_count):
            item = random.choice(training_data)
            self.train_item(item)

        print("\n")

        print(f"[Network after training]")
        print(self)

        print(f"[Trained inputs and outputs]")
        for item in training_data:
            print(f"Inputs: {item.inputs}, Outputs: {self.think(item.inputs)}, Expected Outputs: {item.targets}")
        print("\n")

    def train_item(self, item: TrainingDataItem) -> None:
        # Let all inputs think - to set initial values
        self.think(item.inputs)

        # Calculate deltas for output layer
        output_layer = self.layers[-1]
        for i, neuron in enumerate(output_layer):
            neuron.calculate_output_delta(item.targets[i]) # This is output layer - we calculate output delta

        # Go through layers and calculate deltas based on next layer
        for i in reversed(range(len(self.layers) - 1)):
            current_layer = self.layers[i]
            next_layer = self.layers[i + 1]

            for j, neuron in enumerate(current_layer):
                neuron.calculate_hidden_delta(next_layer, j) # This is hidden layer - we calculate hidden delta

        # Now we have calculated deltas for all neurons, we can update weights
        for layer in self.layers:
            for neuron in layer:
                neuron.update_weights()

    def __str__(self) -> str:
        return f"\n{"".ljust(250, "-")}\n".join(["\n".join([neuron.__str__() for neuron in layer]) for layer in self.layers])

if __name__ == "__main__":
    def train(network: NeuralNetwork, training_data: list[TrainingDataItem], iteration_count: int) -> None:
        print(f"[[Network will be trained for {iteration_count} iterations:]]")
        print("\n")
        network.train(training_data, iteration_count)
        print("\n")

    # Simple example
    # Inputs:
    # - 1: Do I have money (0,1)
    # - 2: Do I have time (0,1)
    # - 3: Is the movie great? (0,1)
    # We care only about input 1 and 2, the third one is irrelevant (we only want to go to the movie if I have money and time - if the movie is nice is irrelevant)
    print("[[[Example 01:]]]")
    print("\n")
    training_data = [
        TrainingDataItem([0, 0, 0], [0]),  # No money, no time, no movie -> not going
        TrainingDataItem([0, 0, 1], [0]),  # No money, no time, nice movie -> not going
        TrainingDataItem([0, 1, 0], [0]),  # No money, has time, no movie -> not going
        TrainingDataItem([0, 1, 1], [0]),  # No money, has time, nice movie -> not going
        TrainingDataItem([1, 0, 0], [0]),  # Has money, no time, no movie -> not going
        TrainingDataItem([1, 0, 1], [0]),  # Has money, no time, nice movie -> not going
        TrainingDataItem([1, 1, 0], [1]),  # Has money, has time, no movie -> going
        TrainingDataItem([1, 1, 1], [1]),  # Has money, has time, nice movie -> going
    ]
    network = NeuralNetwork([3, 2, 3, 1]) # Testing with some number of hidden layers (just random, no deeper meaning - just have to define "first layer" as three inputs and output as 1 output)
    train(network, training_data, 250000)



