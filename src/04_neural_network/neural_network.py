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

    def update_weights(self, learning_rate: float):
        # Adjust weights based on last delta
        for i in range(len(self.weights)):
            change = self.last_inputs[i] * self.last_delta * learning_rate
            self.weights[i] += change

        # And also adjust bias
        self.bias += self.last_delta * learning_rate

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
    learning_rate: float

    def __init__(self, layers_structure: list[int], learning_rate: float):
        """
        Creates neural network based on given structure
        Example: [2, 2, 1]:
        - 2 inputs
        - 2 neurons in hidden layer
        - 1 neuron in output layer
        """
        self.layers = []
        self.learning_rate = learning_rate

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
                neuron.update_weights(self.learning_rate)

    def __str__(self) -> str:
        return f"\n{"".ljust(250, "-")}\n".join(["\n".join([neuron.__str__() for neuron in layer]) for layer in self.layers])

if __name__ == "__main__":
    def train(network: NeuralNetwork, training_data: list[TrainingDataItem], iteration_count: int) -> None:
        print(f"[[Network will be trained for {iteration_count} iterations:]]")
        print("\n")
        network.train(training_data, iteration_count)
        print("\n")

    def example_01():
        # Simple example 01 - Movie
        # Inputs:
        # - 1: Do I have money (0,1)
        # - 2: Do I have time (0,1)
        # - 3: Is the movie great? (0,1)
        # Outputs:
        # - 1: whether we go to see the movie or not
        print("[[[Example 01 - Movie]]]")
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
        network = NeuralNetwork([3, 2, 3, 1], 0.01)  # Testing with some number of hidden layers (just random, no deeper meaning - just have to define "first layer" as three inputs and output as 1 output)
        train(network, training_data, 250000)


    def example_02():
        # Simple example 02 - XOR
        # XOR works as AND of OR and NAND (not AND) - in simpler terms, when the logical values are different, XOR is true - exclusive OR
        # Funny thing about XOR is that it does not work with 1 neuron - meaning it needs network with at least 2 neurons in the hidden layer
        # Inputs:
        # - 1: Logical first - 0 or 1
        # - 2: Logical second - 0 or 1
        # Outputs:
        # - 1: Whether the XOR is true - 1 or false - 0 (basically an and)

        learning_rate = 0.5 # We can define a bit higher learning rate

        print("[[[Example 02_01 - XOR - 1 neuron in hidden layer - won't work]]]")
        print("\n")
        training_data = [
            TrainingDataItem([0, 0], [0]),
            TrainingDataItem([0, 1], [1]),
            TrainingDataItem([1, 0], [1]),
            TrainingDataItem([1, 1], [0]),
        ]
        network = NeuralNetwork([2, 1, 1], learning_rate) # 2 inputs, 1 neuron in hidden layer, 1 output
        train(network, training_data, 1000000) # Even with an extreme amount of iterations, network with just 1 neuron in hidden layer cannot do it

        print("[[[Example 02_02 - XOR - 2 neurons in hidden layer - will work]]]")
        print("\n")
        training_data = [
            TrainingDataItem([0, 0], [0]),
            TrainingDataItem([0, 1], [1]),
            TrainingDataItem([1, 0], [1]),
            TrainingDataItem([1, 1], [0]),
        ]
        network = NeuralNetwork([2, 2, 1], learning_rate) # 2 inputs, 2 neurons in hidden layer, 1 output
        train(network, training_data, 500000) # Voila - when we have 2 neurons in hidden layer, suddenly, the network is capable of learning the XOR


    def example_03():
        # Simple example 01 - Clothing
        # I necessary have to have left and right boot on, otherwise I cant go outside, otherwise, when its cold, I should have pants, jacket and cap on, if its dazzling, I should have glasses on
        # Inputs:
        # - 1: Do I have left boot on?
        # - 2: Do I have right boot on?
        # - 3: Do I have pants on?
        # - 4: Do I have cap oo?
        # - 5: Do I have glasses on?
        # - 6: Do I have jacket on?
        # - 7: Is it cold outside?
        # - 8: Is it hot outside?
        # - 9: Is it dazzling outside?
        # Outputs:
        # - 1: Whether I can go out or not
        print("[[[Example 03 - Clothing]]]")
        print("\n")

        # Logic:
        # 0:L_Boot, 1:R_Boot, 2:Pants, 3:Cap, 4:Glasses, 5:Jacket, 6:Cold, 7:Hot, 8:Dazzling
        # Rules:
        # - Must have L_Boot(0) & R_Boot(1)
        # - If Cold(6): Must have Pants(2) & Cap(3) & Jacket(5)
        # - If Dazzling(8): Must have Glasses(4)
        training_data = [
            # --- Basic Failures (Missing Boots) ---
            TrainingDataItem([0, 1, 1, 1, 1, 1, 0, 0, 0], [0]),  # Missing Left Boot
            TrainingDataItem([1, 0, 1, 1, 1, 1, 0, 0, 0], [0]),  # Missing Right Boot
            TrainingDataItem([0, 0, 1, 1, 1, 1, 1, 0, 1], [0]),  # Missing Both Boots
            TrainingDataItem([0, 1, 0, 0, 0, 0, 1, 0, 0], [0]),  # Missing L Boot (Cold)
            TrainingDataItem([1, 0, 0, 0, 1, 0, 0, 0, 1], [0]),  # Missing R Boot (Dazzle)

            # --- Cold Failures (Have Boots + Cold, Missing Gear) ---
            TrainingDataItem([1, 1, 0, 1, 0, 1, 1, 0, 0], [0]),  # Cold, No Pants
            TrainingDataItem([1, 1, 1, 0, 0, 1, 1, 0, 0], [0]),  # Cold, No Cap
            TrainingDataItem([1, 1, 1, 1, 0, 0, 1, 0, 0], [0]),  # Cold, No Jacket
            TrainingDataItem([1, 1, 0, 0, 0, 0, 1, 0, 0], [0]),  # Cold, Nothing on
            TrainingDataItem([1, 1, 1, 1, 1, 0, 1, 0, 1], [0]),  # Cold+Dazzle, No Jacket

            # --- Dazzle Failures (Have Boots + Dazzle, Missing Glasses) ---
            TrainingDataItem([1, 1, 1, 1, 0, 1, 0, 0, 1], [0]),  # Dazzle, No Glasses
            TrainingDataItem([1, 1, 0, 0, 0, 0, 0, 1, 1], [0]),  # Dazzle+Hot, No Glasses
            TrainingDataItem([1, 1, 1, 1, 0, 1, 1, 0, 1], [0]),  # Cold+Dazzle, No Glasses (Gear OK)

            # --- Success Cases (Neutral Weather) ---
            TrainingDataItem([1, 1, 0, 0, 0, 0, 0, 0, 0], [1]),  # Just Boots
            TrainingDataItem([1, 1, 1, 1, 1, 1, 0, 0, 0], [1]),  # Boots + Everything
            TrainingDataItem([1, 1, 0, 0, 0, 0, 0, 1, 0], [1]),  # Boots + Hot
            TrainingDataItem([1, 1, 1, 0, 1, 0, 0, 1, 0], [1]),  # Random clothes OK

            # --- Success Cases (Cold Weather - Needs 2,3,5) ---
            TrainingDataItem([1, 1, 1, 1, 0, 1, 1, 0, 0], [1]),  # Cold + All Gear
            TrainingDataItem([1, 1, 1, 1, 1, 1, 1, 0, 0], [1]),  # Cold + All Gear + Glasses
            TrainingDataItem([1, 1, 1, 1, 0, 1, 1, 0, 0], [1]),  # Cold + Min Gear

            # --- Success Cases (Dazzling - Needs 4) ---
            TrainingDataItem([1, 1, 0, 0, 1, 0, 0, 0, 1], [1]),  # Dazzle + Glasses
            TrainingDataItem([1, 1, 1, 1, 1, 1, 0, 0, 1], [1]),  # Dazzle + Glasses + Extra

            # --- Success Cases (Cold + Dazzling - Needs 2,3,4,5) ---
            TrainingDataItem([1, 1, 1, 1, 1, 1, 1, 0, 1], [1]),  # All conditions met

            # --- Bulk Data (Generated Combinations) ---
            TrainingDataItem([1, 1, 0, 1, 0, 1, 0, 1, 0], [1]),  # Hot, no cold/dazzle rules
            TrainingDataItem([0, 1, 1, 1, 1, 1, 1, 0, 1], [0]),  # Full gear but missing L boot
            TrainingDataItem([1, 1, 0, 0, 0, 0, 1, 0, 0], [0]),  # Cold, just boots
            TrainingDataItem([1, 1, 0, 0, 1, 0, 0, 1, 1], [1]),  # Hot+Dazzle+Glasses
            TrainingDataItem([1, 1, 1, 1, 0, 1, 0, 0, 0], [1]),  # Random mix, weather nice
            TrainingDataItem([1, 1, 1, 0, 0, 1, 0, 0, 0], [1]),  # Random mix, weather nice
            TrainingDataItem([1, 1, 0, 1, 0, 0, 0, 0, 0], [1]),  # Random mix, weather nice
            TrainingDataItem([1, 1, 0, 0, 0, 0, 0, 0, 1], [0]),  # Dazzle, no glasses
            TrainingDataItem([1, 1, 0, 0, 1, 0, 0, 0, 0], [1]),  # Glasses, no dazzle
            TrainingDataItem([1, 1, 1, 1, 0, 0, 1, 0, 0], [0]),  # Cold, missing jacket
            TrainingDataItem([1, 1, 0, 1, 0, 1, 1, 0, 0], [0]),  # Cold, missing pants
            TrainingDataItem([1, 1, 1, 0, 0, 1, 1, 0, 0], [0]),  # Cold, missing cap
            TrainingDataItem([1, 1, 1, 1, 1, 1, 1, 1, 1], [1]),  # Everything on/active (Passes all constraints)
            TrainingDataItem([0, 0, 0, 0, 0, 0, 0, 0, 0], [0]),  # Naked at home
            TrainingDataItem([1, 1, 0, 0, 0, 0, 0, 0, 0], [1]),  # Boots only, nice day
            TrainingDataItem([1, 1, 0, 0, 0, 0, 0, 1, 0], [1]),  # Boots only, hot day
            TrainingDataItem([1, 1, 0, 0, 0, 0, 1, 0, 0], [0]),  # Boots only, cold day
            TrainingDataItem([1, 1, 1, 1, 1, 1, 0, 1, 0], [1]),  # Full gear, hot day
            TrainingDataItem([1, 1, 0, 0, 1, 0, 0, 0, 1], [1]),  # Boots+Glasses, Dazzle
            TrainingDataItem([1, 1, 0, 0, 0, 0, 0, 0, 1], [0]),  # Boots, Dazzle, No Glasses
            TrainingDataItem([1, 0, 1, 1, 1, 1, 1, 0, 1], [0]),  # Perfect gear, missing R boot
        ]
        network = NeuralNetwork([9, 10, 10, 1], 0.5)
        train(network, training_data, 500000)

        # 0:L_Boot, 1:R_Boot, 2:Pants, 3:Cap, 4:Glasses, 5:Jacket, 6:Cold, 7:Hot, 8:Dazzling
        # Some testing
        print("[[Testing some examples:]]")
        inputs = [
            ["Everything except right foot and it's not cold", [1, 0, 1, 1, 1, 1, 0, 1, 1]],
            ["I have my shoes on except nothing else and its cold", [1, 1, 0, 0, 0, 0, 1, 0, 0]]
        ]
        for text, values in inputs:
            print(text, values, f": {network.think(values)}")


    # example_01()
    # example_02()
    example_03()

