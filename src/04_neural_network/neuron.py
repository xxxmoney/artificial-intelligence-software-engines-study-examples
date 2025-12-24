import random
from abc import ABC, abstractmethod

#
# Neuron is a simple single unit which reacts to input/inputs with output
# Its a basic unit for building Neural-Networks (these are combinations of many neurons)
# Neuron is composed of:
# - Inputs - information given from outside - like health, distance, etc.
# - Weights - how much is an input important - this is different from systems like FSM or BR - we do not directly define weights - neurons "learn" the weights through learning process
# = Bias - how much should the neuron be sensitive to input
# - Activation Function - this function basically tells us if the neuron is "activated (there can be used many functions which define how the neuron activates)
#   - Step - returns result 0 or 1 (0 is activated, 1 is not activated)
#   - Sigmoid - returns result from 0 to 1 (uses scale from 0 to 1)
#   - ReLu - returns result from 0 to infinity (uses scale from 0 to infinity)
#

LEARNING_RATE = 0.01

class TrainingDataItem:
    inputs: list[float]
    target: float

    def __init__(self, inputs: list[float], target: float):
        self.inputs = inputs
        self.target = target

class Neuron(ABC):
    weights: list[float]
    bias: float

    def __init__(self, num_inputs: int):
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        self.bias = random.uniform(-1, 1)

    @abstractmethod
    def _activate(self, value: float) -> float:
        pass

    def think(self, inputs: list[float]) -> float:
        if len(inputs) != len(self.weights):
            raise ValueError(f"Number of inputs ({len(inputs)}) does not match number of weights ({len(self.weights)})")

        sum_value = 0
        # Calculate value for each input and weight and get the sum
        for input, weight in zip(inputs, self.weights):
            sum_value += input * weight

        # Include bias
        sum_value += self.bias

        return self._activate(sum_value)

    def train(self, training_data: list[TrainingDataItem], iterations: int) -> None:
        print(f"Training neuron with {len(self.weights)} inputs/weights with {iterations} iterations...")

        for _ in range(iterations):
            for item in training_data:
                # Predict output with current weights
                prediction = self.think(item.inputs)
                # Calculate error from desired target
                error = item.target - prediction

                # Calculate new weights using the error and the learning rate
                new_weights = []
                for i in range(len(self.weights)):
                    delta = error * item.inputs[i] * LEARNING_RATE
                    new_weights.append(self.weights[i] + delta)

                self.weights = new_weights

                # Also adjust for bias
                self.bias += error * LEARNING_RATE

        print(f"Training complete")


# Perceptron is a neuron which is alone and uses step function - 0 or 1
class Perceptron(Neuron):
    def __init__(self, num_inputs: int):
        super().__init__(num_inputs)

    def _activate(self, value: float) -> float:
        return 1 if value > 0 else 0


if __name__ == "__main__":
    # Simple example with perceptron usage
    # Inputs:
    # - 1: Do I have money (0,1)
    # - 2: Do I have time (0,1)
    # - 3: Is the movie great? (0,1)

    # In this simple example, we are training the perceptron on this training data:
    # - we care only about input 1 and 2, the third one is irrelevant (we only want to go to the movie if I have money and time - if the movie is nice is irrelevant)
    training_iterations = 500
    training_data = [
        TrainingDataItem([0, 0, 0], 0),  # No money, no time, no movie -> not going
        TrainingDataItem([0, 0, 1], 0),  # No money, no time, nice movie -> not going
        TrainingDataItem([0, 1, 0], 0),  # No money, has time, no movie -> not going
        TrainingDataItem([0, 1, 1], 0),  # No money, has time, nice movie -> not going
        TrainingDataItem([1, 0, 0], 0),  # Has money, no time, no movie -> not going
        TrainingDataItem([1, 0, 1], 0),  # Has money, no time, nice movie -> not going
        TrainingDataItem([1, 1, 0], 1),  # Has money, has time, no movie -> going
        TrainingDataItem([1, 1, 1], 1),  # Has money, has time, nice movie -> going
    ]
    perceptron = Perceptron(3)

    # Try default random weights on some cases:
    print("[Testing random perceptron]")
    print("0 money, 0 time, 0 nice movie: ", perceptron.think([0, 0, 0]))
    print("0 money, 1 time, 0 nice movie: ", perceptron.think([0, 1, 0]))
    print("1 money, 0 time, 0 nice movie: ", perceptron.think([1, 0, 0]))
    print("1 money, 1 time, 0 nice movie: ", perceptron.think([1, 1, 0]))
    print("0 money, 0 time, 1 nice movie: ", perceptron.think([0, 0, 1]))
    print("0 money, 1 time, 1 nice movie: ", perceptron.think([0, 1, 1]))
    print("1 money, 0 time, 1 nice movie: ", perceptron.think([1, 0, 1]))
    print("1 money, 1 time, 1 nice movie: ", perceptron.think([1, 1, 1]))

    print(f"[Training perceptron for {training_iterations} iterations]")
    perceptron.train(training_data, training_iterations)

    # Try trained weights on some cases:
    print(f"[Testing trained perceptron for {training_iterations} iterations]")
    print("0 money, 0 time, 0 nice movie: ", perceptron.think([0, 0, 0]))
    print("0 money, 1 time, 0 nice movie: ", perceptron.think([0, 1, 0]))
    print("1 money, 0 time, 0 nice movie: ", perceptron.think([1, 0, 0]))
    print("1 money, 1 time, 0 nice movie: ", perceptron.think([1, 1, 0]))
    print("0 money, 0 time, 1 nice movie: ", perceptron.think([0, 0, 1]))
    print("0 money, 1 time, 1 nice movie: ", perceptron.think([0, 1, 1]))
    print("1 money, 0 time, 1 nice movie: ", perceptron.think([1, 0, 1]))
    print("1 money, 1 time, 1 nice movie: ", perceptron.think([1, 1, 1]))

