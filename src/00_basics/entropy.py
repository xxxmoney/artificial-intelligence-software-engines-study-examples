import math
from typing import Optional


#
# The basic unit of information is bit
# Entropy is the uncertainty - basically how unsure we are of the outcome - how much information do we need to store the uncertainty
# - It is store with bits - so we can for example say that the entropy is 2 bit
#

def calculate_entropy(probabilities: list[float]) -> float:
    """
    Calculates the entropy of a list of probabilities

    Calculates H(X) = - sum(p(x) * log2(p(x)))

    :param probabilities: The distribution of probabilities
    :return: Entropy
    """
    negative_entropy = 0

    for probability in probabilities:
        if probability < 0 or probability > 1:
            raise ValueError(f"Probability must be between 0 and 1, got {probability}")

        # How unexpected is the outcome of this chance
        surprise = math.log2(probability)
        # Sum up the chance by its surprise
        negative_entropy += probability * surprise

    # negation - we want to have bigger positive numbers of more unexpected (the negation is because the log returns negative numbers for lower then 0 - and because chances operate on scale of 0 to 1, we need this)
    return -negative_entropy

def calculate_conditional_entropy(y_probabilities: list[float], x_given_y_probabilities: list[list[float]]) -> float:
    """
    Calculates conditional entropy - in case when x is given by y

    Calculates H(X|Y) = sum( p(y) * H(X|Y=y) )

    :param y_probabilities: Distribution of y probabilities (e.g., Forecast says Rain/Sun)
    :param x_given_y_probabilities: Distributions of x given y probabilities (e.g., Probabilities when forecast says Rain, probabilities when forecast say Sun)
    :return: Conditional entropy
    """

    entropy = 0

    for i, y_probability in enumerate(y_probabilities):
        # Distribution probabilities given y_[i] - so basically x probabilities if y happens
        x_given_y_probability = x_given_y_probabilities[i]

        # Entropy for the x given y
        x_given_y_entropy = calculate_entropy(x_given_y_probability)

        # Overall entropy
        entropy += y_probability * x_given_y_entropy

    return entropy


def print_entropy(type: str, title: str, probabilities: list[float], entropy: float, description: Optional[str] = None) -> None:
    print(f"[[{str.upper(type)}]]\n[{title}] ({probabilities}): {entropy}")
    if description is not None:
        print(f"\t - {description}\n")


if __name__ == "__main__":
    #
    # Simple examples with entropy calculation
    #

    def entropy_fair_coin_example():
        probabilities = [0.5, 0.5] # Two sides of coin - each has same chance
        entropy = calculate_entropy(probabilities)

        print_entropy("normal", "Two coins same chance", probabilities, entropy, "as the coins have the same chance, the entropy is maximum - because we are maximally unsure what will happen")

    def entropy_unfair_coin_example():
        probabilities = [0.9, 0.1] # Two sides of coin - but one has much greater chance than other
        entropy = calculate_entropy(probabilities)

        print_entropy("normal", "Two coins one much greater chance", probabilities, entropy, "as one coin has much greater chance, the entropy is lower - because are more sure about one side happening")


    def conditional_entropy_forecast_example():
        forecast_probabilities = [0.5, 0.5] # 50/50 chance for Rain/Sun
        weather_given_forecast_probabilities = [
            [0.9, 0.1], # If Forecast is Rain - chances for real Weather
            [0.2, 0.8], # If Forecast is Sun - chances for real Weather
        ]

        entropy = calculate_conditional_entropy(forecast_probabilities, weather_given_forecast_probabilities)

        print_entropy("conditional", "Forecast says Rain/Sun VS real Weather Rain/Sun", forecast_probabilities, entropy, "Forecast can be Rain/Sun - we have probabilities for these in Forecast, given this, what are the chances for real Weather Rain/Sun")


    #
    # EXAMPLES:
    #

    #
    # Simple entropy examples:
    #
    entropy_fair_coin_example()
    entropy_unfair_coin_example()

    #
    # Conditional entropy examples:
    #
    conditional_entropy_forecast_example()
