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

def calculate_conditional_entropy_from_join(joint_probabilities: list[list[float]]) -> float:
    """
    Calculates join probability from joint matrix

    Calculates H(X|Y) using the relation: H(X|Y) = H(X,Y) - H(Y)

    :param joint_probabilities: Matrix of probabilities (row x_1 probabilities when y_1, row x_2 probabilities when y_2...)
    :return:
    """

    # Calculate entropy for whole table, so H(X, Y)
    joint_entropy = 0
    for probabilities in joint_probabilities:
        joint_entropy += calculate_entropy(probabilities)

    # Calculate (marginal) y probabilities (basically just sum of each whole column)
    y_probabilities = [0] * len(joint_probabilities[0]) # For starters just 0s in length of each row
    for row in joint_probabilities:
        for i, probability in enumerate(row):
            y_probabilities[i] += probability

    # Now we can get the entropy for marginal y
    y_entropy = calculate_entropy(y_probabilities)

    # Result is the total entropy minus entropy of y
    return joint_entropy - y_entropy

def calculate_mutual_information(x_entropy, x_given_y_conditional_entropy) -> float:
    """
    How much entropy was "removed" by knowing x given y
    In other words - how much uncertainty is gone due to knowing x given y

    Calculates I(X;Y) = H(X) - H(X|Y)

    :param x_entropy: Entropy of x
    :param x_given_y_conditional_entropy: Entropy of x given y (conditional)
    :return: Mutual information
    """

    return x_entropy - x_given_y_conditional_entropy


def calculate_cross_entropy(real_distribution, model_distribution):
    """
    Calculates H(P, Q) = - sum( p(x) * log2(q(x)) )

    real_distribution: True distribution (Reality)
    model_distribution: Predicted distribution (Model)
    """

    negative_cross_entropy = 0

    for i in range(len(real_distribution)):
        p = real_distribution[i]
        q = model_distribution[i]

        # Avoid log(0) in model if reality expects it
        if q == 0 and p > 0:
            # Infinite penalty for impossible event happening
            return float('inf')

        term = p * math.log2(q)
        negative_cross_entropy += term

    return -negative_cross_entropy


def calculate_kl_divergence(real_distribution, model_distribution):
    """
    How much error does the model have against reality

    Calculates D_KL(P||Q) = sum( p(x) * log2(p(x) / q(x)) )

    Alternatively: CrossEntropy(P,Q) - Entropy(P)
    
    :param real_distribution:
    :param model_distribution:
    :return:
    """
    real_entropy = calculate_entropy(real_distribution)
    cross_entropy = calculate_cross_entropy(real_distribution, model_distribution)

    return cross_entropy - real_entropy

def print_entropy(type: str, title: str, probabilities: list[list[float]], entropy: float, description: Optional[str] = None) -> None:
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

        print_entropy("normal", "Two coins same chance", [probabilities], entropy, "as the coins have the same chance, the entropy is maximum - because we are maximally unsure what will happen")

    def entropy_unfair_coin_example():
        probabilities = [0.9, 0.1] # Two sides of coin - but one has much greater chance than other
        entropy = calculate_entropy(probabilities)

        print_entropy("normal", "Two coins one much greater chance", [probabilities], entropy, "as one coin has much greater chance, the entropy is lower - because are more sure about one side happening")


    def conditional_entropy_forecast_example():
        forecast_probabilities = [0.5, 0.5] # 50/50 chance for Rain/Sun
        weather_given_forecast_probabilities = [
            [0.9, 0.1], # If Forecast is Rain - chances for real Weather - Rain/Sun
            [0.2, 0.8], # If Forecast is Sun - chances for real Weather - Rain/Sun
        ]

        entropy = calculate_conditional_entropy(forecast_probabilities, weather_given_forecast_probabilities)

        print_entropy("conditional", "Forecast says Rain/Sun VS real Weather Rain/Sun", [forecast_probabilities], entropy, "we have probabilities of Forecast Rain/Sun, we then have probabilities of real Weather based on Forecast")

    def join_conditional_entropy_forecast_example():
        forecast_weather_join_probabilities = [
            [0.45, 0.05], # If Forecast is Rain - chances for real Weather - Rain/Sun
            [0.10, 0.40]  # If Forecast is Sun - chances for real Weather - Rain/Sun
        ]

        entropy = calculate_conditional_entropy_from_join(forecast_weather_join_probabilities)

        print_entropy("conditional from joint", "Forecast says Rain/Sun VS real Weather Rain/Sun", forecast_weather_join_probabilities, entropy, "we have probabilities of real Weather Rain/Sun - given this, Forecast probabilities are calculated and thus the conditional entropy")


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

    #
    # Joint conditional entropy examples:
    #
    join_conditional_entropy_forecast_example()
