import math
from typing import Optional


#
# The basic unit of information is bit
# Entropy is the uncertainty - basically how unsure we are of the outcome - how much information do we need to store the uncertainty
# - It is store with bits - so we can for example say that the entropy is 2 bit
#

def calculate_entropy(chances: list[float]) -> float:
    negative_entropy = 0

    for chance in chances:
        # How unexpected is the outcome of this chance
        surprise = math.log2(chance)
        # Sum up the chance by its surprise
        negative_entropy += chance * surprise

    # negation - we want to have bigger positive numbers of more unexpected (the negation is because the log returns negative numbers for lower then 0 - and because chances operate on scale of 0 to 1, we need this)
    return -negative_entropy

def print_entropy(title: str, chances: list[float], entropy: float, description: Optional[str] = None) -> None:
    print(f"{title} ({chances}): {entropy}")
    if description is not None:
        print(f"\t - {description}\n")


# TODO: implement other entropies!

if __name__ == "__main__":
    #
    # Simple examples with entropy calculation
    #

    def fair_coin_example():
        # Two sides of coin - each has same chance
        chances = [0.5, 0.5]
        entropy = calculate_entropy(chances)

        print_entropy("Two coins same chance", chances, entropy, "as the coins have the same chance, the entropy is maximum - because we are maximally unsure what will happen")

    def unfair_coin_example():
        # Two sides of coin - but one has much greater chance than other
        chances = [0.9, 0.1]
        entropy = calculate_entropy(chances)

        print_entropy("Two coins one much greater chance", chances, entropy, "as one coin has much greater chance, the entropy is lower - because are more sure about one side happening")


    #
    # Examples:
    #

    fair_coin_example()
    unfair_coin_example()