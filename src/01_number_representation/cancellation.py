#
# Cancellation causes loosing the precision
# It happens when we have very big numbers, close to each other - with subtraction the "small" part disappears due to precision issues
#


def subtract(number_one: float, number_two: float):
    result = number_one - number_two

    print(result)

print("Can be represented correctly:")
subtract(10.5, 10)

print("Still represented correctly:")
subtract(1000000.5, 1000000)

print("Loosing precision:")
subtract(1000000.0000000000005, 1000000)
