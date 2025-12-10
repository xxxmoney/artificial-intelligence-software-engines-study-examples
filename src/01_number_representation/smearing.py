
#
# Smearing causes loosing the precision
# It happens when we have one big and one very small numbers, and we are adding them - the bigger number is not affected as we expected
#

def add(number_one: float, number_two: float):
    result = number_one + number_two

    print(result)


print("Does not cause issues:")
add(100000, 100)

print("Causes issues - we have big number and very small number:")
add(100000, 0.0000000000005)
