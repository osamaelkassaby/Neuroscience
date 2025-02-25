import random

def tanh(x):
    return (2 / (1 + pow(2.718281828459045, -2 * x))) - 1 

w1 = random.uniform(-0.5, 0.5)
w2 = random.uniform(-0.5, 0.5)
w3 = random.uniform(-0.5, 0.5)
w4 = random.uniform(-0.5, 0.5)
w5 = random.uniform(-0.5, 0.5)
w6 = random.uniform(-0.5, 0.5)

b1 = 0.5
b2 = 0.7


x1 = 0.05
x2 = 0.1


h1 = tanh(w1 * x1 + w2 * x2 + b1)
h2 = tanh(w3 * x1 + w4 * x2 + b1)

y = tanh(w5 * h1 + w6 * h2 + b2)

print("Output of the network:", y)
