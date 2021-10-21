import numpy as np

# Sigmoid Function


def nonlin(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


# Input Dataset
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

# Output Dataset
y = np.array([[0],
              [0],
              [1],
              [1]])

# Seed random numbers to make calculations deterministic
np.random.seed(1)

# Initialise weights randomly with mean 0
syn0 = 2 * np.random.random((3, 4)) - 1
syn1 = 2 * np.random.random((4, 1)) - 1

for iter in range(60000):
    # Forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))

    # Error
    l2_error = y - l2

    if iter % 10000 == 0:
        print("Error: " + str(np.mean(np.abs(l2_error))))

    l2_delta = l2_error*nonlin(l2, deriv=True)
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error*nonlin(l1, deriv=True)

    # Weight Updating
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

print(f"Output after training: {l2}")

