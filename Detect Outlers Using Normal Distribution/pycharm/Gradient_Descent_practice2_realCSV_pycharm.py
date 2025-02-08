import numpy as np


def gradient_descent(x, y, learning_rate=0.01, epochs=10000):
    m, b = 0.0, 0.0
    for epoch in range(epochs):
        y_pred = m*x + b
        error = y - y_pred
        cost = np.mean(error**2)

        dm = -2*np.mean(error*x)
        db = -2*np.mean(error)

        b -= db*learning_rate
        m -= dm*learning_rate

        print(f"m:{m}, b:{b}, Epoch:{epoch}, cost:{cost}")
        print("\n")


if __name__ == "__main__":
    x = np.array([1,2,3,4,5])
    y = np.array([5,7,9,11,13])
    gradient_descent(x, y)


