import numpy as np

data = np.array([[1, 2, 3, 5], [1, 6, 3, 5]])

print(data.argmax() // data.shape[1])
print(data.argmax() % data.shape[1])
