import numpy as np
u = np.array([1, 2, 3])
print(u *2)

v = np.array([4, 5, 6])
print(u + v)
print(u * v)

def vector_vector_multiply(u, v):
    return np.dot(u, v)
p