import timeit
import numpy as np


import timeit

def test_matmul(mat1, mat2):
    return mat1.reshape(mat1.shape[0], -1) @ mat2.flatten()

def test_mul1(mat1, mat2):
    out = mat1.reshape(mat1.shape[0], -1) * mat2.flatten().reshape(1, -1)
    return out.sum(axis=1)

def test_mul2(mat1, mat2):
    out = mat1 * mat2.reshape(1, mat2.shape[0], mat2.shape[1])

    return out.sum(axis=1).sum(axis=1)

N = 500
F = 100
K = 15

mat1 = np.random.uniform(0, 1, size=(N, F, K))
mat2 = np.random.uniform(0, 1, size=(F, K))

test_matmul(mat1, mat2)

loop = 100

result1 = timeit.timeit('test_matmul(mat1, mat2)', globals=globals(), number=loop)
result2 = timeit.timeit('test_mul1(mat1, mat2)', globals=globals(), number=loop)
result3 = timeit.timeit('test_mul2(mat1, mat2)', globals=globals(), number=loop)
print(result1 / loop)
print(result2 / loop)
print(result3 / loop)