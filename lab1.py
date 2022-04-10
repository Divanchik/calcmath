import numpy as np
from random import randint
# матрица из вещественных единиц
a = np.ones((10, 10), float)
print(a, end='\n\n')
# единичная матрица
b = np.identity(10)
print(b, end='\n\n')
# вычислить определитель
c = np.array([[2, 1, 3, 6], [4, 1, 3, 3], [5, 2, 4, 1], [5, 1, 2, 2]])
print(c)
print("определитель:", np.linalg.det(c), end = '\n\n')
# 4
A = np.array([[randint(-3, 5) for i in range(4)] for i in range(4)])
B = np.array([[randint(-3, 5)] for i in range(4)])
print("A\n", A)
print("B\n", B)
print("X\n", np.linalg.solve(A, B))
