import numpy as np
import scipy.integrate as integrate
import scipy.constants as constant
from random import randint
import matplotlib.pyplot as plt


# пункт 1
print("1)\n")
a = np.ones((10, 10), float)
print(a, end='\n\n')
b = np.identity(10)
print(b, end='\n\n')


# пункт 2
print("2)\n")
c = np.array([[2, 1, 3, 6], [4, 1, 3, 3], [5, 2, 4, 1], [5, 1, 2, 2]])
print(c, end='\n\n')
print("Определитель:", np.linalg.det(c), end = '\n\n')


# пункт 3
print("3)")
A = np.array([[randint(-3, 5) for i in range(4)] for i in range(4)])
B = np.array([[randint(-3, 5)] for i in range(4)])
print("A\n", A)
print("B\n", B)
print("X\n", np.linalg.solve(A, B), end='\n\n')


# пункт 4
print("4)")
def f1(x: float) -> float:
    return 1/np.sqrt(1 - x**2)
res = integrate.quad(f1, -1/2, 1/2)
print("integral = {0} +- {1}".format(res[0], res[1]), end='\n\n')


# пункт 5
print("5)")
def f2(x):
    return np.cos(2*x)
res = integrate.quad(f2, 0, np.inf)
print("integral = {0} +- {1}".format(res[0], res[1]), end='\n\n')


# пункт 6
x1 = np.arange(-10, 10, 0.01)
x2 = np.arange(-5, 5, 1)
def f3(x):
    return np.sin(x + np.pi/3)
def f4(x):
    return x * 2
y1 = f3(x1)
y2 = f4(x2)

plt.figure(figsize=(10, 10))
plt.xlabel("X")
plt.ylabel("Y")
plt.plot(x1, y1, "r-", label=r"$y = \sin(x+\frac{\pi}{3})$")
plt.plot(x2, y2, "b-", label="y = 2x")
plt.legend()
plt.savefig("lab1_plot.png")