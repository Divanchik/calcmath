import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from math import factorial as fac


def Lagrange(x: list, y: list, req: float):
    n = len(x)
    L = 0
    for i in range(n):
        t = y[i]
        t1 = 1
        for j in range(n):
            t1 *= req - x[j] if i != j else 1
        t2 = 1
        for j in range(n):
            t2 *= x[i] - x[j] if i != j else 1
        L += t * t1 / t2
    return L


def LagrangeP(X: list, Y: list):
    n = len(X)
    x = sp.Symbol('x')
    L = 0
    for i in range(n):
        t1 = 1
        for j in range(n):
            t1 *= x - X[j] if i != j else 1
        t2 = 1
        for j in range(n):
            t2 *= X[i] - X[j] if i != j else 1
        L += Y[i] * t1 / t2
    return sp.expand(L)


def Newton1(x: list, y: list, req: float):
    n = len(y)
    dy = [y[i+1]-y[i] for i in range(n-1)]
    t = (req - x[0])/(x[1]-x[0])
    P = y[0]
    for i in range(1, n):
        tmp = 1
        for j in range(i):
            tmp *= t - j
        P += dy[i-1] * tmp / fac(i)
    return P


def Newton2(x: list, y: list, req: float):
    n = len(y)
    dy = [y[i+1]-y[i] for i in range(n-1)]
    t = (req - x[n-1])/(x[1]-x[0])
    P = y[n-1]
    for i in range(1, n):
        tmp = 1
        for j in range(i):
            tmp *= t + j
        P += dy[i-1] * tmp / fac(i)
    return P



X = [0.41, 0.46, 0.52, 0.60, 0.65, 0.72]
Y = [2.57418, 2.32513, 2.09336, 1.86203, 1.74926, 1.62098]
req1 = [0.616, 0.478, 0.665, 0.537, 0.673]

x = list(np.linspace(0.180, 0.235, 12))
y = [5.61543, 5.46693, 5.32634, 5.19304, 5.06649, 4.94619,
     4.83170, 4.72261, 4.61855, 4.51919, 4.42422, 4.33337]
req2 = [0.1817, 0.2275, 0.175, 0.2375]


req1.sort()
req2.sort()


plt.figure(figsize=(9, 6))
res1 = [round(Lagrange(X, Y, i), 5) for i in req1]
print("Lagrange", req1, res1, sep='\n')
print(LagrangeP(X, Y))
plt.xticks(X)
plt.xlim(0.4, 0.8)
plt.ylim(1.6, 2.6)
plt.scatter(X, Y, c = "#0000ff", label="initial")
plt.scatter(req1, res1, c = "#00ff00", label="interpolated")
plt.grid(True)
plt.legend()
plt.show()
sp.plot(LagrangeP(X, Y))


plt.figure(figsize=(9, 6))
res2 = [round(Newton1(x, y, i), 5) for i in req2]
res3 = [round(Newton2(x, y, i), 5) for i in req2]
print("Newton1", req2, res2, sep='\n')
print("Newton2", req2, res3, sep='\n')
plt.xticks(x)
plt.ylim(3.5, 6)
plt.scatter(x, y, c="#0000ff", label="initial")
plt.scatter(req2, res2, c="#00ff00", label="interpolated ->")
plt.scatter(req2, res3, c="#ff0000", label="interpolated <-")
plt.grid(True)
plt.legend()
plt.show()
