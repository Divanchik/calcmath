from typing import Callable
import matplotlib.pyplot as plt
import numpy as np
from math import cos, sin


def f1(x: float):
    return -1.78*x**3 - 5.05*x**2 + 3.64*x + 1.37


def f1d(x: float):
    return -5.34*x**2-10.1*x+3.64


def halfdiv(func: Callable[[float], float], a: float, b: float, eps: float):
    itercount = 0
    while True:
        itercount += 1
        x = (a+b)/2
        y = func(x)
        if abs(y) < eps:
            print("halfdiv i =", itercount)
            return round(x, 5)
        if y < 0:
            a = x
        else:
            b = x


def hordes(func: Callable[[float], float], a: float, b: float, eps: float):
    itercount = 0
    while True:
        itercount += 1
        x = a - (func(a)*(b-a))/(func(b)-func(a))
        y = func(x)
        if abs(y) < eps:
            print("hordes i =", itercount)
            return round(x, 5)
        if y < 0:
            a = x
        else:
            b = x


def newton(func: Callable[[float], float], funcd: Callable[[float], float], a: float, b: float, eps: float):
    x1 = a
    x2 = b
    itercount = 0
    while True:
        itercount += 1
        x1 = x1 - func(x1)/funcd(x1)
        if abs(func(x1)) < eps:
            print("newton a i =", itercount)
            break
    itercount = 0
    while True:
        itercount += 1
        x2 = x2 - func(x2)/funcd(x2)
        if abs(func(x2)) < eps:
            print("newton b i =", itercount)
            break
    return round(x1, 5), round(x2, 5)


def F1(x: float):
    return cos(x + 0.5) - 2


def F2(y: float):
    return sin(y)/2


res1 = halfdiv(f1, -5, 2, 0.0001)
print("Answer:", res1)
res2 = hordes(f1, -5, 2, 0.0001)
print("Answer:", res2)
res3, res4 = newton(f1, f1d, -5, 2, 0.0001)
print("Answer:", res3, res4)

plt.figure(figsize=(12, 7))
X1 = list(np.linspace(-5, 2, 100))
plt.plot(X1, [f1(i) for i in X1])
plt.scatter([res3, res4], [f1(res3), f1(res4)],
            c="#005555", label="newton", s=80)
plt.scatter([res1], [f1(res1)], c="#ff0000", label="halfdiv", s=40)
plt.scatter([res2], [f1(res2)], c="#00ff00", label="hordes", s=40)
plt.grid(True)
plt.legend()
plt.show()

print('')
x0 = -0.5
y0 = -1
itercount = 0
while True:
    itercount += 1
    x = x0
    y = y0
    J = np.array([[-sin(x+0.5), -1], [-2, cos(y)]], float)
    F = np.array([-(cos(x+0.5)-y-2), -(sin(y)-2*x)])
    delta = np.linalg.solve(J, F)
    x0, y0 = x0+delta[0], y0+delta[1]
    if delta[0] < 0.0001:
        break
x0, y0 = round(x0, 8), round(y0, 8)
print("Answer:", (x0, y0))
print("Iterations:", itercount)


plt.figure(figsize=(9, 9))
X1 = list(np.linspace(-5, 5, 100))
Y1 = [F1(i) for i in X1]
Y2 = list(np.linspace(-5, 5, 100))
X2 = [F2(i) for i in Y2]
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.xticks(range(-5, 6))
plt.yticks(range(-5, 6))
plt.plot(X1, Y1, c="#0000ff")
plt.plot(X2, Y2, c="#ff0000")
plt.scatter([x0], [y0], c="#00aaaa")
plt.grid(True)
plt.show()