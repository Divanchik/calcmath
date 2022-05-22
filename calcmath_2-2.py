import numpy as np
from random import uniform
from time import sleep


def A2LU(A: np.ndarray):
    n = A.shape[0]
    L = np.identity(n, float)
    U = np.zeros((n, n), float)
    for i in range(n):
        for j in range(n):
            if i <= j:
                U[i, j] = A[i, j] - np.dot(L[i, :i], U[:i, j])
            if i > j:
                L[i, j] = (A[i, j] - np.dot(L[i, :j], U[:j, j])) / U[j, j]
    return L, U


def proj(a: np.ndarray, b: np.ndarray):
    return (np.dot(a, b) / np.dot(b, b)) * b


def gram_shmidt(A: np.ndarray):
    Q = np.zeros_like(A)
    n, m = A.shape
    for i in range(m):
        b = np.copy(A[:, i])
        for j in range(i):
            b -= proj(A[:, i], Q[:, j])
        e = b / np.linalg.norm(b)
        Q[:, i] = e
    R = Q.T @ A
    return Q, R


def rotations(A: np.ndarray):
    n, m = A.shape
    R = np.copy(A)
    Q = np.identity(n)
    rows, cols = np.tril_indices(n, -1, m)
    for i, j in zip(rows, cols):
        if R[i, j] != 0:
            h = np.hypot(R[j, j], R[i, j])
            c = R[j, j]/h
            s = -R[i, j]/h

            T = np.identity(n)
            T[[j, i], [j, i]] = c
            T[i, j] = s
            T[j, i] = -s

            R = T @ R
            Q = Q @ T.T
    return Q, R


def reflections(A: np.ndarray):
    n = A.shape[0]
    Q = np.identity(n)
    R = np.copy(A)
    for cnt in range(n - 1):
        a = R[cnt:, cnt]
        e = np.zeros_like(a)
        e[0] = np.linalg.norm(a)
        v = a - e
        w = v / np.linalg.norm(v)

        Q_cnt = np.identity(n)
        Q_cnt[cnt:, cnt:] -= 2.0 * np.outer(w, w)

        R = Q_cnt @ R
        Q = Q @ Q_cnt
    return Q, R


def simple_iterations(A: np.ndarray, b: np.ndarray, prec: float, itermax: int):
    print("simple iterations start")
    n, m = A.shape
    c = np.zeros(n)
    B = np.zeros((m, m))

    for i in range(n):
        c[i] = b[i]/A[i, i]
        for j in range(m):
            B[i, j] = 0 if i==j else -A[i, j]/A[i, i]
    print("B\n", B)
    print("c\n", c)
    print("||B|| < 1" if np.linalg.norm(B) < 1 else "||B|| >= 1")
    
    x = np.copy(b)
    itercount = 0
    while True:
        x1 = B @ x + c
        if np.linalg.norm(x1 - x) < prec:
            break
        x = x1
        itercount += 1
        if itercount == itermax:
            print("Iterations limit exceeded")
            return None
    return x


def seidel(A: np.ndarray, b: np.ndarray, eps):
    n = A.shape[0]
    x = np.zeros(n) # столбец нулей
    converge = False
    while not converge:
        x_new = np.copy(x) # copy x
        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - s1 - s2) / A[i][i]

        converge = np.sqrt(sum((x_new[i] - x[i]) ** 2 for i in range(n))) <= eps
        x = x_new
    return x


def seidel1(A: np.ndarray, B: np.ndarray, prec: float):
    n = A.shape[0]
    x = np.zeros(n)

    # L, D, U
    L = np.copy(A)
    L[np.triu_indices(n)] = 0
    U = np.copy(A)
    U[np.tril_indices(n)] = 0
    D = np.copy(A)
    D[np.triu_indices(n, 1)] = 0
    D[np.tril_indices(n, -1)] = 0
    # print(L, D, U, sep='\n')

    count = 0
    while True:
        x1 = np.copy(x)
        x1 = np.linalg.inv(L + D) @ (-U @ x1 + B)
        print("norm", np.linalg.norm(A @ x - B))
        # if np.linalg.norm(x1 - x) <= prec:
        #     break
        if np.linalg.norm(A @ x - B) <= prec:
            break
        x = x1
        count += 1
        print(count)
        print(x)
        print('')
        sleep(1)
    print("iterations:", count)
    return x


np.set_printoptions(precision=4, suppress=True)
# Создать квадратную матрицу из случайных вещественных чисел из (2, 5) размера 4.
# Найти матрицу, обратную к этой матрице с помощью LU разложения.
# Проверить вычисления непосредственным умножением на матрицу А.
print("1)")
n = 4
A = np.array([[uniform(2, 5) for i in range(n)] for i in range(n)])
L, U = A2LU(A)
E = np.identity(n)
y = np.linalg.solve(L, E)
x = np.linalg.solve(U, y)

print("Обратная матрица\n", x)
print("Контроль\n", np.linalg.inv(A))
print("Произведение\n", A @ x, end='\n\n\n')


# Найдите QR разложение матрицы, созданной в пункте 1.
# Проверьте правильность найденного разложения:
#   с помощью умножения Q на R
#   с помощью функции np.linalg.qr
print("2)")
Q, R = np.linalg.qr(x)
print("Контрольный образец", "Q", Q, "R", R, sep='\n')
print("Произведение\n", Q @ R, end='\n\n\n')
Q, R = gram_shmidt(x)
print("Грам-Шмидт", "Q", Q, "R", R, sep='\n')
print("Произведение\n", Q @ R, end='\n\n\n')
Q, R = rotations(x)
print("Метод вращений", "Q", Q, "R", R, sep='\n')
print("Произведение\n", Q @ R, end='\n\n\n')
Q, R = reflections(x)
print("Метод отражений", "Q", Q, "R", R, sep='\n')
print("Произведение\n", Q @ R, end='\n\n\n')


# Решить систему, используя метод Зейделя, с точностью до 0.001,
# приведя к виду, удобному для итераций
print("3)")
A_arr = [
    [0.21, -0.18, 0.75],
    [0.13, 0.75, -0.11],
    [3.01, -0.33, 0.11]]
B_arr = [
    0.11,
    2.00,
    0.13]
A = np.array(A_arr)
B = np.array(B_arr)
prec = 0.001

# A = A[[0, 2, 1], 0:3] # меняем местами строки 1 и 2
# B_new: np.ndarray = np.zeros_like(A) # B
# C = np.zeros_like(B) # C

# n, m = A.shape
# for i in range(n):
#     C[i] = B[i] / A[i, i]
#     for j in range(m):
#         if i != j:
#             B_new[i, j] = -A[i, j] / A[i, i]
# B1 = np.zeros_like(B_new)
# B2 = np.zeros_like(B_new)

# B1[1][0] = B_new[1][0]
# B1[2][0] = B_new[2][0]
# B1[2][1] = B_new[2][1]

# B2[0][1] = B_new[0][1]
# B2[0][2] = B_new[0][2]
# B2[1][2] = B_new[1][2]

# prevX = np.zeros_like(B)

# X = B1.dot(prevX) + B2.dot(prevX) + C
# iter = 0
# while abs(X - prevX).max() > prec:
#     iter += 1
#     prevX = X
#     X = B1.dot(prevX) + B2.dot(prevX) + C

# print("Результат:\n", X)
# print("Результат:\n", seidel(A, B, prec))
# print("Итераций:", iter)
# print("Контрольный образец:\n", np.linalg.solve(A, B))
# print(seidel(A, B, prec))
# print(seidel1(A, B, prec))

print("A\n", A)
print("B\n", B)
print("det A =", round(np.linalg.det(A), 4))
print("rank A =", np.linalg.matrix_rank(A))
print(simple_iterations(A, B, prec, 100))
print("Контроль\n", np.linalg.solve(A, B))