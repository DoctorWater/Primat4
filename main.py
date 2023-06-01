import numpy as np
import scipy.sparse as sp
import numpy.linalg as la
from matplotlib import pyplot as plt


def generate_system_data(n, k):
    A = generate_matrix_A(n, k)
    F = generate_vector_F(n)
    return A, F


def generate_matrix_A(n, k):
    A = np.zeros((n, n))

    for i in range(n):
        row_sum = 0
        for j in range(n):
            if i != j:
                A[i, j] = np.random.choice([-4, -3, -2, -1, 0])
                row_sum += abs(A[i, j])

        if i == 0:
            A[i, i] = -row_sum + 10 ** (-k)
        else:
            A[i, i] = -row_sum

    return A


def generate_vector_F(n):
    return np.arange(1, n + 1)


# Метод Гаусса
def gauss_elimination(A, b):
    n = len(A)

    # Прямой ход метода Гаусса
    for i in range(n):
        # Поиск ведущего элемента в столбце i
        max_index = i
        for j in range(i + 1, n):
            if abs(A[j, i]) > abs(A[max_index, i]):
                max_index = j

        # Обмен строк, если найденный ведущий элемент не находится в текущей строке
        if max_index != i:
            A[[i, max_index]] = A[[max_index, i]]
            b[[i, max_index]] = b[[max_index, i]]

        # Приведение к треугольному виду
        for j in range(i + 1, n):
            factor = A[j, i] // A[i, i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]

    # Обратный ход метода Гаусса
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) // A[i, i]

    return x


# Метод LU-разложения
def lu_decomposition(A):
    n = A.shape[0]
    L = sp.eye(n, format='lil')  # Единичная матрица L
    U = sp.lil_matrix((n, n))  # Пустая матрица U

    for k in range(n):
        U[k, k] = A[k, k] - L[k, :k].dot(U[:k, k].toarray().flatten())
        for i in range(k + 1, n):
            L[i, k] = (A[i, k] - L[i, :k].dot(U[:k, k].toarray().flatten())) / U[k, k]
        for j in range(k + 1, n):
            U[k, j] = A[k, j] - L[k, :k].dot(U[:k, j].toarray().flatten())

    return L, U


def solve_lu(L, U, b):
    n = L.shape[0]
    y = np.zeros(n)
    x = np.zeros(n)

    # Решение Ly = b
    for i in range(n):
        y[i] = (b[i] - L[i, :i].dot(y[:i])) / L[i, i]

    # Решение Ux = y
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1:].dot(x[i + 1:])) / U[i, i]

    return x


# Метод Якоби
def jacobi(A, b, x0, max_iterations=100, tolerance=1e-6):
    n = len(A)
    x = x0.copy()

    for _ in range(max_iterations):
        x_new = np.zeros(n)
        for i in range(n):
            x_new[i] = (b[i] - A[i, :].dot(x) + A[i, i] * x[i]) / A[i, i]

        if np.linalg.norm(x_new - x) < tolerance:
            return x_new

        x = x_new

    return x


def evaluate_jacobi(N, K):
    while 1:
        A, F = generate_system_data(N, K)
        det_A = np.linalg.det(A)
        if det_A != 0:
            break
    x0 = np.zeros(n)
    x = jacobi(A, F, x0)
    cond1 = la.cond(A)
    exact_solution = np.linalg.solve(A, F)
    error1 = la.norm(x - exact_solution)
    return cond1, error1


n = 3
k_values = [i for i in range(1, 10)]
cond_values = []
error_values = []

for k in k_values:
    cond, error = evaluate_jacobi(n, k)
    cond_values.append(cond)
    error_values.append(error)

plt.plot(k_values, cond_values, label="Number of Condition")
plt.xlabel('k')
plt.ylabel('Condition Number')
plt.title('Dependency of Condition Number on k')
plt.legend()
plt.show()

plt.plot(k_values, error_values, label="Solution Error")
plt.xlabel('k')
plt.ylabel('Error')
plt.title('Dependency of Solution Error on k')
plt.legend()
plt.show()
