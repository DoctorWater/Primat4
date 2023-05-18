import matplotlib.pyplot as plt
import numdifftools as nd
import numpy as np
from openpyxl import Workbook

ITER_COUNT_GLOBAL = 0


def generate_quadratic_function(num_variables, condition_number):
    # Генерация случайной положительно определенной матрицы A
    A = np.random.rand(num_variables, num_variables)
    A = np.dot(A, A.transpose())

    # Вычисление степени обусловленности матрицы A
    _, s, _ = np.linalg.svd(A)
    if np.min(s) > 0:
        cond = np.max(s) / np.min(s)
    else:
        cond = np.inf

    # Масштабирование матрицы A для достижения заданной степени обусловленности
    A = A * (condition_number / cond) ** 0.5

    # Генерация случайного вектора b
    b = np.random.rand(num_variables)

    # Обычная функция, вычисляющая значение квадратичной функции для вектора x
    def quadratic_function(x):
        return 0.5 * np.dot(x, np.dot(A, x)) - np.dot(b, x)

    return quadratic_function


def f(x):
    return (x[0] + x[1]) ** 2 - 5


def func(x, y):
    return (x + y) ** 2 - 5


def df(x):
    con_fin = lambda inner_x: (inner_x[0] + inner_x[1]) ** 2 - 5
    return gradient(con_fin, x)


def gradient(x, y):
    return nd.Gradient(x)(y)


def constGradient():
    x = np.array([-2, 3])

    # Задаем постоянный шаг
    alpha = 0.1

    # Задаем максимальное число итераций
    max_iterations = 200

    path = [x]
    # Начинаем итерации
    print("Constant Gradient method")
    for i in range(max_iterations):
        # Вычисляем градиент в текущей точке
        grad = df(x)
        # Обновляем значение точки по формуле градиентного спуска
        x = x - alpha * grad
        # Вычисляем значение функции в новой точке
        f_val = f(x)
        # Выводим информацию о текущей итерации
        path.append(x)
    print("Iterations:", max_iterations, "Point:", x[0], ",", x[1], " Function value:", f_val)
    return np.array(path)


def stepDivision(current_function, x0_0):
    # Параметры градиентного спуска
    alpha = 1  # Начальное значение шага
    beta = 0.5  # Коэффициент дробления шага
    c = 0.5  # Коэффициент уловия Армихо
    grad_tol = 1e-6  # Порог точности для градиента
    iterCount = 0
    path = [x0_0]
    # Цикл градиентного спуска
    max_iter = 1000
    for i in range(max_iter):
        grad = gradient(current_function, x0_0)  # Вычисление градиента
        d = -grad  # Направление движения
        alpha0 = alpha  # Запоминаем начальное значение шага
        while current_function(x0_0 + alpha * d) > current_function(x0_0) + c * alpha * np.dot(grad, d):
            alpha *= beta  # Дробим шаг
        x1 = x0_0 + alpha * d  # Обновление значения переменных
        if np.linalg.norm(grad) < grad_tol:  # Проверка на достижение точности
            break
        x0_0 = x1  # Перезапись значения переменных
        alpha = alpha0  # Возвращаем начальное значение шага
        iterCount += 1
        path.append(x0_0)

    # Вывод результата
    print("Минимум функции достигается в точке:", x0_0)
    print("Значение функции в этой точке:", current_function(x0_0))
    print("Количество итераций:", iterCount, "\n")
    global ITER_COUNT_GLOBAL
    ITER_COUNT_GLOBAL = iterCount
    return np.array(path)


def fastestLowering():
    # Определение начальной точки
    x0 = np.array([4, -5])

    # Параметры метода
    tol = 1e-6  # Порог точности
    eta = 0.1  # Начальный шаг метода
    max_iter = 1000  # Максимальное число итераций
    countIter = 0
    path = [x0]
    # Цикл метода
    for i in range(max_iter):
        grad = df(x0)  # Вычисление градиента
        if np.linalg.norm(grad) < tol:  # Проверка на достижение точности
            break
        d = -grad  # Направление движения
        a = eta  # Начальное значение шага
        f0 = f(x0)
        counter = 0
        while True:
            counter += 1
            x1 = x0 + a * d  # Вычисление новой точки
            f1 = f(x1)  # Вычисление значения функции в новой точке
            if f1 > f0 + 0.5 * a * np.dot(grad, d):
                # Если значение функции увеличилось, используем параболическую интерполяцию
                a = -0.5 * np.dot(grad, d) * a ** 2 / (f1 - f0 - np.dot(grad, d) * a)
            else:
                # Иначе, мы нашли подходящее значение шага
                break
            if counter > 1000:
                break
        x0 = x1  # Перезапись значения переменных
        countIter += 1
        path.append(x0)

    # Вывод результата
    print("Fastest lowering method")
    print("Минимум функции достигается в точке:", x0)
    print("Значение функции в этой точке:", f(x0))
    print("Количество итераций:", countIter)
    return np.array(path)


def conjugate_gradient(f, df, x0, tol=1e-6, max_iter=1000):
    x = x0
    g = df(x)
    d = -g
    k = 0
    path = [x]
    while k < max_iter:
        alpha = 0.1
        phi = lambda a: f(x + a * d)
        phi_p = lambda a: np.dot(df(x + a * d), d)
        alpha = backtracking_line_search(phi, phi_p, alpha)
        x_prev = x
        x = x + alpha * d
        g_prev = g
        g = df(x)
        beta = np.dot(g, g) / np.dot(g_prev, g_prev)
        d = -g + beta * d
        if np.linalg.norm(x - x_prev) < tol:
            break
        k += 1
        path.append(x)
    print("Conjurate gradient method")
    print("Количество итераций:", k)
    print("Минимум функции:", x)
    print("Значение функции в минимуме:", f(x))
    return np.array(path)


def backtracking_line_search(phi, phi_p, alpha, rho=0.5, c=1e-4):
    while phi(alpha) > phi(0) + c * alpha * phi_p(0):
        alpha = rho * alpha
    return alpha


def draw(path, name):
    # print(path)
    x0 = np.array([1, 1])

    # создание сетки значений для графика
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    # создание графика с линиями уровня
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    cont = ax.contour(X, Y, Z, levels=np.linspace(-5000, 5000, 200), cmap='gray')
    plt.clabel(cont, fontsize=10)

    # отрисовка траектории
    ax.scatter(x[0], x[1])
    ax.plot([x[0] for x in path], [x[1] for x in path], 'r', linewidth=2)

    # настройка внешнего вида графика
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(name)

    plt.show()


wb = Workbook()
sheet = wb.active
column = 2
sheet['A1'] = "Количество переменных"
sheet['B1'] = "Степень обусловленности"
sheet['C1'] = "Количество итераций"
x0_0 = np.array([1])
for i in range(1, 21):
    for j in range(1, 21):
        stepDivision(generate_quadratic_function(i, j), x0_0)
        sheet['A' + str(column)] = i
        sheet['B' + str(column)] = j
        sheet['C' + str(column)] = ITER_COUNT_GLOBAL
        column += 1
    x0_0 = np.append(x0_0, [1])
wb.save("example.xlsx")
