import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon

# Файлы (те же, что в вашем коде)
FILENAME_R2Z1_0 = 'r2z1.0.csv'
FILENAME_R2Z1 = 'r2z1.csv'
FILENAME_R2Z2 = 'r2z2.csv'

# Настройка стиля (простой)
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)

# ================================
# 1. Парные данные (r2z1.0.csv) – разности X-Y
# ================================
df1 = pd.read_csv(FILENAME_R2Z1_0)
# Предполагаем, что в файле есть колонки X и Y (как в вашем коде)
diff = df1['X'] - df1['Y']

plt.figure()
plt.hist(diff, bins=15, edgecolor='black', alpha=0.7, color='skyblue')
plt.axvline(x=0, color='red', linestyle='--', label='нулевая разность')
plt.axvline(x=diff.mean(), color='green', linestyle='-', label=f'среднее = {diff.mean():.3f}')
plt.title('Гистограмма разностей (X - Y) для r2z1.0.csv')
plt.xlabel('X - Y')
plt.ylabel('Частота')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ================================
# 2. Две независимые выборки (r2z1.csv) – сравнение X и Y
# ================================
df2 = pd.read_csv(FILENAME_R2Z1)
x2 = df2['X'].dropna()
y2 = df2['Y'].dropna()

plt.figure()
# Гистограммы с прозрачностью
plt.hist(x2, bins=15, alpha=0.5, label='X', color='blue', edgecolor='black')
plt.hist(y2, bins=15, alpha=0.5, label='Y', color='orange', edgecolor='black')
plt.axvline(x=x2.mean(), color='blue', linestyle='--', label=f'среднее X = {x2.mean():.3f}')
plt.axvline(x=y2.mean(), color='orange', linestyle='--', label=f'среднее Y = {y2.mean():.3f}')
plt.title('Сравнение распределений X и Y (r2z1.csv)')
plt.xlabel('Значение')
plt.ylabel('Частота')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Ящик с усами (boxplot) – наглядно показывает медианы и разброс
plt.figure()
plt.boxplot([x2, y2], labels=['X', 'Y'])
plt.title('Ящик с усами для X и Y (r2z1.csv)')
plt.ylabel('Значение')
plt.grid(True, alpha=0.3)
plt.show()

# ================================
# 3. Выборка для критерия Колмогорова (r2z2.csv) – X
# ================================
df3 = pd.read_csv(FILENAME_R2Z2)
sample = df3['X'].dropna()

# Теоретическая плотность экспоненциального распределения с масштабом 2 (интенсивность 0.5)
# f(x) = 0.5 * exp(-0.5*x) для x>=0
x_vals = np.linspace(0, sample.max(), 200)
theor_pdf = expon.pdf(x_vals, scale=2.0)  # scale = 1/lambda = 2

plt.figure()
plt.hist(sample, bins=20, density=True, alpha=0.6, color='green', edgecolor='black', label='Эмпирическая гистограмма')
plt.plot(x_vals, theor_pdf, 'r-', linewidth=2, label='Теоретическая плотность (Exp(λ=0.5))')
plt.title('Проверка согласия с экспоненциальным распределением (r2z2.csv)')
plt.xlabel('X')
plt.ylabel('Плотность')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Дополнительно: эмпирическая функция распределения vs теоретическая
from statsmodels.distributions.empirical_distribution import ECDF
ecdf = ECDF(sample)
x_grid = np.sort(sample)
theor_cdf = expon.cdf(x_grid, scale=2.0)

plt.figure()
plt.step(x_grid, ecdf(x_grid), where='post', label='Эмпирическая ФР')
plt.plot(x_grid, theor_cdf, 'r--', label='Теоретическая ФР (Exp(λ=0.5))')
plt.title('Функции распределения (r2z2.csv)')
plt.xlabel('X')
plt.ylabel('F(x)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()