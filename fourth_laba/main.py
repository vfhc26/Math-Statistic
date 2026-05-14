import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
DIR = Path(__file__).resolve().parent
FILENAME = DIR / 'r4z2.csv'
X = 117

class Regression:
    def __init__(self, path):
        self._path = path
        self._data = pd.read_csv(path)
        print(self._data.describe())
        self._fitted = False
        self._a = 0
        self._b = 0
        self._r = 0
    def fit(self):
        x = self._data['X']
        y = self._data['Y']
        n = len(self._data)
        xy_mean = x@y / n
        x_mean = x.mean()
        y_mean = y.mean()
        x2_mean = x@x / n
        y2_mean = y@y / n
        cov = xy_mean - x_mean*y_mean
        self._b =  cov / (x2_mean - x_mean**2)
        self._a = y_mean - self._b * x_mean
        self._r = cov / (np.sqrt(x2_mean - x_mean ** 2) * np.sqrt(y2_mean - y_mean**2))
        self._fitted = True
    def plot(self):
        x = self._data['X']
        y = self._data['Y']
        plt.scatter(x, y, color='blue', label='Данные')
        if (self._fitted):
            x_line = np.array([x.min(), x.max()])
            y_line = self.predict(x_line)
            plt.plot(x_line, y_line, color='red', label='Линия регрессии')
        plt.title(f'Диаграмма рассеяния')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        plt.show()
    def stats(self):
        if not self._fitted:
            print("Модель еще не обучена")
            return
        print(f"Уравнение регрессии: Y = {self._b:.4f} * X + {self._a:.4f}")
        print(f"Коэффициент корреляции (r): {self._r:.4f}")
    def predict(self, x):
        return self._b * x + self._a

if __name__ == '__main__':
    reg = Regression(FILENAME)
    reg.plot()
    reg.fit()
    reg.plot()
    reg.stats()
    predicted = reg.predict(X)
    print(f"Предсказанный Y по X = {X} равен {predicted}")