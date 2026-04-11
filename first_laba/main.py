import matplotlib.pyplot as plt
from math import sqrt, ceil, floor

class Statistics:
    def __init__(self, file_name):
        self.file_name = file_name
        self.data = self.read_data(file_name)

    def read_data(self, file_name):
        with open(file_name, 'r') as f:
            lines = f.readlines()
            data = [float(line.strip()) for line in lines[1:]]
        return data
    def size(self):
        return len(self.data)
    def min(self):
        return min(self.data)
    def max(self):
        return max(self.data)
    def range(self):
        return self.max() - self.min()
    def mean(self):
        return sum(self.data) / self.size()
    def variance(self, unbiased=True):
        mean = self.mean()
        n = self.size()
        return sum((x - mean) ** 2 for x in self.data) / (n - 1 if unbiased else n)
    def standard_deviation(self, unbiased=True):
        return sqrt(self.variance(unbiased))
    def asymmetry(self):
        mean = self.mean()
        std_dev = self.standard_deviation()
        denominator = self.size() * std_dev ** 3
        return sum((x - mean) ** 3 for x in self.data) / denominator
    def quantiles(self, p):
        sorted_data = sorted(self.data)
        n = self.size()
        pos = (n - 1) * p
        lower = floor(pos)
        upper = ceil(pos)
        if lower == upper:
            return sorted_data[lower]
        return  (sorted_data[lower] + sorted_data[upper]) / 2
    def median(self):
        return self.quantiles(0.5)
    def iqr(self):
        return self.quantiles(0.75) - self.quantiles(0.25)
    def statistics_summary(self):
        return {
            'Размер выборки': self.size(),
            'Минимум': self.min(),
            'Максимум': self.max(),
            'Размах': self.range(),
            'Среднее': self.mean(),
            'Выборочная дисперсия (несмещенная)': self.variance(),
            'Выборочная дисперсия (смещенная)': self.variance(unbiased=False),
            'Стандартное отклонение': self.standard_deviation(),
            'Асимметрия': self.asymmetry(),
            'Медиана': self.median(),
            'Интерквартильная широта': self.iqr()
        }
    def print_statistics(self):
        summary = self.statistics_summary()
        for key, value in summary.items():
            print(f"{key}: {value:.3f}" if isinstance(value, float) else f"{key}: {value}")
class Histogram(Statistics):
    def _argmax(self, lst):
        if not lst:
            return None
        max_idx = 0
        max_val = lst[0]
        for i, val in enumerate(lst[1:], 1):
            if val > max_val:
                max_val = val
                max_idx = i
        return max_idx
    
    def moda(self, a, h):
        i = self._argmax(h)
        if i == 0 or i == len(h) - 1:
            return (a[i] + a[i + 1]) / 2
        w = a[1] - a[0]
        left = h[i] - h[i - 1]
        right = h[i] - h[i + 1]

        if left + right == 0:
            return (a[i] + a[i + 1]) / 2
        return a[i] + w * left / (left + right)

    def plot_histogram(self):
        x = sorted(self.data)
        n = self.size()
        k = n // 10 if n >= 10 else n
        x_max = self.max()
        x_min = self.min()
        delta = (x_max - x_min) / (k - 1)
        a0 = x_min - delta / 2
        a = [a0 + i * delta for i in range(k + 1)]
        h = []
        for i in range(k):
            count = sum(1 for value in x if a[i] <= value < a[i + 1])
            h.append(count / (n * delta))
            
        moda = self.moda(a, h)
        print(f"Мода по гистограмме: {moda:.3f}")
        plt.figure(figsize=(10, 6))
        plt.bar(a[:-1], h, width=delta, align='edge', edgecolor='black', alpha=0.7)


        i = self._argmax(h)
        if 0 < i < len(h) - 1:
            x_coords1 = [a[i], a[i+1]]
            y_coords1 = [h[i-1], h[i]]

            x_coords2 = [a[i], a[i+1]]
            y_coords2 = [h[i], h[i+1]]
            
            plt.plot(x_coords1, y_coords1, color='orange', linestyle='-', linewidth=1.5, label='Построение моды')
            plt.plot(x_coords2, y_coords2, color='orange', linestyle='-', linewidth=1.5)
        
        plt.axvline(moda, color='green', label=f'Мода: {moda:.3f}')

        plt.xlabel('x')
        plt.ylabel('Плотность вероятности')
        plt.title('Вероятностная гистограмма')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.show()
        

class EDF(Statistics):
    def plot_edf(self):
        x = sorted(self.data)
        n = self.size()
        
        y = [sum(1 for val in x if val < x_i) / n for x_i in x]
        plt.figure(figsize=(10, 6))
        plt.step(x, y)
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        plt.axhline(y=1, color='red', linestyle='--', alpha=0.5, linewidth=1)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.title('Эмпирическая функция распределения')
        plt.xlabel('x')
        plt.ylabel('F(x)', rotation=0)
        plt.show()

def main():
    FILENAME = 'r1z1.csv'
    stats = Statistics(FILENAME)
    stats.print_statistics()
    hist = Histogram(FILENAME)
    hist.plot_histogram()
    edf = EDF(FILENAME)
    edf.plot_edf()

if __name__ == "__main__":
    main()