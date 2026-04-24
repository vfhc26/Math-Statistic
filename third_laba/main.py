import pandas as pd
from scipy.stats import chi2
from pathlib import Path

class Confidents_limits:
    def __init__(self, path, Q, type):
        self._file_name= path
        self._data = pd.read_csv(path)
        self._Q = Q
        self._type = type
    def mean(self, x):
        return sum(x) / len(x)
    def variance(self, x, unbiased=True):
        mean = self.mean(x)
        size = len(x)
        return sum((xx - mean)**2 for xx in x) / (size - 1 if unbiased else size)
    def limits(self):
        x = self._data["X"]
        n = len(x)
        S_2 = self.variance(x, unbiased=False)
        alpha = 1 - self._Q
        df = n - 1
        if self._type == 'lower':
            X_crit = chi2.ppf(1 - alpha, df)
            return n * S_2 / X_crit, float('inf')
        elif self._type == 'upper':
            X_crit = chi2.ppf(alpha, df)
            return float('-inf'), n * S_2 / X_crit
        else:
            X_lower = chi2.ppf(1 - alpha / 2, df) 
            X_upper = chi2.ppf(alpha / 2, df)
            return n * S_2 / X_lower, n * S_2 / X_upper
    def stats(self):
        [lower, upper] = self.limits()
        print(f"[{lower}, {upper}]")

def main():
    DIR = Path(__file__).resolve().parent
    FILENAME = DIR / 'r3z2.csv'
    Q = 0.975
    LIMITS = 'lower'
    cl = Confidents_limits(FILENAME, Q, LIMITS)
    cl.stats()

if __name__ == "__main__":
    main()