import pandas as pd
import numpy as np
from scipy.stats import t

class One_Sample_Student:
    def __init__(self, file_name):
        self.file_name = file_name
        self.data = pd.read_csv(file_name)

    def mean(self, x):
        return sum(x) / len(x)
    
    def variance(self, x, unbiased=True):
        mean = self.mean(x)
        size = len(x)
        return sum((xx - mean)**2 for xx in x) / (size - 1 if unbiased else size)
    
    def std(self, x, unbiased=True):
        return np.sqrt(self.variance(x , unbiased=unbiased))    
    
    def t_statistics(self, d):
        n = len(d)
        d_mean = self.mean(d)
        d_std = self.std(d, False)
        return d_mean * np.sqrt(n - 1) / d_std
    
    def C_crit(self, alpha, n, mode = '!='):
        df = n - 1
        if mode == '>':
            return t.ppf(1 - alpha, df)
        elif mode == '<':
            return t.ppf(alpha, df)
        elif mode == '!=':
            return t.ppf(1 - alpha/2, df)
        else: 
            raise ValueError('неверно введенная альтернатива')
    def p_value(self, t_stat, n, mode='!='):
        df = n - 1
        if mode == '>':
            return 1 - t.cdf(t_stat, df)
        elif mode == '<':
            return t.cdf(t_stat, df)
        elif mode == '!=':
            return 2 * (1 - t.cdf(abs(t_stat), df))
        else: 
            raise ValueError('неверно введенная альтернатива')
    def decision(self, t_stat, c_crit, mode):
        if mode == '>':
            return 'H0 отвергается' if t_stat > c_crit else 'нет оснований отвергнуть H0'
        elif mode == '<':
            return 'H0 отвергается' if t_stat < c_crit else 'нет оснований отвергнуть H0'
        elif mode == '!=':
            return 'H0 отвергается' if abs(t_stat) > c_crit else 'нет оснований отвергнуть H0'
        else:
            raise ValueError('неверно введенная альтернатива')


    def stats(self, alpha=0.05, mode = '!='):
        d = self.data['X'] - self.data['Y']
        n = len(d)
        t_stat = self.t_statistics(d)
        c_crit = self.C_crit(alpha, n, mode)
        p_value = self.p_value(t_stat, n, mode)
        print("="*80)
        print("Одновыборочный парный критерий стьюдента")
        print(f"\nn = {n}")
        print(f"t_набл = {t_stat}")
        print(f"С_крит = {c_crit}")
        print(f"p-значение = {p_value}")
        print(self.decision(t_stat, c_crit, mode), '\n')
class Two_Sample_Student(One_Sample_Student):
    def t_statistics(self, x, y):
        x_mean = self.mean(x)
        y_mean = self.mean(y)
        S2_x = self.variance(x)
        S2_y = self.variance(y)
        n1, n2 = len(x), len(y)
        return (x_mean - y_mean) / np.sqrt(n1 * S2_x + n2*S2_y) * np.sqrt(n1*n2*(n1 + n2 - 2) / (n1 + n2))
    def C_crit(self, alpha, n1, n2, mode='!='):
        df = n1+n2-2
        if mode == '>':
            return t.ppf(1 - alpha, df)
        elif mode == '<':
            return t.ppf(alpha, df)
        elif mode == '!=':
            return t.ppf(1 - alpha/2, df)
        else: 
            raise ValueError('неверно введенная альтернатива')
    def p_value(self, t_stat, n1, n2, mode='!='):
        df = n1+n2-2
        if mode == '>':
            return 1 - t.cdf(t_stat, df)
        elif mode == '<':
            return t.cdf(t_stat, df)
        elif mode == '!=':
            return 2 * (1 - t.cdf(abs(t_stat), df))
        else: 
            raise ValueError('неверно введенная альтернатива')
    def stats(self, alpha=0.025, mode='<'):
        x = self.data['X'].dropna()
        y = self.data['Y'].dropna()
        n1, n2 = len(x), len(y)
        t_stat = self.t_statistics(x, y)
        c_crit = self.C_crit(alpha, n1, n2, mode)
        p_value = self.p_value(t_stat, n1, n2, mode)
        print("="*80)
        print("Двухвыборочный критерий стьюдента")
        print(f"\nn1 = {n1}, n2 = {n2}")
        print(f"t_набл = {t_stat}")
        print(f"С_крит = {c_crit}")
        print(f"p-значение = {p_value}")
        print(self.decision(t_stat, c_crit, mode), '\n')

class KolmogorovTest:
    def __init__(self, file_name):
        self.file_name = file_name
        self.data = pd.read_csv(file_name)
        self.sample = self.data['X'].dropna().values
    def d_statistics(self):
        sample_sorted = np.sort(self.sample)
        n = len(sample_sorted)
        F_theor = 1 - np.exp(-0.5 * sample_sorted)
        F_emp = np.array([np.sum(self.sample <= x) / n for x in sample_sorted])
        D = np.max(np.abs(F_emp - F_theor))
        return D
    def k_crit(self, alpha):
        return np.sqrt(-0.5 * np.log(alpha / 2))
    def p_value(self, T, max_k=200):
        p = 0.0
        for k in range(1, max_k + 1):
            term = (-1)**(k-1) * 2 * np.exp(-2 * k**2 * T**2)
            p += term
        return min(max(p, 0.0), 1.0)
    def stats(self, alpha=0.05):
        n = len(self.sample)
        D = self.d_statistics()
        T = np.sqrt(n) * D
        K_crit = self.k_crit(alpha)
        p_value = self.p_value(T)
        print("=" * 80)
        print("Критерий согласия Колмогорова")
        print(f"\nn = {n}")
        print(f"D_набл = {D:.6f}")
        print(f"T = {T:.6f}")
        print(f"K_crit = {K_crit:.6f}")
        print(f"p-значение = {p_value:.6f}")

        if T > K_crit:
            print("H0 отвергается")
        else:
            print("Нет оснований отвергнуть H0")


def main():
    FILENAME_R2Z1_0 = 'r2z1.0.csv'
    FILENAME_R2Z1 = 'r2z1.csv'
    FILENAME_R2Z2 = 'r2z2.csv'
    one_sample = One_Sample_Student(FILENAME_R2Z1_0)
    one_sample.stats(alpha=0.05, mode='!=')

    two_sample = Two_Sample_Student(FILENAME_R2Z1)
    two_sample.stats(alpha=0.025, mode ='<')

    kolmogorov = KolmogorovTest(FILENAME_R2Z2)
    kolmogorov.stats(alpha=0.05)

if __name__ == '__main__':
    main()
