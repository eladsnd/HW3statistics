import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy import stats

size = 30
exp = 5
std = 1
x = np.array(norm.rvs(size=size, loc=exp, scale=std))


def r1(x, y):
    return np.cov(x, y, bias=True)[0][1] / (np.std(y) * np.std(x))


def B_l_s(x, y):
    return np.std(y) / np.std(x)


if __name__ == '__main__':
    y = 5 * x + 2
    r = r1(x, y)
    bls = B_l_s(x, y)
    print("r : " + str(r) + "\nbls : " + str(bls))

    noise = norm.rvs(loc=0, scale=1, size=size)
    y_noise = y + noise
    r_noise = r1(x, y_noise)
    bls_noise = B_l_s(x, y_noise)
    print("\nnoise:\nr : " + str(r_noise) + "\nbls : " + str(bls_noise))

    X = []
    R = []
    B = []
    for i in range(100):
        i_noise = 0.1 * (i + 5)
        noise = norm.rvs(loc=0, scale=i_noise, size=size)
        y_noise = y + noise
        X.append(i_noise)
        R.append(r1(x, y_noise))
        B.append(B_l_s(x, y_noise))
    X = np.array(X)
    R = np.array(R)
    B = np.array(B)
    plt.figure(0)
    plt.scatter(X, R)
    plt.figure(1)
    plt.scatter(X, B)
    plt.show()
