import math

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats

path = "C:\\Users\\Elad\\PycharmProjects\\HW3statistics\\heights.csv"
df = pd.read_csv(path)


def B_r_l(x, y):
    denominator = x[math.ceil(len(x) / 6)] - x[math.ceil(5 * len(x) / 6)]
    numerator = y[math.ceil(len(x) / 6)] - y[math.ceil(5 * len(x) / 6)]
    return numerator / denominator


def A_r_l(x, y):
    brl = B_r_l(x, y)
    r = y - x * brl
    return r[math.ceil(len(x) / 2)]


def B_l_s(x, y):
    cov = np.cov(x, y)[0][1]
    var = np.var(x)
    return cov / var


def A_l_s(x, y):
    bls = B_l_s(x, y)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    return y_mean - x_mean * bls


if __name__ == '__main__':
    plt.figure(0)
    # Q2_a
    df.sort_values(by=['HEIGHT'])
    y = df['WEIGHT'].to_numpy()
    x = df['HEIGHT'].to_numpy()
    plt.ylim(60, 80)
    plt.scatter(x, y)
    # Q3_b_res
    a = A_r_l(x, y)
    b = B_r_l(x, y)
    res = b * x + a
    plt.plot(x, res, '-r')

    # Q3_c_ls
    bls = B_l_s(x, y)
    als = A_l_s(x, y)
    ls = bls * x + als
    plt.plot(x, ls, '-b')

    # Q3_d
    r = np.cov(x, y)[0][1] / (np.std(y) * np.std(x))

    r_square = np.square(r)
    print("B_ls : " + str(bls) + "\nR : " + str(r) + "\nR^2 : " + str(r_square))

    plt.figure(1)
    df= df.drop(df['HEIGHT'].idxmax())
    df.sort_values(by=['HEIGHT'])
    y = df['WEIGHT'].to_numpy()
    x = df['HEIGHT'].to_numpy()
    plt.ylim(60, 80)
    plt.scatter(x, y)
    # Q3_b_res
    a = A_r_l(x, y)
    b = B_r_l(x, y)
    res = b * x + a
    plt.plot(x, res, '-r')

    # Q3_c_ls
    bls = B_l_s(x, y)
    als = A_l_s(x, y)
    ls = bls * x + als
    plt.plot(x, ls, '-b')

    # Q3_d
    r = np.cov(x, y)[0][1] / (np.std(y) * np.std(x))

    r_square = np.square(r)
    print("B_ls : " + str(bls) + "\nR : " + str(r) + "\nR^2 : " + str(r_square))

    plt.show()
