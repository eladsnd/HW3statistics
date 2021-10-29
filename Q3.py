import math

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

path = "C:\\Users\\Elad\\PycharmProjects\\HW3statistics\\flatprices.csv"
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


def scatterP(x, y):
    plt.scatter(x, y)


def LS(x, y):
    bls = B_l_s(x, y)
    als = A_l_s(x, y)
    ls = bls * x + als
    plt.plot(x, ls, '-b')


def ls_func(size , x, y):
    bls = B_l_s(x, y)
    als = A_l_s(x, y)
    return bls * size + als


def funcs_a_to_c(x, y,title):
    plt.title(title)
    # Q3_A
    scatterP(x, y)
    # Q3_B
    LS(x, y)
    # Q3_C
    slope = B_l_s(x, y)
    r = np.cov(x, y)[0][1] / (np.std(y) * np.std(x))
    r_square = np.square(r)
    sst = np.var(y)
    sse = (r_square + 1) * sst
    print("B_ls : " + str(slope) + "\nR : " + str(r) + "\nR^2 : " + str(r_square) +
          "\nsst : " + str(sst) + "\nsse : " + str(sse))


if __name__ == '__main__':
    x = df['space'].to_numpy()
    y = df['price'].to_numpy()
    print("data for Y : \n")
    plt.figure(0)
    funcs_a_to_c(x, y , "y")
    # Q3_D
    print("the price for an apartment of size 125 sqm is : " + str(ls_func(125,x ,y)))


    print("\ndata for log2 Y : \n")
    plt.figure(1)
    log2y = np.log2(y)
    funcs_a_to_c(x, log2y , "log2 y")
    print("the price for an apartment of size 125 sqm is : " + str(np.power(2,ls_func(125 , x, log2y))))


    print("\ndata for square root Y : \n")
    plt.figure(2)
    sqrty = np.sqrt(y)
    funcs_a_to_c(x, sqrty , "square root y")
    print("the price for an apartment of size 125 sqm is : " + str(np.square(ls_func(125,x ,sqrty))))

    plt.show()
