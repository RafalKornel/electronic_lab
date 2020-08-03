#!/usr/bin/python3.7

from scipy.optimize import curve_fit
import pylab as pl
import matplotlib.pyplot as plt
import numpy as np

def run(n, name):
    d = pl.loadtxt(name)
    x   = d[:n, 0]
    y   = d[:n, 1]
    u_y = d[:n, 2]


    def f(x, a, b): return a*x + b

    par, cov = curve_fit(f, x, y, sigma=u_y)
    cov = [ cov[i][i]**0.5 for i in range(len(cov)) ]

    print(f"Parametr a dopasowania: {par}, niepewności: {cov}")

    _x = np.linspace(0, x[-1], 30)
    plt.errorbar(d[:, 0], d[:, 1], yerr=d[:, 2], fmt='ro', markersize=4)
    #plt.plot(_x, f(_x, *par), 'b')
    #plt.ylabel("$I_B$ [mV]")
    #plt.xlabel("$U_{we}$")
    plt.show()


run(5, "dane1.txt")
run(13, "dane2.txt")
