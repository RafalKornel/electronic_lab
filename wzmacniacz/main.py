#!/usr/bin/python3.8

import pylab as pl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

def part_one():
    d       = pl.loadtxt("dane1.txt")
    Uwe     = d[:, 0]
    Uwy     = d[:, 1]
    d_Uwe   = 0.5*0.1 + 0.002 + 0.01*Uwe
    d_Uwy   = 5*0.1 + 0.002 + 0.01*Uwy

    f = lambda x, a, b: a*x + b
    p, cov = curve_fit(f, Uwe, Uwy, sigma=d_Uwy)
    print(f"Parameters: {p}")
    print(f"Errors: {[ cov[i][i]**0.5 for i in range(len(cov))]}")

    plt.errorbar(Uwe, Uwy, xerr=d_Uwe, yerr=d_Uwy, fmt='bo')
    plt.plot(Uwe, f(Uwe, *p), 'r', label="Krzywa najlepszego dopasowania")
    plt.xlabel("$U_{we}$ [V]")
    plt.ylabel("$U_{wy}$ [V]")
    plt.legend()
    plt.show()

def part_two():
    d       = pl.loadtxt("dane2.txt")
    f       = d[:, 0]
    Uwe     = d[:, 1]/1000
    Uwy     = d[:, 2]
    d_Uwe   = 0.2*0.1 + 0.002 + 0.01*Uwe
    d_Uwy   = 2*0.1 + 0.002 + 0.01*Uwy

    A = Uwy/Uwe
    d_A = ( (d_Uwy/Uwe)**2 + ( (Uwy*d_Uwe)/(Uwe*Uwe))**2 )**0.5

    plt.errorbar(f, A, yerr=d_A, fmt='bo')
    plt.xscale("log")
    plt.xlabel("f [Hz]")
    plt.ylabel("$|k(f)|=|U_{wy}/U_{we}|$")
    plt.show()

if __name__ == "__main__":
    part_one()
    part_two()
