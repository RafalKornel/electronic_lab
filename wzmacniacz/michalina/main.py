#!/usr/bin/python3.7

import pylab as pl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

def part_one():
    d       = pl.loadtxt("dane1.txt")
    Uwe     = d[:, 0]
    Uwy     = d[:, 1]
    d_Uwe   = d[:, 2]*0.1 + 0.001 + 0.05*Uwe
    d_Uwy   = d[:, 3]*0.1 + 0.001 + 0.05*Uwy
    k       = Uwy/Uwe
    d_k = ( (d_Uwy/Uwe)**2 + ( (Uwy*d_Uwe)/(Uwe*Uwe))**2 )**0.5

    f = lambda x, a: a*x
    p, cov = curve_fit(f, Uwe[:5:], Uwy[:5:], sigma=d_Uwy[:5:])
    print(f"Parameters: {p}")
    print(f"Errors: {[ cov[i][i]**0.5 for i in range(len(cov))]}")

    plt.errorbar(Uwe, Uwy, yerr=d_Uwy, fmt='bo')
    _x = np.linspace(0, Uwe[4], 10)
    plt.plot(_x, f(_x, *p), 'r', label="Krzywa najlepszego dopasowania")
    plt.xlabel("$U_{we}$ [V]")
    plt.ylabel("$U_{wy}$ [V]")
    plt.show()

    chi_sq      = sum([ (f(Uwe[i], *p) - Uwy[i])**2/d_Uwy[i]**2 for i in range(5) ])
    print(f"Chi_sq kwadrat: {chi_sq}")
    print(f"Chi_sq kreślone: {chi_sq/4}")

    plt.errorbar(Uwe, k, yerr=d_k, fmt='bo')
    plt.xlabel("$U_{we}$ [V]")
    plt.ylabel("$|k|$")
    plt.show()

def part_two():
    d       = pl.loadtxt("dane2.txt")
    f       = d[:, 0]
    Uwe     = d[:, 1]
    Uwy     = d[:, 2]
    d_Uwe   = d[:, 3]*0.1 + 0.001 + 0.05*Uwe
    d_Uwy   = d[:, 4]*0.1 + 0.001 + 0.05*Uwy

    A = Uwy/Uwe
    d_A = ( (d_Uwy/Uwe)**2 + ( (Uwy*d_Uwe)/(Uwe*Uwe))**2 )**0.5

    plt.errorbar(f, A, yerr=d_A, fmt='bo')
    plt.xscale("log")
    plt.xlabel("$f$ [Hz]")
    plt.ylabel("$|k|$")
    plt.show()

if __name__ == "__main__":
    part_one()
    part_two()
