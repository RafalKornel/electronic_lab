#!/usr/bin/python3.7

import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def something():
    d   = pl.loadtxt("alt_dane2.txt")
    x   = d[:, 0]
    u_x = d[:, 1]
    y   = d[:, 2]
    u_y = d[:, 3]

    plt.errorbar(x, y, xerr=u_x, yerr=u_y, fmt='bo')

    plt.xlabel("Uwe [V]")
    plt.ylabel("Uwy [V]")
    plt.show()

    print(d)

def part_one():
    d = pl.loadtxt("dane1.txt")
    Uwe     = d[:, 0]
    Uwy     = d[:, 1]
    d_Uwy   = d[:, 2]*0.1 + 0.002 + 0.01*Uwy
    E       = d[:, 3]
    d_E     = d[:, 4]*0.1 + 0.002 + 0.01*E

    Rb = 329.5e3
    Rl = 234.92

    Ib = (Uwe - 0.65)/Rb
    Ic = abs(E - Uwy)/Rl
    d_Ic = ( (d_E/Rl)**2 + (d_Uwy/Rl)**2 )**0.5

    #zamiana z A na mA
    Ib      *= 1e3
    Ic      *= 1e3
    d_Ic    *= 1e3

    def I(x, beta, b): return beta * x + b

    p, cov = curve_fit(I, Ib, Ic, sigma=d_Ic)
    print(f"Parameters: {p}")
    print(f"Errors: {[ cov[i][i]**0.5 for i in range(len(cov)) ]}")

    print(f"Ib: {Ib}")
    print(f"Ic: {Ic}")
    plt.errorbar(Ib, Ic, yerr=d_Ic, fmt='bo')
    plt.plot(Ib, I(Ib, *p), 'r')
    #plt.plot(Ib, Ib, 'bo')
    plt.xlabel("$I_B$ [mA]")
    plt.ylabel("$I_C$ [mA]")
    plt.show()

def part_two():
    d       = pl.loadtxt("alt_dane2.txt")
    Uwe     = d[:, 0]
    d_Uwe   = d[:, 1]
    Uwy     = d[:, 2]
    d_Uwy   = d[:, 3]

#    plt.errorbar(Uwe, Uwy, xerr=d_Uwe, yerr=d_Uwy, fmt='bo')
#    plt.show()

    n = 12
    def f(x, a, b): return a*x + b

    par, cov = curve_fit(f, Uwe[:n], Uwy[:n], sigma=d_Uwy[:n])
    cov = [ cov[i][i]**0.5 for i in range(len(cov)) ]

    print(f"Parametr a dopasowania: {par}, niepewności: {cov}")

    _x = np.linspace(0, Uwe[n], 30)
    plt.errorbar(d[:, 0], d[:, 2], xerr=d[:, 1], yerr=d[:, 3], fmt='ro', markersize=4)
    plt.plot(_x, f(_x, *par), 'b')
    plt.xlabel("$U_{we}$ [V]")
    plt.ylabel("$U_{wy}$ [V]")
    plt.show()

def part_three():
    d       = pl.loadtxt("alt_dane3.txt")
    f       = d[:, 0]   # [hz]
    Uwe     = d[:, 1]   # [V]
    d_Uwe   = d[:, 2]   # [V]
    Uwy     = d[:, 3]   # [V]
    d_Uwy   = d[:, 4]   # [V]

    A = Uwy/Uwe
    d_A = ( (d_Uwy/Uwe)**2 + ( (Uwy*d_Uwe)/(Uwe*Uwe))**2 )**0.5

    plt.errorbar(f, A, yerr=d_A, fmt='bo')
    plt.xscale('log')
    plt.xlabel("f [Hz]")
    plt.ylabel("k = Uwe/Uwy")
    plt.show()


if __name__ == "__main__":
    part_one()
    part_two()
    part_three()
