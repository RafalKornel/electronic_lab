#!/usr/bin/python3.7

import matplotlib.pyplot as pl
import matplotlib.pylab as lab
import numpy as np
from scipy.optimize import curve_fit

def f(u, a, b, c):
    return a * np.exp(b*u) - c

def g(i, a, ig):
    return a * np.log(i/ig + 1)

r = 1.00082e3
data = lab.loadtxt("dane.txt")

# napięcia są w woltach
u_gen   = data[:, 0] * 1e-3
u_r     = data[:, 1] * 1e-3
u_d     = data[: ,2] * 1e-3
_sigma  = data[:, 3] * 1e-3

#natężenie zrobimy w mikroapmerach
i_d     = u_r/r * 1e3

print(data)

p, cov = curve_fit(g, i_d, u_d, sigma = _sigma)
print(f"Parameters: {p}")
print(f"Covariance: {cov}")
print(f"u_a: {cov[0][0]**0.5}")
print(f"u_c: {cov[1][1]**0.5}")


'''
pl.errorbar(u_d, i_d, xerr=_sigma, fmt='bo', label="Punkty pomiarowe")

pl.ylabel("Natężenie I [mA]")
pl.xlabel("Napięcie U [V]")
pl.legend(loc="upper left")
pl.show()
'''

#pl.plot(i_d, u_d, 'bo')
pl.errorbar(i_d, u_d, yerr=_sigma, fmt='bo', label="Punkty pomiarowe")
#pl.plot(u_d, f(u_d, *p), linewidth=1.2)

x = np.linspace(min(i_d)*0.9, max(i_d)*1.1, 100)

pl.plot(x, g(x, *p), 'r', linewidth=1.2, label="Krzywa najlepszego dopasowania")
pl.xlabel("Natężenie I [mA]")
pl.ylabel("Napięcie U [V]")
pl.legend(loc="lower right")
pl.show()
