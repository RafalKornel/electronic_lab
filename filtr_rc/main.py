#!/usr/bin/python3.7

import numpy as np
import matplotlib.pyplot as pl
from scipy.optimize import curve_fit


def read_file(filename):
    data = []
    labels = ""
    with open(filename, 'r') as f:
        data = f.read().split("\n")                         # data array is filled with strings , ex: "3900.53154 18.997 0.167 11  0 A"
        data = [ o.split() for o in data ]                  # converting strings to lists,        ex: ['3900.53154', '18.997', '0.167', '11', '0', 'A']
        labels = data[0]
        data = data[1:]
        data.pop()                                          # removes last, empty list from data array

    data_dict = {}
    for label in labels:
        data_dict[label] = []

    for entry in data:
        for i, label in enumerate(labels):
            data_dict[label].append(float(entry[i]))
    
    return data_dict


def transmitancja(f, RC):
    return 1/( (2*np.pi*f*RC)**2 + 1) ** 0.5 

def faza(f, RC):
    return np.arctan(-2*np.pi*f*RC)

data = read_file("dane.txt")    #tutaj w argumencie podajesz sciezke do pliku

data["alfa"] = [ data["u_out"][i]/data["u_in"][i] for i in range(len(data["u_in"])) ]
data["alfa_err"] = [ data["u_out_err"][i]**2 + (data["u_out_err"][0]*data["alfa"][i])**2/data["u_in"][i]**2 for i in range(len(data["u_in"])) ]  

f = np.array(data["freq"])
a = np.array(data["alfa"])

a_err = np.array(data["alfa_err"])

dt = np.array(data["delta_t"]) * 1e-6 
dt_err = np.array(data["delta_t_err"]) * 1e-6
phi = - 2*np.pi * (dt * f)
RC_true = 95.8718e-6

p_trans, cov_trans = curve_fit(transmitancja, f, a, p0=1e3*1e-9, sigma=a_err, method='lm')
p_faza, cov_faza = curve_fit(faza, f/(2*np.pi), phi, p0=1e3*1e-9, sigma=dt_err, method='lm')

x = np.linspace(f[0] * 0.75, f[-1] * 1.25, 10000)
fit_alfa = transmitancja(x, p_trans)
fit_phi = faza(x, p_faza)

pl.errorbar(f, phi, yerr = dt_err, fmt='bo', markersize=1.8, elinewidth=0.8, label="Zmierzone przesunięcie fazowe")

pl.plot(x, fit_phi, 'r', linewidth=0.8, label="Dopasowanie modelowe")
#pl.plot(x, faza(x, RC_true), 'g', linewidth = 0.8, label="Zmierzone")
pl.xscale('log')
pl.xlabel("Częstotliwość [Hz]")
pl.ylabel("Przesunięcie fazowe")
pl.legend()
pl.show()


print("RC dla transmitancji:", p_trans)
print("u_rc:", cov_trans[0]**0.5)
print("relative error:", cov_trans[0]**0.5/p_trans)

print("RC dla fazy:", p_faza)
print("u_rc:", cov_faza[0]**0.5)
print("relative error:", cov_faza[0]**0.5/p_faza)


pl.errorbar(f, a, yerr = a_err, fmt='bo', markersize=1.8, elinewidth = 0.8, label="Stosunek zmierzonych napięć")
pl.yscale("log")
pl.xscale("log")

#h = lambda _: [0.5**0.5 for i in x]
pl.plot(x, fit_alfa, 'r', linewidth = 0.8, label="Dopasowanie modelowe")
#pl.plot(x, transmitancja(x, RC_true), 'g', linewidth = 0.8, label="Zmierzone")
#pl.plot(x, h(x), "--r", linewidth = 0.8, label="Prosta y=1/sqrt(2)")
pl.legend()
#pl.title("Zależność transmitancji od częstości w skali log-log")
pl.xlabel("Częstotliwość [Hz]")
pl.ylabel("Transmitancja")

pl.show()
