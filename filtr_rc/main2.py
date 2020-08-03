#!/usr/bin/python3.7

import numpy as np
import matplotlib.pyplot as pl
from scipy.optimize import curve_fit


def read_file(filename):
    #funkcja do wczytywania pliku do programu

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


_x = np.linspace(0, 6, 10000)
y1 = [ 10 if int(_)%2 == 0 else -10 for  _ in _x]
y2 = [ (_x[i] - int(_x[i])) * y1[i] - y1[i]/2 for i in range(len(_x))]

pl.plot(_x, y1)
pl.plot(_x, y2)
pl.show()



#data = read_file("C:/Users/Michalina/Documents/LABY_elektroniczna/Filtry_RC/wykresy/dane.txt")    #tutaj w argumencie podajesz sciezke do pliku
data = read_file("dane.txt")

# TUTAJ POLICZYĆ NIEPEWNOŚCI I POTRZEBNE WIELKOŚCI ITP

f       = np.array(data["freq"])
u_out   = np.array(data["u_out"])
u_in    = np.array(data["u_in"])
alfa    = u_out / u_in

dt      = np.array(data["delta_t"]) * 1e-6 
dt_err  = np.array(data["delta_t_err"]) * 1e-6

data["phi"] = - 2*np.pi * (dt * f)
data["phi_err"] = - 2*np.pi * (dt_err * f)

phi     = np.array(data["phi"])
u_in_err = np.full
alfa_err   = np.array(data["u_out_err"])

#data["phi_err"] = phi * 0.02

phi_err = np.array(data["phi_err"])

x = np.linspace(f[0] * 0.75, f[-1] * 1.25, 10000)



#policzony współczynnik oraz niepewność dla wykresu transmitancji
p_t, cov_t = curve_fit(transmitancja, f, alfa, p0=1e3*1e-9, sigma=alfa_err, method='lm')

fit_alfa = transmitancja(x, p_t)

print("\nParametr RC oraz jego niepewność dopasowana na podstawie transmitancji")
print("RC (transmitancja):", p_t[0])
print("u_RC:", cov_t[0][0]**0.5)
print("relative error:", cov_t[0][0]**0.5/p_t)


pl.errorbar(f, alfa, yerr = alfa_err, fmt='bo', markersize=1.8, elinewidth = 0.8)
pl.yscale("log")
pl.xscale("log")

pl.plot(x, fit_alfa, linewidth = 0.8)

pl.show()



#policzony współczynnik oraz niepewność dla wykresu fazowego
p_p, cov_p = curve_fit(faza, f, phi, p0=1e3*1e-9, sigma=phi_err, method='lm')

fit_phi = faza(x, p_p)
print(phi_err)

print("\n\nParametr RC oraz jego niepewność dopasowana na podstawie fazy")
print("RC (faza):", p_p[0])
print("u_RC:", cov_p[0][0]**0.5)
print("relative error:", cov_p[0][0]**0.5/p_p)
print()

pl.errorbar(f, phi, yerr = phi_err, fmt='bo', markersize=1.8, elinewidth=0.8)

pl.plot(x, fit_phi, linewidth=0.8)
pl.xscale('log')
pl.show()
