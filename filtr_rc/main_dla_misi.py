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

def chi(fun, args, x, y, u):
    f_x = fun(x, *args)
    return np.sum((f_x - y)**2 / u**2)


#data = read_file("C:/Users/Michalina/Documents/LABY_elektroniczna/Filtry_RC/wykresy/dane.txt")    #tutaj w argumencie podajesz sciezke do pliku
data = read_file("dane_michalina.txt")

f       = np.array(data["f"])
alfa    = np.array(data["trans"])
phi     = np.array(data["phi"])

alfa_err   = np.array(data["trans_err"])
phi_err = np.array(data["phi_err"])
x = np.linspace(f[0] * 0.75, f[-1] * 1.25, 10000)



#policzony współczynnik oraz niepewność dla wykresu transmitancji
p_trans, cov_trans = curve_fit(transmitancja, f, alfa, p0=1e3*1e-9, sigma=alfa_err, method='lm')

fit_alfa = transmitancja(x, p_trans)

print("\nParametr RC oraz jego niepewność dopasowana na podstawie transmitancji")
print("RC (transmitancja):", p_trans[0])
print("u_RC:", cov_trans[0][0]**0.5)
print("relative error:", cov_trans[0][0]**0.5/p_trans[0])

chi_transmitancja = chi(transmitancja, p_trans, f, alfa, alfa_err)
print(f"Chi kwadrat dla transmitancji: {chi_transmitancja}")
print(f"Chi kwadrat kreślone: {chi_transmitancja/(len(f)-1)}")


pl.errorbar(f, alfa, yerr = alfa_err, fmt='bo', markersize=1.8, elinewidth = 0.8)
pl.yscale("log")
pl.xscale("log")

pl.plot(x, fit_alfa, linewidth = 0.8)

pl.show()



#policzony współczynnik oraz niepewność dla wykresu fazowego
p_faza, cov_faza = curve_fit(faza, f, phi, p0=1e3*1e-9, sigma=phi_err, method='lm')

fit_phi = faza(x, p_faza)

print("\n\nParametr RC oraz jego niepewność dopasowana na podstawie fazy")
print("RC (faza):", p_faza[0])
print("u_RC:", cov_faza[0][0]**0.5)
print("relative error:", cov_faza[0][0]**0.5/p_faza[0])

chi_faza = chi(faza, p_faza, f, phi, phi_err)
print(f"Chi kwadrat dla fazy: {chi_faza}")
print(f"Chi kwadrat kreślone: {chi_faza/(len(f)-1)}")

pl.errorbar(f, phi, yerr = phi_err, fmt='bo', markersize=1.8, elinewidth=0.8)

pl.plot(x, fit_phi, linewidth=0.8)
pl.xscale('log')
pl.show()
