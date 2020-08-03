#!/usr/bin/python3.7

import numpy as np
import os
import matplotlib.pyplot as pl
from scipy.optimize import curve_fit

class Parser:
    def __init__(self, filename, threshold=0.2, folder=False):
        self.filename = filename
        self.threshold = threshold
        

    def read_one_curve(self, filename):
        data = []
        labels = ""

        with open(filename, 'r') as f:
            try:
                data = f.read().split("\n")                         # data array is filled with strings , ex: "3900.53154 18.997 0.167 11  0 A"
            except:
                print(filename)
                return [ (_, 0, 0) for _ in range(2418) ]
                pass
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



parser = Parser("test")
data = parser.read_one_curve("C:\Users\Michalina\Documents\LABY elektroniczna\wykresy\dane.txt")

def transmitancja(f, RC):
    return 1/( (2*np.pi*f*RC)**2 + 1) ** 0.5 


x = np.array(data["freq"])
y = np.array(data["alfa"])
err = np.array(data["alfa_err"])

p, cov = curve_fit(transmitancja, x, y, p0=1e3*1e-9, sigma=err, method='lm')

fit_y = transmitancja(x, p)

print(p)

pl.errorbar(x, y, yerr = err, fmt='bo', markersize=1.8, elinewidth = 0.8)
pl.yscale("log")
pl.xscale("log")

pl.plot(x, fit_y, linewidth = 0.8)

pl.show()
