# -*- coding: utf-8 -*-
"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ANALYSIS OF BTW SANDPILE MODEL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Python 3.12

@author: Alberto Zaghini

ColSup 2024
"""

# Necessary packages
import numpy as np
import matplotlib.pyplot as plt
import lmfit
import datetime

# Styling
plt.style.use('fast')

# Load data
avalanches = np.load('data.npz')['avalanches']
grid_size = int(np.size(avalanches, axis=1)/4)

print('Simulated grid size: ', grid_size, 'x', grid_size)

# Distance between points


def distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

# Maximum distance between points in an array


def max_distance(arr):
    max_dist = 0
    for i in range(np.size(arr, axis=0)-1):
        for j in range(i+1, np.size(arr, axis=0)-1):
            dist = np.linalg.norm(arr[i] - arr[j])
            max_dist = max(max_dist, dist)
    return max_dist

# Fit function (power law)


def fit_func(x, c, ld):
    return c * (x ** (-ld))


model = lmfit.model.Model(fit_func, independent_vars=['x'])


# Close sim plot if still open
plt.close()

# Initialize arrays for data
diameter, sites, topplings = np.array([], dtype=int), np.array(
    [], dtype=int), np.array([], dtype=int)

# Extract avalanches' properties
for av_ix in range(np.size(avalanches, axis=0)):
    # Number of individual toppling events
    topplings = np.append(topplings, [int(np.count_nonzero(
        avalanches[av_ix] != grid_size)/2)], axis=0)
    # Unique sites in avalanche's list
    uniques = np.unique(avalanches[av_ix], return_index=False, axis=0)
    # Remove default initialize element
    uniques = np.delete(uniques, -1, 0)
    # Diameter, i.e. maximum distance between two sites toppled in the same avalanche
    diameter = np.append(diameter, [max_distance(uniques)], axis=0)
    # Number of different sites toppled
    sites = np.append(
        sites, [np.size(uniques, axis=0)-1], axis=0)

# Divide figure
fig, ((th, dh), (sh, hh)) = plt.subplots(2, 2)
fig.set_figheight(12)
fig.set_figwidth(12)

# Collect data and structure it in a suitable
# way for plotting and fitting
t_h, t_b = np.histogram(topplings, grid_size)
d_h, d_b = np.histogram(diameter, grid_size)
s_h, s_b = np.histogram(sites, grid_size)

t_c = 0.5 * (t_b[1:] + t_b[:-1])
d_c = 0.5 * (d_b[1:] + d_b[:-1])
s_c = 0.5 * (s_b[1:] + s_b[:-1])

# Expected scale factor
print('-------------------------------\nExpected: 0 <= y <= 1')

fit_max = int(0.3*grid_size)

# Fit topplings data
params = model.make_params(
    c=dict(value=4e3, min=3e3, max=5e3), ld=dict(value=1.8, min=1.7, max=1.9))
result = model.fit(t_h[:fit_max], params=params, x=t_c[:fit_max])
print('Normalization for topplings: ', round(result.params['c'].value, 2))
print('Topplings exponent [(2+2y)/(2+y)]: ',
      round(result.params['ld'].value, 2), '\n')
f_t = model.eval(params=result.params, x=t_c)
th.loglog(t_c, t_h, marker='.',
          linestyle=' ', color='blue', label='Data')
th.loglog(t_c, f_t, marker=' ',
          linestyle='-', color='black', label='Fit')
th.set_xlabel('Topplings')
th.set_ylabel('N° of avalanches')
th.legend()

fit_max = int(fit_max*1.5)

# Fit diameter data
params = model.make_params(
    c=dict(value=1e3, min=7e2, max=12e2), ld=dict(value=1.8, min=1.8, max=2))
result = model.fit(d_h[3:fit_max], params=params, x=d_c[3:fit_max])
print('Normalization for diameters: ', round(result.params['c'].value, 2))
print('Diameters exponent [1+y]: ', round(result.params['ld'].value, 2), '\n')
f_d = model.eval(params=result.params, x=d_c[1:])
dh.loglog(d_c[1:], d_h[1:], marker='.',
          linestyle=' ', color='red', label='Data')
dh.loglog(d_c[1:], f_d, marker=' ',
          linestyle='-', color='black', label='Fit')
dh.set_xlim(0.7, grid_size/10)
dh.set_xlabel('Diameter')
dh.set_ylabel('N° of avalanches')
dh.legend()

# Fit sites data
params = model.make_params(
    c=dict(value=6e3, min=2e3, max=8e3), ld=dict(value=1.6, min=1.5, max=1.8))
result = model.fit(s_h[3:fit_max], params=params, x=s_c[3:fit_max])
print('Normalization for sites: ', round(result.params['c'].value, 2))
print('Sites exponent [1+y/2]: ', round(result.params['ld'].value, 2), '\n')
f_s = model.eval(params=result.params, x=s_c[1:])
sh.loglog(s_c[1:], s_h[1:], marker='.',
          linestyle=' ', color='green', label='Data')
sh.loglog(s_c[1:], f_s, marker=' ',
          linestyle='-', color='black', label='Fit')
sh.set_xlim(0.4, grid_size/4)
sh.set_xlabel('Sites')
sh.set_ylabel('N° of avalanches')
sh.legend()

hh.set_visible(False)

name = datetime.datetime.now().strftime(" %b_%d - %H_%M_%S")
plt.savefig('./anl/anl-' + str(name) + '.jpg', dpi=300)
plt.show()
print('Analysis completed')
