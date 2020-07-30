
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from scipy.ndimage.filters import gaussian_filter



filename = "results/grid_Pmin_fbin_Nstars_20_Mass_6_20.pkl"
# filename = "results/grid_Pmin_fbin_Nstars_20_Mass_15_60.pkl"
with open(filename, "r") as f:
    results = pickle.load(f)

number_stars = filename.split('Nstars_')[1].split('_')[0]
min_mass = filename.split('Mass_')[1].split('_')[0]
max_mass = filename.split('Mass_')[1].split('_')[1].split('.')[0]

fbins = np.array([res['fbin'] for res in results])
periods = np.array([res['period'] for res in results])
medians = np.array([res['median'] for res in results])

# Make the plot
levels = MaxNLocator(nbins=20).tick_values(medians.min(), medians.max())
cmap = plt.get_cmap('PiYG')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

img1 = plt.pcolormesh(np.log10(periods[0]), fbins, medians)#, norm=norm)
cbar = plt.colorbar(img1)
# CS = plt.contour(np.log10(periods[0]), fbins, medians,
#                  [5.5, 15, 18.0, 25., 25.3, 25.7, 30.9, 31.4, 50.3, 61, 65.5, 67.6],
#                  colors='k', linestyles='dashed')
# strs = ['M17', 'Wd2', 'G333', 'R136', 'NGC6611', 'NGC6357', 'M8', 'IC2944', 'IC1848', '61', 'IC1805', 'NGC6231']

CS = plt.contour(np.log10(periods[0]), fbins, medians,
                 [5.5, 15, 18.0, 25.3, 30.9, 50.3, 66],
                 colors='k', linestyles='dashed')
strs = ['M17', 'Wd2', 'G333', 'R136, NGC6611, NGC6357', 'M8, IC2944', 'IC1848', 'IC1805, NGC6231']

fmt = {}
for l, s in zip(CS.levels, strs):
    fmt[l] = s

plt.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=10)

cbar.set_label(r'median $\rm{\sigma_{1D}}$ ($\rm{km\,s^{-1}}$)')

plt.title(r'N = {} stars / Mass =[{} - {}] Msun'.format(number_stars, min_mass, max_mass))
plt.xlabel(r'$\rm{P_{min}}$ (log days)')
plt.ylabel(r'$\rm{f_{bin}}$')
plt.savefig('results/median_Nstars_{}_Mass_{}_{}_contours.png'.format(number_stars, min_mass, max_mass))
plt.show()

# print(type(res))
# print(res['fbin'])

# periods = res['period']


# print(periods)
