import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from astropy.io import ascii
from scipy.ndimage.filters import gaussian_filter
import glob

import matplotlib as mpl

mpl.rc('font', family='Serif')

def get_pkl_info(filename):
    with open(filename, "r") as f:
        results = pickle.load(f)

    number_stars = filename.split('Nstars_')[1].split('_')[0]
    min_mass = filename.split('Mass_')[1].split('_')[0]
    max_mass = filename.split('Mass_')[1].split('_')[1].split('.')[0]

    fbins = np.array([res['fbin'] for res in results])
    periods = np.array([res['period'] for res in results])
    medians = np.array([res['median'] for res in results])

    return number_stars, min_mass, max_mass, periods[0], fbins, medians

def plot_grid(filename):

    number_stars, min_mass, max_mass, periods, fbins, medians = get_pkl_info(filename)

    # Make the plot
    levels = MaxNLocator(nbins=20).tick_values(medians.min(), medians.max())
    cmap = plt.get_cmap('PiYG')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    fig = plt.figure(figsize=(6, 5))
    img1 = plt.pcolormesh(np.log10(periods), fbins, medians)  # , cmap='cividis')#, norm=norm)
    cbar = plt.colorbar(img1)
    CS = plt.contour(np.log10(periods), fbins, medians,
                     [5.5, 15, 25.3, 30.9, 50.3, 66],
                     colors='w', linestyles='dashed')  # 18.0,

    stars = ['M17', 'Wd2', 'R136, NGC6611, NGC6357', 'M8, IC2944', 'IC1848', 'IC1805, NGC6231']  # 'G333',

    fmt = {}
    for l, s in zip(CS.levels, stars):
        fmt[l] = s

    plt.clabel(CS, CS.levels, inline=True, inline_spacing=3, rightside_up=True, fmt=fmt, fontsize=9)

    cbar.set_label(r'median $\rm{\sigma_{1D}}$ ($\rm{km\,s^{-1}}$)')

    plt.annotate('   IC805,\nNGC6231', xy=(0.15, 0.95), rotation=80, fontsize=9, color='w')
    plt.title('N = {} stars / Mass =[{} - {}]'.format(number_stars, min_mass, max_mass) + ' $\mathrm{M_{\odot}}$')
    plt.xlabel(r'$\rm{P_{min}}$ (log days)')
    plt.ylabel(r'$\rm{f_{bin}}$')

    return fig


if __name__ == '__main__':
    # filename = "results/grid_Pmin_fbin_Nstars_8_Mass_20_60.pkl"
    # number_stars, min_mass, max_mass, periods, fbins, medians = get_pkl_info(filename)
    # print number_stars, min_mass, max_mass, np.max(medians)
    # fig = plot_grid(filename)
    # # plt.savefig('results/median_Nstars_{}_Mass_{}_{}_contours.png'.format(number_stars, min_mass, max_mass),
    # #             bbox_inches='tight')
    # # plt.savefig('/Users/ramirez/Dropbox/KMOS/papers/KMOS_paperII/figures/median_Nstars_{}_Mass_{}_{}_contours.png'
    # #             .format(number_stars, min_mass, max_mass),
    # #             bbox_inches='tight')
    # plt.show()
    #

    data = ascii.read('data_RT20/Nstars_massRange.txt')
    name = data['name']
    minMass = data['minMass']
    maxMass = data['maxMass']
    Nstars = data['Nstars']
    sigma = data['sigma']

    pkl_list = glob.glob('results/grids/*pkl')
    pkl_data = [get_pkl_info(filename) for filename in pkl_list]
    number_stars = [int(pkl_data[i][0]) for i in range(len(pkl_data))]
    min_mass = [int(pkl_data[i][1]) for i in range(len(pkl_data))]
    max_mass = [int(pkl_data[i][2]) for i in range(len(pkl_data))]
    periods = [pkl_data[i][3] for i in range(len(pkl_data))]
    fbins = [pkl_data[i][4] for i in range(len(pkl_data))]
    medians = [pkl_data[i][5] for i in range(len(pkl_data))]


    sigma_sorted, name_sorted = [], []
    for i in range(len(number_stars)):
        for j in range(len(Nstars)):
            if (number_stars[i] == Nstars[j]):# and (minMass[j] == min_mass[j]):
                sigma_sorted.append(sigma[j])
                name_sorted.append(name[j])

    for n, s, ns in zip(name_sorted, sigma_sorted, number_stars):
        print n, s, ns

    fig = plt.figure(figsize=(4,4))
    fmt = {}
    for i in range(len(pkl_data)):
        CS = plt.contour(np.log10(pkl_data[i][3]), pkl_data[i][4], pkl_data[i][5], [sigma_sorted[i]], colors='k')
        fmt[CS.levels[0]] = name_sorted[i]

        plt.clabel(CS, CS.levels, inline=True, inline_spacing=3, rightside_up=True, fmt=fmt, fontsize=9)

    fig.savefig('results/contour_levels_individualGrids.pdf', bbox_inches='tight')
    plt.xlabel(r'$\rm{P_{min}}$ (log days)')
    plt.ylabel(r'$\rm{f_{bin}}$')
    plt.show()

    # number_stars, min_mass, max_mass, fbins, periods, medians = [], [], [], [], [], []
    # for filename in pkl_list[0:2]:
    #     print filename
    #
    #     with open(filename, "r") as f:
    #         results = pickle.load(f)
    #
    #     number_stars.append(filename.split('Nstars_')[1].split('_')[0])
    #     min_mass.append(split('Mass_')[1].split('_')[0])
    #     max_mass.append(split('Mass_')[1].split('_')[1].split('.')[0])
    #
    #     fbins.append(np.array([res['fbin'] for res in results]))
    #     periods.append(np.array([res['period'] for res in results]))
    #     medians.append(np.array([res['median'] for res in results]))

# CS = plt.contour(np.log10(periods[0]), fbins, medians,
#                  [5.5, 15, 18.0, 25., 25.3, 25.7, 30.9, 31.4, 50.3, 61, 65.5, 67.6],
#                  colors='k', linestyles='dashed')
# strs = ['M17', 'Wd2', 'G333', 'R136', 'NGC6611', 'NGC6357', 'M8', 'IC2944', 'IC1848', '61', 'IC1805', 'NGC6231']
