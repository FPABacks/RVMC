import matplotlib.pyplot as plt
import numpy as np
import main_RVMC as RVMC
from astropy.io import ascii
import pickle
import os
from scipy.stats import skewnorm

from scipy.optimize import curve_fit

# This comand is to open plots in the second display
plt.switch_backend('QT4Agg')


def create_sig1D_distribution(measured_errors=np.zeros(12), number_of_samples=10 ** 5, sample_size=12,
                              number_of_stars=10 ** 5, min_mass=6, max_mass=20, binary_fraction=0.7, min_period=1.4,
                              max_period=3500, sigma_dyn=2.0):
    # measured_errors = RV_errors

    RV_dist = RVMC.synthetic_RV_distribution(number_of_stars, min_mass, max_mass, binary_fraction, min_period,
                                             max_period, sigma_dyn)
    # np.random.seed(0)
    sig1D = np.std(np.random.choice(RV_dist, (number_of_samples, sample_size))
                   + np.random.normal(0, measured_errors, (number_of_samples, sample_size)), axis=1)

    return sig1D


def get_Pdistr_stats(measured_errors=np.zeros(12), sample_size=12, pmin=1.4, pmax=3300, Npoints=5, bin=0.5, fbin=0.7):
    allSig1D, mode, mean, median, Allvalues, Allbins, p05, p16, p84, p95, periods = \
        [], [], [], [], [], [], [], [], [], [], []

    for period in np.logspace(np.log10(pmin), np.log10(pmax), Npoints):  # np.linspace(pmin, pmax, Npoints):#
        print '\n --------------------------------------------- \n' \
              'Calulating distribution for fbin = %5.2f and Pmin = %5.2f' % (fbin, period)

        sig1D = create_sig1D_distribution(measured_errors=measured_errors, sample_size=sample_size,
                                          min_period=period, binary_fraction=fbin)

        values, bins, patches = plt.hist(sig1D, histtype="step", bins=np.arange(0, max(sig1D) + bin, bin),
                                         density=True, label=r'$P_{min}=$ %5.2f' % (period))
        index = np.where(values == np.max(values))
        if len(index[0]) > 1:
            index = index[0][0]
        mode.append(bins[index] + (bins[1] - bins[0]) / 2.)

        allSig1D.append(sig1D)
        mean.append(np.mean(sig1D))
        median.append(np.median(sig1D))
        p05.append(np.percentile(sig1D, 5))
        p16.append(np.percentile(sig1D, 16))
        p84.append(np.percentile(sig1D, 84))
        p95.append(np.percentile(sig1D, 95))
        periods.append(period)

        Allvalues.append(values)
        Allbins.append(bins)
        print 'mode = %5.2f, median = %5.2f, mean = %5.2f, p84 = %5.2f, p16 = %5.2f' % (
            bins[index] + (bins[1] - bins[0]) / 2., np.median(sig1D), np.mean(sig1D),
            np.percentile(sig1D, 84), np.percentile(sig1D, 16))
    # plt.legend()
    # plt.show()
    plt.close()
    return allSig1D, mode, mean, median, Allvalues, Allbins, p05, p16, p84, p95, periods


def get_fbinDist_stats(measured_errors=np.zeros(12), sample_size=12, fmin=0., fmax=1., Npoints=100, bin=0.5,
                       min_period=1.4):
    allSig1D, mode, mean, median, Allvalues, Allbins, p05, p16, p84, p95, fbins = \
        [], [], [], [], [], [], [], [], [], [], []
    for fbin in np.linspace(fmin, fmax, Npoints):  #
        print '\n --------------------------------------------- \n' \
              'Calulating distribution for fbin = %5.2f and Pmin = %5.2f' % (fbin, min_period)

        sig1D = create_sig1D_distribution(sample_size=sample_size, measured_errors=measured_errors,
                                          min_period=min_period, binary_fraction=fbin)

        values, bins, patches = plt.hist(sig1D, histtype="step", bins=np.arange(0, max(sig1D) + bin, bin),
                                         density=True, label=r'$f_{bin}=$ %5.2f' % (fbin))
        index = np.where(values == np.max(values))
        if len(index[0]) > 1:
            index = index[0][0]
        mode.append(bins[index] + (bins[1] - bins[0]) / 2.)

        allSig1D.append(sig1D)
        mean.append(np.mean(sig1D))
        median.append(np.median(sig1D))
        p05.append(np.percentile(sig1D, 5))
        p16.append(np.percentile(sig1D, 16))
        p84.append(np.percentile(sig1D, 84))
        p95.append(np.percentile(sig1D, 95))
        fbins.append(fbin)

        Allvalues.append(values)
        Allbins.append(bins)
        print 'mode = %5.2f, median = %5.2f, mean = %5.2f, p84 = %5.2f, p16 = %5.2f' % (
            bins[index] + (bins[1] - bins[0]) / 2., np.median(sig1D), np.mean(sig1D),
            np.percentile(sig1D, 84), np.percentile(sig1D, 16))
    plt.close()
    # plt.legend()
    # plt.show()
    return allSig1D, mode, mean, median, Allvalues, Allbins, p05, p16, p84, p95, fbins


def gaussian(x, mean, amplitude, standard_deviation):
    return amplitude * np.exp(- ((x - mean) / standard_deviation) ** 2)


def find_nearest(array, value):
    n = [abs(i - value) for i in array]
    idx = n.index(min(n))
    return idx, array[idx]


def func(x, a, b, c):
    return a + (b * x) + (c * x * x) + (x * x * x)


def find_best_Pmin(observed_sigma, sample_size, mode, median, p05, p16, p84, p95, bins, values, period, usemode=False):
    diff, diff05, diff16, diff84, diff95 = 100, 100, 100, 100, 100
    index, index05, index16, index84, index95 = 0, 0, 0, 0, 0
    if usemode == True:
        median_or_mode = mode
        messagem = 'mode'
    else:
        median_or_mode = median
        messagem = 'median'

    for i, m in enumerate(median_or_mode):
        diff_temp = np.abs(m - observed_sigma)
        diff_temp05 = np.abs(p05[i] - observed_sigma)
        diff_temp16 = np.abs(p16[i] - observed_sigma)
        diff_temp84 = np.abs(p84[i] - observed_sigma)
        diff_temp95 = np.abs(p95[i] - observed_sigma)
        if diff_temp < diff:
            diff = diff_temp
            index = i
        if diff_temp05 < diff05:
            diff05 = diff_temp05
            index05 = i
        if diff_temp16 < diff16:
            diff16 = diff_temp16
            index16 = i
        if diff_temp84 < diff84:
            diff84 = diff_temp84
            index84 = i
        if diff_temp95 < diff95:
            diff95 = diff_temp95
            index95 = i
            # print index, diff, m
    print 'diff= %5.2f, i = %3i, %s = %5.2f, p16 = %5.2f, p84 = %5.2f' \
          % (diff, index, messagem, median_or_mode[index], p16[index16], p84[index84])

    min_period = period[index]
    print 'Pmin = %5.2f' % (min_period)
    plt.gca().set_prop_cycle(None)

    tags = [index05, index16, index, index84, index95, 0]
    message = ['05perc', '16perc', 'best', '84perc', '95perc', 'Sana12']

    for i, tag in enumerate(tags):
        ax = plt.subplot(111)
        color = next(ax._get_lines.prop_cycler)['color']
        plt.step(bins[tag][:-1], values[tag], where='post', color=color,
                 label=r'$P_{min}$=%5.2f, %s' % (period[tag], message[i]))
        plt.hlines(0.17 - ((i + 1) / 100.), p05[tag], p95[tag], color=color)
        plt.vlines([p16[tag], p84[tag]], 0.165 - ((i + 1) / 100.), 0.175 - ((i + 1) / 100.), color=color)
        plt.vlines(median_or_mode[tag], 0.165 - ((i + 1) / 100.), 0.175 - ((i + 1) / 100.), lw=3, color=color)

    plt.xlim(0, 70)
    plt.ylim(0, 0.25)

    # plt.vlines(mode,0,0.15)
    #
    plt.vlines(observed_sigma, 0, 0.1, 'k')
    # plt.axvline(5.6, 'k')
    plt.axvspan(observed_sigma - 0.5, observed_sigma + 0.5, ymin=0, ymax=0.4, color='k', alpha=0.5, lw=0)
    # plt.axvspan(5.1, 6.1, color='k', alpha=0.5, lw=0)

    plt.xlabel(r'${\sigma_{1D}}$ ($\rm km\,s^{-1}$)')
    plt.ylabel('Frequency')

    plt.legend()
    plt.savefig('results/Pmin_hist_ObsSig_' + str(observed_sigma) + '_Nstars_' + str(sample_size) + '.pdf')

    # plt.show()
    ascii.write(np.array([period[index], period[index05], period[index16], period[index84], period[index95]]),
                'results/Pmin_ObsSig_' + str(observed_sigma) + '_Nstars_' + str(sample_size) + '.dat',
                names=['best_Pmin', '0.05perc_Pmin', '0.16perc_Pmin', '0.84perc_Pmin', '0.95perc_Pmin'],
                overwrite=True)


def find_best_fbin(observed_sigma, sample_size, mode, median, p05, p16, p84, p95, bins, values, fbins, usemode=False):
    diff, diff05, diff16, diff84, diff95 = 100, 100, 100, 100, 100
    index, index05, index16, index84, index95 = 0, 0, 0, 0, 0
    if usemode == True:
        median_or_mode = mode
        messagem = 'mode'
    else:
        median_or_mode = median
        messagem = 'median'

    for i, m in enumerate(median_or_mode):
        diff_temp = np.abs(m - observed_sigma)
        diff_temp05 = np.abs(p05[i] - observed_sigma)
        diff_temp16 = np.abs(p16[i] - observed_sigma)
        diff_temp84 = np.abs(p84[i] - observed_sigma)
        diff_temp95 = np.abs(p95[i] - observed_sigma)
        if diff_temp < diff:
            diff = diff_temp
            index = i
        if diff_temp05 < diff05:
            diff05 = diff_temp05
            index05 = i
        if diff_temp16 < diff16:
            diff16 = diff_temp16
            index16 = i
        if diff_temp84 < diff84:
            diff84 = diff_temp84
            index84 = i
        if diff_temp95 < diff95:
            diff95 = diff_temp95
            index95 = i
            # print index, diff, m
    print 'diff= %5.2f, i = %3i, %s = %5.2f, p16 = %5.2f, p84 = %5.2f' \
          % (diff, index, messagem, median_or_mode[index], p84[index84], p16[index16])

    print 'fbin = %5.2f' % (fbins[index])
    plt.gca().set_prop_cycle(None)

    index_fbin70, fbinclosest70 = find_nearest(fbins, 0.70)

    tags = [index05, index16, index, index84, index95, index_fbin70]
    message = ['05perc', '16perc', 'best', '84perc', '95perc', 'Sana12']

    for i, tag in enumerate(tags):
        ax = plt.subplot(111)
        color = next(ax._get_lines.prop_cycler)['color']
        plt.step(bins[tag][:-1], values[tag], where='post', color=color,
                 label=r'$f_\mathrm{bin}$=%5.2f, %s' % (fbins[tag], message[i]))
        plt.hlines(0.06 + ((i + 1) / 100.), p05[tag], p95[tag], color=color)
        plt.vlines([p16[tag], p84[tag]], 0.055 + ((i + 1) / 100.), 0.065 + ((i + 1) / 100.), color=color)
        plt.vlines(median_or_mode[tag], 0.055 + ((i + 1) / 100.), 0.065 + ((i + 1) / 100.), lw=3, color=color)

    plt.xlim(0, 70)
    plt.ylim(0, 0.25)

    # plt.vlines(mode,0,0.15)
    #
    plt.vlines(observed_sigma, 0, 0.1, 'k')
    plt.axvspan(observed_sigma - 0.5, observed_sigma + 0.5, ymin=0, ymax=0.4, color='k', alpha=0.5, lw=0)
    # plt.axvspan(5.1, 6.1, ymin=0, ymax=0.1, color='k', alpha=0.5, lw=0)
    # plt.axvspan(5.1, 6.1, color='k', alpha=0.5, lw=0)

    plt.xlabel(r'${\sigma_{1D}}$ ($\rm km\,s^{-1}$)')
    plt.ylabel('Frequency')

    plt.legend()

    plt.savefig('results/fbin_hist_ObsSig_' + str(observed_sigma) + '_Nstars_' + str(sample_size) + '.pdf')
    # plt.show()

    ascii.write(np.array([fbins[index], fbins[index05], fbins[index16], fbins[index84], fbins[index95]]),
                'results/fbin_ObsSig_' + str(observed_sigma) + '_Nstars_' + str(sample_size) + '.dat',
                names=['best_fbin', '0.05perc_fbin', '0.16perc_fbin', '0.84perc_fbin', '0.95perc_fbin'],
                overwrite=True)

    return observed_sigma


def find_Pmin_fbin_individual_clulsters(name_cluster, observed_dispersion, number_stars):
    for n, observed_dispersion, number_stars in zip(name_cluster, observed_dispersion, number_stars):
        print '\n Cluster:', n, '\n'

        RV_errors = np.ones(number_stars) * 1.2

        allSig1D, mode, mean, median, values, bins, p05, p16, p84, p95, period = get_Pdistr_stats(
            measured_errors=RV_errors,
            sample_size=number_stars,
            pmin=1.4, pmax=3500,
            Npoints=100,
            bin=0.5, fbin=0.7)

        find_best_Pmin(observed_dispersion, number_stars, mode, median, p05, p16, p84, p95, bins, values, period,
                       usemode=False)

        allSig1D, mode, mean, median, values, bins, p05, p16, p84, p95, fbins = get_fbinDist_stats(
            measured_errors=RV_errors,
            sample_size=number_stars,
            fmin=0.00, fmax=1.,
            Npoints=200, bin=0.5)

        find_best_fbin(observed_dispersion, number_stars, mode, median, p05, p16, p84, p95, bins, values, fbins,
                       usemode=False)


if __name__ == '__main__':
    name_cluster = ['IC1805', 'IC1848', 'IC2944', 'NGC6231', 'NGC6611', 'Wd2', 'M17', 'M8', 'NGC6357', 'G333', 'R136']
    observed_dispersion = [65.45329203402903, 50.26275741127991, 31.357122212648363, 67.61903406110999,
                           25.32524131575383, 14.999765386529885, 5.5, 30.94, 25.73, 18.04, 25.0]
    number_stars = [8, 5, 14, 13, 9, 44, 12, 22, 30, 8, 332]
    # RV_errors = np.array([0.8, 2.3, 1.0, 1.8, 0.8, 1.0, 1.4, 2.5, 2.0, 0.4, 1.5, 0.6]) # for M17
    find_Pmin_fbin_individual_clulsters(name_cluster, observed_dispersion, number_stars)

##################### Edit Tim ###############################

# name_cluster = ['IC1805', 'IC1848', 'IC2944', 'NGC6231', 'NGC6611', 'Wd2', 'M17', 'M8', 'NGC6357', 'G333', 'R136']
#  observed_dispersion = 5.5#[65.45329203402903, 50.26275741127991, 31.357122212648363, 67.61903406110999,
#                         #25.32524131575383, 14.999765386529885, 5.5, 30.94, 25.73, 18.04, 25.0]
#  number_stars = 12#[8, 5, 14, 13, 9, 44, 12, 22, 30, 8, 332]
#  # RV_errors = np.array([0.8, 2.3, 1.0, 1.8, 0.8, 1.0, 1.4, 2.5, 2.0, 0.4, 1.5, 0.6]) # for M17
#  # for n, observed_dispersion, number_stars in zip(name_cluster,observed_dispersion,number_stars):
#
#  # print '\n Cluster:', n, '\n'
#
#  RV_errors = np.ones(number_stars)*1.2
#
#  output_name = "myres.pkl"
#  l_res = []
#  fbins = np.linspace(0, 1, 10)
#  for fbin in fbins:
#      allSig1D, mode, mean, median, values, bins, p05, p16, p84, p95, period = get_Pdistr_stats(measured_errors=RV_errors,
#                                                                                                    sample_size=number_stars,
#                                                                                                    pmin=1.4, pmax=3500,
#                                                                                                    Npoints=2,
#                                                                                                    bin=0.5, fbin=0.7)
#
#      result = dict()
#      result['values'] = values
#      result['bins'] = bins
#      result['period'] = period
#      result['fbin'] = fbin
#      result['mean'] = mean
#      result['median'] = median
#      result['mode'] = mode
#      result['p05'] = p05
#      result['p16'] = p16
#      result['p84'] = p84
#      result['p95'] = p95
#
#      l_res.append(result)
#  with open(output_name, "w") as f:
#      pickle.dump(l_res, f)
#  if os.path.exists(output_name):
#      print "{output_name} created.".format(output_name=output_name)
#
#      # find_best_Pmin(observed_dispersion, number_stars, mode, median, p05, p16, p84, p95, bins, values, period,
#      #                usemode=False)
#      #
#      # allSig1D, mode, mean, median, values, bins, p05, p16, p84, p95, fbins = get_fbinDist_stats(
#      #     measured_errors=RV_errors,
#      #     sample_size=number_stars,
#      #     fmin=0.00, fmax=1.,
#      #     Npoints=200, bin=0.5)
#      #
#      # find_best_fbin(observed_dispersion, number_stars, mode, median, p05, p16, p84, p95, bins, values, fbins,
#      #                usemode=False)