import matplotlib.pyplot as plt
import numpy as np
import main_RVMC as RVMC
from scipy.stats import skewnorm

from scipy.optimize import curve_fit

# This comand is to open plots in the second display
plt.switch_backend('QT4Agg')


def create_sig1D_distribution(measured_errors, number_of_samples=10 ** 5, sample_size=12,
                              number_of_stars=10 ** 5, min_mass=6, max_mass=20, binary_fraction=0.7, min_period=1.4,
                              max_period=3500, sigma_dyn=2.0):
    measured_errors = RV_errors

    RV_dist = RVMC.synthetic_RV_distribution(number_of_stars, min_mass, max_mass, binary_fraction, min_period,
                                             max_period, sigma_dyn)
    # np.random.seed(0)
    sig1D = np.std(np.random.choice(RV_dist, (number_of_samples, sample_size))
                   + np.random.normal(0, measured_errors, (number_of_samples, sample_size)), axis=1)

    return sig1D


def get_Pdistr_stats(pmin=1.4, pmax=3300, Npoints=5, bin=0.5, fbin=0.7):
    allSig1D, mode, mean, median, Allvalues, Allbins, p05, p16, p84, p95, periods = \
        [], [], [], [], [], [], [], [], [], [], []
    for period in np.logspace(np.log10(pmin), np.log10(pmax), Npoints):  # np.linspace(pmin, pmax, Npoints):#
        print 'Calulating distribution for fbin = %5.2f and Pmin = %5.2f' % (fbin, period)
        sig1D = create_sig1D_distribution(measured_errors=RV_errors, min_period=period, binary_fraction=fbin)
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
        print 'mode = %5.2f, median = %5.2f, mean = %5.2f' % (
            bins[index] + (bins[1] - bins[0]) / 2., np.median(sig1D), np.mean(sig1D))
    # plt.legend()
    # plt.show()
    plt.close()
    return allSig1D, mode, mean, median, Allvalues, Allbins, p05, p16, p84, p95, periods


def get_fbinDist_stats(fmin=0., fmax=1., Npoints=100, bin=0.5, min_period=1.4):
    allSig1D, mode, mean, median, Allvalues, Allbins, p05, p16, p84, p95, fbins = \
        [], [], [], [], [], [], [], [], [], [], []
    for fbin in np.linspace(fmin, fmax, Npoints):  #
        print 'Calulating distribution for fbin = %5.2f and Pmin = %5.2f' % (fbin, min_period)
        sig1D = create_sig1D_distribution(measured_errors=RV_errors, min_period=min_period, binary_fraction=fbin)
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
        print 'mode = %5.2f, median = %5.2f, mean = %5.2f' % (
            bins[index] + (bins[1] - bins[0]) / 2., np.median(sig1D), np.mean(sig1D))
    plt.close()
    # plt.legend()
    # plt.show()
    return allSig1D, mode, mean, median, Allvalues, Allbins, p05, p16, p84, p95, fbins


def gaussian(x, mean, amplitude, standard_deviation):
    return amplitude * np.exp(- ((x - mean) / standard_deviation) ** 2)

def find_nearest(array, value):
    n = [abs(i-value) for i in array]
    idx = n.index(min(n))
    return idx, array[idx]

def func(x, a, b, c):
    return a + (b * x) + (c * x * x) + (x * x * x)


def find_best_Pmin(mode, median, p05, p16, p84, p95, bins, values, period, usemode=True):
    diff, diff05, diff16, diff84, diff95 = 10, 10, 10, 10, 10
    index, index05, index16, index84, index95 = 0, 0, 0, 0, 0
    if usemode == True:
        median_or_mode = mode
        messagem = 'mode'
    else:
        median_or_mode = median
        messagem = 'median'

    for i, m in enumerate(median_or_mode):
        diff_temp = np.abs(m - 5.6)
        diff_temp05 = np.abs(p05[i] - 5.6)
        diff_temp16 = np.abs(p16[i] - 5.6)
        diff_temp84 = np.abs(p84[i] - 5.6)
        diff_temp95 = np.abs(p95[i] - 5.6)
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
          % (diff, index, messagem, median_or_mode[index], p16[index], p84[index])

    min_period = period[index]
    print 'Pmin = %5.2f' % (min_period)
    plt.gca().set_prop_cycle(None)

    tags = [0, index05, index16, index]
    message = ['Sana12', '05perc', '16perc', 'best']


    for i, tag in enumerate(tags):
        ax = plt.subplot(111)
        color = next(ax._get_lines.prop_cycler)['color']
        plt.step(bins[tag][:-1], values[tag], where='post', color=color,
                 label=r'$f_\mathrm{bin}$=%5.2f, %s' % (period[tag], message[i]))
        plt.hlines(0.17-((i+1)/100.), p05[tag], p95[tag], color=color)
        plt.vlines([p16[tag], p84[tag]], 0.165-((i+1)/100.), 0.175-((i+1)/100.), color=color)
        plt.vlines(median_or_mode[tag], 0.165-((i+1)/100.), 0.175-((i+1)/100.), lw=3, color=color)

    plt.xlim(0, 50)

    # plt.vlines(mode,0,0.15)
    #
    plt.vlines(5.5, 0, 0.05, 'k')
    plt.legend()
    plt.show()


def find_best_fbin(mode, median, p05, p16, p84, p95, bins, values, fbins, usemode=True):
    diff, diff05, diff16, diff84, diff95 = 10, 10, 10, 10, 10
    index, index05, index16, index84, index95 = 0, 0, 0, 0, 0
    if usemode == True:
        median_or_mode = mode
        messagem = 'mode'
    else:
        median_or_mode = median
        messagem = 'median'

    for i, m in enumerate(median_or_mode):
        diff_temp = np.abs(m - 5.6)
        diff_temp05 = np.abs(p05[i] - 5.6)
        diff_temp16 = np.abs(p16[i] - 5.6)
        diff_temp84 = np.abs(p84[i] - 5.6)
        diff_temp95 = np.abs(p95[i] - 5.6)
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
          % (diff, index, messagem, median_or_mode[index], p16[index], p84[index])

    print 'fbin = %5.2f' % (fbins[index])
    plt.gca().set_prop_cycle(None)

    index_fbin70, fbinclosest70 = find_nearest(fbins,0.70)

    tags = [index16, index, index05, index_fbin70]
    message = ['16perc', 'best', '05perc', 'Sana12']

    for i, tag in enumerate(tags):
        ax = plt.subplot(111)
        color = next(ax._get_lines.prop_cycler)['color']
        plt.step(bins[tag][:-1], values[tag], where='post', color=color,
                 label=r'$P_\mathrm{min}$=%5.2f, %s' % (fbins[tag], message[i]))
        plt.hlines(0.06+((i+1)/100.), p05[tag], p95[tag], color=color)
        plt.vlines([p16[tag], p84[tag]], 0.055+((i+1)/100.), 0.065+((i+1)/100.), color=color)
        plt.vlines(median_or_mode[tag], 0.055+((i+1)/100.), 0.065+((i+1)/100.), lw=3, color=color)

    plt.xlim(0, 50)

    # plt.vlines(mode,0,0.15)
    #
    plt.vlines(5.6, 0, 0.05, 'k')
    plt.axvspan(5.1, 6.1, ymin=0, ymax=0.1, color='k', alpha=0.5, lw=0)

    plt.legend()
    plt.show()


RV_errors = np.array([0.8, 2.3, 1.0, 1.8, 0.8, 1.0, 1.4, 2.5, 2.0, 0.4, 1.5, 0.6])

allSig1D, mode, mean, median, values, bins, p05, p16, p84, p95, period = get_Pdistr_stats(pmin=1.4, pmax=3300,
                                                                                          Npoints=100,
                                                                                          bin=0.5)

find_best_Pmin(mode, median, p05, p16, p84, p95, bins, values, period, usemode=False)

# allSig1D, mode, mean, median, values, bins, p05, p16, p84, p95, fbins = get_fbinDist_stats(fmin=0.00, fmax=1., Npoints=100,
#                                                                                  bin=0.5)
#
# find_best_fbin(mode, median, p05, p16, p84, p95, bins, values, fbins, usemode=False)

# for i in range(len(bins)):
#     bin_centers = bins[i][:-1] + np.diff(bins[i]) / 2
#     a, b, c = skewnorm.fit(allSig1D[i],  5, loc=mode[i][0] , scale=1.)
#     x_interval_for_fit = np.linspace(bins[i][0], bins[i][-1], 10000)
#     print a, b, c
#     y_fit = skewnorm.pdf(x_interval_for_fit, a,b,c)
#     plt.plot(x_interval_for_fit, y_fit, '--')
#
# plt.gca().set_prop_cycle(None)

# for i in range(len(bins)):
#     bin_centers = bins[i][:-1] + np.diff(bins[i]) / 2
#     z = np.polyfit(bin_centers, values[i],5)
#     p = np.poly1d(z)
#
#     x_interval_for_fit = np.linspace(bins[i][0], bins[i][-1], 10000)
#     y_fit = p(x_interval_for_fit)
#
#     plt.plot(x_interval_for_fit, y_fit, ':')
#     print 'max_fit = %5.2f' % (x_interval_for_fit[y_fit == max(y_fit)])


#
# for i in range(len(bins)):
#     bin_centers = bins[i][:-1] + np.diff(bins[i]) / 2
#
#     popt, _ = curve_fit(gaussian, bin_centers, values[i], p0=[mode[i][0], 0., 1.])
#
#     x_interval_for_fit = np.linspace(bins[i][0], bins[i][-1], 10000)
#     y_fit = gaussian(x_interval_for_fit, *popt)
#     # plt.plot(x_interval_for_fit, y_fit)
#     # print 'max_fit = %5.2f' % (x_interval_for_fit[y_fit == max(y_fit)])
#
# plt.gca().set_prop_cycle(None)
