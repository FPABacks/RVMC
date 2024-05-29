# NOTE THIS FILE IS OUTDATED!
# IT DOES NOT WORK WITH CURRENT VERSIONS
# (IT ALSO USES PYTHON 2)
# JUST LEFT IT IN CASE IT CAN BE USED TO INSPIRE THE NEW SCRIPT

import matplotlib.pyplot as plt
import numpy as np
import main_RVMC as RVMC
from astropy.io import ascii
import glob
import time
import os
from scipy.stats import skewnorm
import pickle
from scipy import odr

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


def get_Pdistr_stats(measured_errors=np.zeros(12), sample_size=12, pmin=1.4, pmax=3300, Npoints=5, bin=0.5,
                     fbin=0.7, min_mass=6, max_mass=20):
    allSig1D, mode, mean, median, Allvalues, Allbins, p05, p16, p84, p95, periods = \
        [], [], [], [], [], [], [], [], [], [], []

    for period in np.logspace(np.log10(pmin), np.log10(pmax), Npoints):  # np.linspace(pmin, pmax, Npoints):#
        print '\n --------------------------------------------- \n' \
              'Calulating distribution for fbin = %5.2f and Pmin = %5.2f' % (fbin, period)

        sig1D = create_sig1D_distribution(measured_errors=measured_errors, sample_size=sample_size,
                                          min_period=period, binary_fraction=fbin, min_mass=min_mass, max_mass=max_mass)

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
                       min_period=1.4, min_mass=6, max_mass=20):
    allSig1D, mode, mean, median, Allvalues, Allbins, p05, p16, p84, p95, fbins = \
        [], [], [], [], [], [], [], [], [], [], []
    for fbin in np.linspace(fmin, fmax, Npoints):  #
        print '\n --------------------------------------------- \n' \
              'Calulating distribution for fbin = %5.2f and Pmin = %5.2f' % (fbin, min_period)

        sig1D = create_sig1D_distribution(sample_size=sample_size, measured_errors=measured_errors,
                                          min_period=min_period, binary_fraction=fbin, min_mass=min_mass,
                                          max_mass=max_mass)

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


def find_best_Pmin(observed_sigma, min_mass, max_mass, sample_size, mode, median, p05, p16, p84, p95, bins, values,
                   period, usemode=False):
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
    plt.savefig('results/Pmin_hist_ObsSig_' + str(observed_sigma) + '_Nstars_' + str(sample_size) +
                '_Mass_' + str(min_mass) + '_' + str(max_mass) + '.pdf')

    # plt.show()
    ascii.write(np.array([period[index], period[index05], period[index16], period[index84], period[index95]]),
                'results/Pmin_ObsSig_' + str(observed_sigma) + '_Nstars_' + str(sample_size) +
                '_Mass_' + str(min_mass) + '_' + str(max_mass) + '.dat',
                names=['best_Pmin', '0.05perc_Pmin', '0.16perc_Pmin', '0.84perc_Pmin', '0.95perc_Pmin'],
                overwrite=True)


def find_best_fbin(observed_sigma, min_mass, max_mass, sample_size, mode, median, p05, p16, p84, p95, bins, values,
                   fbins, usemode=False):
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

    plt.savefig('results/fbin_hist_ObsSig_' + str(observed_sigma) + '_Nstars_' + str(sample_size) +
                '_Mass_' + str(min_mass) + '_' + str(max_mass) + '.pdf')
    # plt.show()

    ascii.write(np.array([fbins[index], fbins[index05], fbins[index16], fbins[index84], fbins[index95]]),
                'results/fbin_ObsSig_' + str(observed_sigma) + '_Nstars_' + str(sample_size) +
                '_Mass_' + str(min_mass) + '_' + str(max_mass) + '.dat',
                names=['best_fbin', '0.05perc_fbin', '0.16perc_fbin', '0.84perc_fbin', '0.95perc_fbin'],
                overwrite=True)

    return observed_sigma


def find_Pmin_fbin_individual_clulsters(name_cluster, observed_dispersion, number_stars, min_mass, max_mass):
    for n, observed_dispersion, number_stars, min_mass, max_mass in \
            zip(name_cluster, observed_dispersion, number_stars, min_mass, max_mass):

        print '\n Cluster:', n, '\n'

        if n == 'M17':
            RV_errors = np.array([0.8, 2.3, 1.0, 1.8, 0.8, 1.0, 1.4, 2.5, 2.0, 0.4, 1.5, 0.6])  # for M17
        else:
            RV_errors = np.ones(number_stars) * 1.2

        allSig1D, mode, mean, median, values, bins, p05, p16, p84, p95, period = get_Pdistr_stats(
            measured_errors=RV_errors,
            sample_size=number_stars,
            pmin=1.4, pmax=3500,
            Npoints=100,
            bin=0.5, fbin=0.7, min_mass=min_mass, max_mass=max_mass)

        find_best_Pmin(observed_dispersion, min_mass, max_mass, number_stars, mode, median,
                       p05, p16, p84, p95, bins, values, period, usemode=False)

        allSig1D, mode, mean, median, values, bins, p05, p16, p84, p95, fbins = get_fbinDist_stats(
            measured_errors=RV_errors,
            sample_size=number_stars,
            fmin=0.00, fmax=1.,
            Npoints=200, bin=0.5, min_mass=min_mass, max_mass=max_mass)

        find_best_fbin(observed_dispersion, min_mass, max_mass, number_stars, mode, median,
                       p05, p16, p84, p95, bins, values, fbins, usemode=False)


def exp_func(p, x):
    a, b = p
    y = a * np.exp(b * x) + c
    return y


def quad_func(p, x):
    a, b, c = p
    return a * np.exp(- x / b) + c
    # return (a * x) + b


def plot_Pmin_vs_age(fit=False):
    import matplotlib as mpl
    mpl.rc('font', family='Serif')

    Pmin_filelist = glob.glob('results/Pmin_ObsSig_*_Nstars_*.dat')
    Pmin = [ascii.read(f)['best_Pmin'] for f in Pmin_filelist]
    Pmin016 = [ascii.read(f)['0.16perc_Pmin'] for f in Pmin_filelist]
    Pmin084 = [ascii.read(f)['0.84perc_Pmin'] for f in Pmin_filelist]

    sig1D_fromfilename = np.array([float(f.split('ObsSig_')[1].split('_Nstars')[0]) for f in Pmin_filelist])
    sig1D_fromdata = ascii.read('data_RT20/weighted_RVdisp_vs_age.txt', delimiter=';')['sig1D']
    age_fromdata = ascii.read('data_RT20/weighted_RVdisp_vs_age.txt', delimiter=';')['age']
    age_err_fromdata = ascii.read('data_RT20/weighted_RVdisp_vs_age.txt', delimiter=';')['age_err']
    name_fromdata = ascii.read('data_RT20/weighted_RVdisp_vs_age.txt', delimiter=';')['name']

    age_sorted = []
    age_err_sorted = []
    sig1D_sorted = []
    Pmin_sorted = []
    Pmin016_sorted = []
    Pmin084_sorted = []
    name_sorted = []
    for i in range(len(sig1D_fromfilename)):
        for j in range(len(sig1D_fromdata)):
            if round(sig1D_fromfilename[i], 2) == round(sig1D_fromdata[j], 2):
                name_sorted.append(name_fromdata[j])
                age_sorted.append(age_fromdata[j])
                age_err_sorted.append(age_err_fromdata[j])
                sig1D_sorted.append(sig1D_fromdata[j])
                Pmin_sorted.append(Pmin[i].data[0])
                Pmin016_sorted.append(Pmin016[i].data[0])
                Pmin084_sorted.append(Pmin084[i].data[0])

    age_sorted = np.array(age_sorted)
    age_err_sorted = np.array(age_err_sorted)
    sig1D_sorted = np.array(sig1D_sorted)
    Pmin_sorted = np.array(Pmin_sorted)
    Pmin016_sorted = np.array(Pmin016_sorted)
    Pmin084_sorted = np.array(Pmin084_sorted)

    Pmin_errs = [Pmin_sorted - Pmin016_sorted, Pmin084_sorted - Pmin_sorted]

    ascii.write([name_sorted, age_sorted, age_err_sorted, Pmin_sorted, Pmin_errs[0], Pmin_errs[1]],
                'll',
                names=['name', 'age', 'age_err', 'Pmin', 'Pmin_min', 'Pmin_plus'])

    fig = plt.figure(figsize=(4, 4))

    if fit == True:
        Pmin_errs_fit = np.abs(Pmin_errs[0] - Pmin_errs[1])
        realdata = odr.RealData(np.array(age_sorted), np.array(Pmin_sorted), sx=np.array(age_err_sorted),
                                sy=np.array(Pmin_errs_fit))

        data = odr.Data(age_sorted, Pmin_sorted)
        # Create a model for fitting.
        quad_model = odr.Model(quad_func)
        odr_obj = odr.ODR(realdata, quad_model, beta0=[1e5, 0.2, 1.3], ifixb=[0., 1., 1.])  # beta0=[1., 1., 1.])  #
        out = odr_obj.run()
        popt = out.beta
        perr = out.sd_beta

        odr_obj1 = odr.ODR(realdata, quad_model, beta0=[1e4, 0.2, 1.3], ifixb=[0., 1., 1.])  # beta0=[1., 1., 1.])  #
        out1 = odr_obj1.run()
        popt1 = out1.beta
        perr1 = out1.sd_beta

        odr_obj2 = odr.ODR(realdata, quad_model, beta0=[1e6, 0.2, 1.3], ifixb=[0., 1., 1.])  # beta0=[1., 1., 1.])  #
        out2 = odr_obj2.run()
        popt2 = out2.beta
        perr2 = out2.sd_beta

        # prepare confidence level curves
        nstd = 1.  # to draw 5-sigma intervals
        popt_up = popt + nstd * np.array([0, 0.0394, 0.0713])  # perr
        popt_dw = popt - nstd * np.array([0, 0.0689, 0.038])  # perr
        # popt_dw[0] = popt_dw[0]*-1.

        print out.beta, out.sd_beta
        # print popt_dw, popt_up

        x_fit = np.linspace(0., 6, 1000)
        # y_fit1 = 1.9e+05*np.exp(x_fit * -5.4) + 1.4
        y_fit = quad_func(out.beta, x_fit)
        fit_up = quad_func(popt_up, x_fit)
        fit_dw = quad_func(popt_dw, x_fit)

        y_fit1 = quad_func(out1.beta, x_fit)
        y_fit2 = quad_func(out2.beta, x_fit)

        print out.beta, out1.beta, out2.beta, '\n', '\n', out.beta - out1.beta, -out.beta + out2.beta

        plt.plot(x_fit[y_fit < 1e4], y_fit[y_fit < 1e4], '#1f77b4',
                 label=r'$\mathrm{P_{min}}$' + '(t)=$10^{(5\pm1)}$' +
                       '$exp({-t/({%.2f}_{-%.2f}^{+%.2f})}) + (%.2f_{%.2f}^{+%.2f})$' % (round(out.beta[1], 2),
                                                                                round(out.beta[1]-out2.beta[1], 2),
                                                                                round(out1.beta[1] - out.beta[1], 2),
                                                                                round(out.beta[2], 2),
                                                                                round(out1.beta[2] - out.beta[2], 2),
                                                                                round(out2.beta[2] - out.beta[2], 2)))

        # plt.plot(x_fit, fit_up, 'r')
        # plt.plot(x_fit, fit_dw, 'r')
        plt.fill_between(x_fit, fit_up, fit_dw, alpha=.25)
        # plt.fill_between(x_fit, y_fit1, y_fit2, alpha=.25)
        # plt.plot(x_fit, y_fit1, 'r')

    plt.errorbar(age_sorted, Pmin_sorted, xerr=age_err_sorted,
                 yerr=Pmin_errs, fmt='.', color='grey', linewidth=2)

    for n, x, y, xer, yer1, yer2 in zip(name_sorted, age_sorted, Pmin_sorted, age_err_sorted, Pmin_errs[0],
                                        Pmin_errs[1]):
        if n == 'NGC6357':
            plt.annotate(n, xy=(x - 1.7, y + 0.1), color='k', fontsize=9)
        elif n == 'IC1848':
            plt.annotate(n, xy=(x, y + 2), color='k', fontsize=9, rotation=45)
        else:
            plt.annotate(n, xy=(x, y + 0.1), color='k', fontsize=9)

    plt.yscale('log')
    plt.ylim(1, 3e4)
    # plt.xlim(-0.5, 8)
    plt.xlabel('age (Myr)')
    plt.ylabel(r'$\rm{P_{min}}$ (log days)')
    plt.legend(fontsize=8.6, loc=1)
    fig.savefig('/Users/ramirez/Dropbox/KMOS/papers/KMOS_paperII/figures/Pcutoff_vs_age_fixP0.pdf', bbox_inches='tight')
    # fig.savefig('results/Pcutoff_vs_age.pdf', bbox_inches='tight')
    plt.show()


def run_bigGrid(number_stars=20, min_mass=6, max_mass=20):
    output_name = "results/grid_Pmin_fbin_Nstars_{}_Mass_{}_{}.pkl".format(number_stars, min_mass, max_mass)
    l_res = []
    fbins = np.linspace(0, 1, 100)

    RV_errors = np.ones(number_stars) * 1.2

    for fbin in fbins:
        allSig1D, mode, mean, median, values, bins, p05, p16, p84, p95, period = get_Pdistr_stats(
            measured_errors=RV_errors,
            sample_size=number_stars,
            pmin=1.4, pmax=3500,
            Npoints=100,
            bin=0.5, fbin=fbin, min_mass=min_mass, max_mass=max_mass)

        result = dict()
        result['values'] = values
        result['bins'] = bins
        result['period'] = period
        result['fbin'] = fbin
        result['mean'] = mean
        result['median'] = median
        result['mode'] = mode
        result['p05'] = p05
        result['p16'] = p16
        result['p84'] = p84
        result['p95'] = p95

        l_res.append(result)
    with open(output_name, "w") as f:
        pickle.dump(l_res, f)
    if os.path.exists(output_name):
        print "{output_name} created.".format(output_name=output_name)


if __name__ == '__main__':
    name_cluster = ['IC1805', 'IC1848', 'IC2944', 'NGC6231', 'NGC6611', 'Wd2', 'M8', 'NGC6357', 'R136',
                    'M17']  # 'G333',
    # observed_dispersion = [65.45329203402903, 50.26275741127991, 31.357122212648363, 67.61903406110999,
    #                        25.32524131575383, 14.999765386529885, 30.94, 25.73, 18.04, 25.0, 5.5]
    # number_stars = [8, 5, 14, 13, 9, 44, 22, 30, 8, 332, 12]
    # min_masses = [15, 15, 15, 15, 15, 6, 6, 6, 6, 15, 6]
    # max_masses = [60, 60, 60, 60, 60, 60, 20, 20, 20, 60, 20]
    observed_dispersion = [65.45329203402903, 50.26275741127991, 31.357122212648363, 67.61903406110999,
                           25.32524131575383, 14.999765386529885, 32.69110237188463, 26.850667796133674,
                           25.0, 5.5]  # 5.56089079139046,
    number_stars = [8, 5, 14, 13, 9, 44, 16, 22, 332, 12]  # 4,
    min_masses = [20, 15, 15, 15, 15, 6, 6, 6, 15, 6]  # 13,
    max_masses = [60, 60, 60, 60, 60, 60, 20, 30, 60, 20]  # 30,

    refs = ['\citet{2017ApJS..230....3S}', '\citet{2014MNRAS.438.1451L}', '\citet{2014MNRAS.443..411B}',
            '\citet{2013AJ....145...37S}', '\citet{2008AA...490.1071G}', '\citet{2018AJ....156..211Z}',
            '\citetalias{2020AA...633A.155R}', '\citetalias{2020AA...633A.155R}', '\citet{2012AA...546A..73H}',
            '\citet{paper1}']

    # ascii.write([name_cluster, observed_dispersion, number_stars, min_masses, max_masses, refs],
    #             'data_RT20/Nstars_massRange.txt',
    #             names=['name', 'sigma', 'Nstars', 'minMass', 'maxMass', 'refs'])

    # print name_cluster[0:4]
    # find_Pmin_fbin_individual_clulsters(name_cluster[0:1], observed_dispersion[0:1], number_stars[0:1],
    #                                     min_masses[0:1], max_masses[0:1])
    plot_Pmin_vs_age(fit=True)

    # n, minmass, maxmass = number_stars[0], min_masses[0], max_masses[0]
    # for n, minmass, maxmass in zip(number_stars, min_masses, max_masses):
    #     print n, minmass, maxmass
    #     start = time.time()
    #     run_bigGrid(number_stars=n, min_mass=minmass, max_mass=maxmass)
    #     print "It took %.3g seconds" % (time.time() - start)
