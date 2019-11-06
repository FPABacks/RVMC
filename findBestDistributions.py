import matplotlib.pyplot as plt
import numpy as np
import main_RVMC as RVMC
from statistics import mode
from scipy import stats


# This comand is to open plots in the second display
plt.switch_backend('QT4Agg')


def create_sig1D_distribution(measured_errors, number_of_samples=10 ** 5, sample_size=12,
                              number_of_stars=10**5, min_mass=6, max_mass=20, binary_fraction=0.7, min_period=1.4,
                              max_period=3500, sigma_dyn=2.0):
    measured_errors = RV_errors

    RV_dist = RVMC.synthetic_RV_distribution(number_of_stars, min_mass, max_mass, binary_fraction, min_period,
                              max_period, sigma_dyn)
    np.random.seed(0)
    sig1D = np.std(np.random.choice(RV_dist, (number_of_samples, sample_size))
                   + np.random.normal(0, measured_errors, (number_of_samples, sample_size)), axis=1)

    return sig1D



RV_errors = np.array([0.8, 2.3, 1.0, 1.8, 0.8, 1.0, 1.4, 2.5, 2.0, 0.4, 1.5, 0.6])
def get_Pdistr_stats(pmin=1.4,pmax=10000,Npoints=5,nbins=500, fbin=0.7):
    mode, mean, median, Allvalues, Allbins, p16, p84, periods = [], [], [], [], [], [], [], []
    for period in np.logspace(np.log10(pmin), np.log10(pmax), Npoints):#np.linspace(pmin, pmax, Npoints):#
        print 'Calulating distribution for fbin = %5.2f and Pmin = %5.2f' %(fbin, period)
        sig1D = create_sig1D_distribution(measured_errors=RV_errors, min_period=period, binary_fraction=fbin)
        values, bins, patches = plt.hist(sig1D, histtype="step", bins=np.linspace(0, 500, nbins),
                                         density=True, label=r'$P_{min}=$ %5.2f' % (period) )
        index = np.where(values == np.max(values))
        mode.append(bins[index]+(bins[1] - bins[0])/2.)

        mean.append(np.mean(sig1D))
        median.append(np.median(sig1D))
        p16.append(np.percentile(sig1D,16))
        p84.append(np.percentile(sig1D,84))
        periods.append(period)

        Allvalues.append(values)
        Allbins.append(bins)
        print 'mode = %5.2f' %(bins[index]+(bins[1] - bins[0])/2.)
    # plt.close()
    return mode, mean, median, Allvalues, Allbins, p16, p84, periods

mode, mean, median, values, bins, p16, p84, period = get_Pdistr_stats(pmin=1.4,pmax=10000,Npoints=20,nbins=800)

diff = 10
index = 0
for i, m in enumerate(mode):
    diff_temp = np.abs(m - 5.5)
    if diff_temp < diff:
        diff = diff_temp
        index = i
        print index, diff, m
# print diff, index, mode[index], median[index], p16[index], p84[index]

# min_period = period[index]
# # print min_period
# plt.step(bins[index][:-1], values[index], where='post')
# plt.vlines([p16[index], p84[index]], 0, 0.1, 'b')
# plt.vlines(mode[index], 0, 0.1, 'b', lw=3)
#
# plt.step(bins[0][:-1], values[0], where='post')
# plt.vlines([p16[0], p84[0]], 0, 0.1, 'orange')
# plt.vlines(mode[0], 0, 0.1, 'orange')
#
# plt.step(bins[-1][:-1], values[-1], where='post')
# plt.vlines([p16[-1], p84[-1]], 0, 0.1, 'g')
# plt.vlines(mode[-1], 0, 0.1, 'g')

plt.xlim(0,50)

# plt.vlines(mode,0,0.15)
#
plt.vlines(5.5, 0, 0.05, 'b')
plt.legend()
plt.show()
