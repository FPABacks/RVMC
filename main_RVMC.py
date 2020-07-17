import numpy as np
import matplotlib.pyplot as plt
from astropy.constants import G
from scipy.optimize import brentq
from scipy import stats
# from matplotlib.animation import FuncAnimation
from matplotlib.animation import FuncAnimation
import matplotlib.ticker

plt.switch_backend('QT4Agg')

# np.random.seed(0)
RV_errors = np.array([0.8, 2.3, 1.0, 1.8, 0.8, 1.0, 1.4, 2.5, 2.0, 0.4, 1.5, 0.6])
# RV_errors = np.zeros(12)

def initialize_parameters(number_of_stars=100000, min_mass=6, max_mass=20, min_period=1.41, max_period=3500):
    """
    Create the random samples following their respective distributions.
    :param number_of_stars:     (int) number of stars (duh)
    :param min_mass:            (scalar) minimum mass in solar masses
    :param max_mass:            (scalar) maximum mass in solar masses
    :param min_period:          (scalar) minimum period in days
    :param max_period:          (scalar) maximum period in days.
    :return:    Note returns the masses in kg and period in seconds.
    """
    # <<<<<<< HEAD
    inclination = np.arccos(np.random.random(size=number_of_stars))
    # =======
    #     inclination = np.random.uniform(0, 1, size=number_of_stars)# * np.pi * 0.5 #cosi = random(0,1)
    #     inclination = np.arccos(inclination)
    # >>>>>>> b87fcb5dc52eee01ddf66b2f6134e958ddde04cb
    a = 2.3
    # primary_mass = np.random.uniform(min_mass**(-a), max_mass**(-a), size=number_of_stars) ** -(1. / a) * 2e30

    # CDF sampling of the IMF powerlaw
    Pmin = (min_mass / min_mass) ** (-a + 1)
    Pmax = (max_mass / min_mass) ** (-a + 1)
    P = np.random.uniform(Pmax, Pmin, size=number_of_stars)
    primary_mass = P ** (1. / (-a + 1)) * min_mass * 2e30

    period = 10 ** (np.random.uniform(np.log10(min_period) ** 0.5, np.log10(max_period) ** 0.5,
                                      number_of_stars) ** 2) * 24 * 3600
    # Oepik distribution
    # period = 10**(np.random.uniform(np.log10(min_period), np.log10(max_period), number_of_stars)) * 24 * 3600

    mass_ratio = np.random.uniform(0.1, 1., number_of_stars)
    # mass_ratio = np.zeros(number_of_stars)
    # In case you want a mass ratio distribution as f(q) ~ q**-0.1
    # mass_ratio = np.random.uniform(0.1 ** (1. / 1.1), 1, number_of_stars) ** 1.1

    orbit_rotation = np.random.uniform(0, 1, size=number_of_stars) * 2 * np.pi

    eccentricity = np.zeros(number_of_stars)
    index6d = (period > (4 * 24 * 3600)) & (period < (6 * 24 * 3600))
    eccentricity[index6d] = np.random.uniform(0 ** 0.5, 0.5 ** 0.5, np.sum(index6d)) ** 2
    index4d = period < (4 * 24 * 3600)
    eccentricity[np.logical_not(index6d) & np.logical_not(index4d)] = (
                np.random.uniform(0 ** 0.5, 0.9 ** 0.5, np.sum(np.logical_not(index6d) & np.logical_not(index4d))) ** 2)

    # eccentricity = (np.random.uniform(0**0.5, 0.9**0.5, number_of_stars)**2)

    # Edit eccentricity array to avoid unphysical combinations of eccentricity and period
    # All orbits with periods of 4 or less days are circularized e=0
    # If the period is between 4 and 6 days and the eccentricity is high, divide it in half to better repesent the
    # observed distribution
    # index4d = np.where(period <= 4 * 24 * 3600)
    # index6d = np.where(period[np.where(eccentricity[np.where((period > 4 * 24 * 3600) &
    #                                                          (period < 6 * 24 * 3600))])] > 0.9)

    # index4d = period <= (4 * 24 * 3600)

    # index09e = (eccentricity > 0.9)

    # eccentricity[index4d] = 0.
    # for i6d in index6d:
    #     eccentricity[i6d] = eccentricity[i6d] / 2.
    #
    # eccentricity[index09e] = eccentricity[index09e] / 2.

    time = np.random.uniform(0, 1, size=number_of_stars) * period
    eccentric_anomaly = find_eccentric_anomaly(eccentricity, time, period)

    return inclination, eccentric_anomaly, primary_mass, period, mass_ratio, orbit_rotation, eccentricity


def anomaly(E, e, t, P):
    return E - e * np.sin(E) - t * 2 * np.pi / P


def find_eccentric_anomaly(eccentricity, time, period):
    """
    Finds the eccentric anomaly based on the eccentricity, period and part of the orbit that has been completed (here
    based on time with 0 <= time < period). This has no analytical solution so it is numerically solved using brentq.
    This can probably be sped up significantly by changing xtol to something higher than 2e-12 (the default tolerance).
    """
    eccentric_anomaly = np.zeros(len(time))
    for i in range(len(time)):
        eccentric_anomaly[i] = brentq(anomaly, 0, np.pi * 2, args=(eccentricity[i], time[i], period[i]), xtol=10 ** -2)
    return eccentric_anomaly


def max_orbital_velocity(primary_mass, mass_ratio, eccentricity, semi_major_axes):
    return ((mass_ratio / (1 + mass_ratio)) * ((G.value * primary_mass * (1 + mass_ratio) * (1 + eccentricity)) /
                                               ((1 - eccentricity) * semi_major_axes)) ** 0.5) / 1000.


# def max_orbital_velocity(primary_mass, mass_ratio, eccentricity, semi_major_axes):
#     return ((1. / (1 + mass_ratio)) * ((G.value * primary_mass * (1 + mass_ratio) * (1 + eccentricity)) /
#                                                ((1 - eccentricity) * semi_major_axes))**0.5) / 1000.


def binary_radial_velocity(v_max, i, e, w, E):
    return v_max * np.sin(i) * (
                e * np.cos(w) + np.cos(2 * np.arctan((((1 + e) / (1 - e)) ** 0.5) * np.tan(0.5 * E)) + w))


def semi_major_axis(period, primary_mass, mass_ratio):
    return ((4 * np.pi ** 2) / (G.value * (1 + mass_ratio) * primary_mass * period ** 2)) ** -0.333


def synthetic_RV_distribution(number_of_stars=10 ** 5, min_mass=6, max_mass=20, binary_fraction=0.7, min_period=1.4,
                              max_period=3500, sigma_dyn=2.0):
    """
    Simulates the radial velocities of <number of stars> in a cluster for specified parameters below.
    :param number_of_stars:
    :param min_mass:
    :param max_mass:
    :param binary_fraction:
    :param min_period:
    :param max_period:
    :return:
    """

    # Print model parameters
    print '\n #### Creating synthetic RV distribution #### \n ' \
          'Number of stars: %i \n Mass range: [%5.2f, %5.2f] Msun \n Binary fraction: %5.2f \n ' \
          'Period range: [%5.2f, %5.2f] days \n Cluster dynamical dispersion: %5.2f km/s\n ########' \
          % (number_of_stars, min_mass, max_mass, binary_fraction, min_period, max_period, sigma_dyn)

    # The number of binaries is randomly determined based on the binary fraction. It can also be a fixed number:
    # number_of_binaries = int(binary_fraction * number_of_stars)
    binaries = np.random.uniform(0, 1, number_of_stars) < binary_fraction
    number_of_binaries = np.sum(binaries)

    # Normally distributed cluster velocities around 0, currently with sigma_1D_cluster = 2 km/s
    cluster_velocities = np.random.normal(0.0, sigma_dyn, size=number_of_stars)

    # An extra dispersion caused by measurement errors??
    # measurement_errors = np.random.normal(0.0, measured_errors, size=number_of_stars)

    # generate orbital parameters and stellar properties. Note: only for the binary stars!
    inclination, eccentric_anomaly, primary_mass, period, mass_ratio, orbit_rotation, eccentricity = \
        initialize_parameters(number_of_stars=number_of_binaries, min_mass=min_mass, max_mass=max_mass,
                              min_period=min_period, max_period=max_period)

    # Get the radial velocities of the binaries
    semi_major_axes = semi_major_axis(period, primary_mass, mass_ratio)
    v_orb_max = max_orbital_velocity(primary_mass, mass_ratio, eccentricity, semi_major_axes)
    RV_binary = binary_radial_velocity(v_orb_max, inclination, eccentricity, orbit_rotation, eccentric_anomaly)

    RV = cluster_velocities  # + measurement_errors
    RV[binaries] += RV_binary

    return RV


def N_samples_with_error(number_of_stars=12, number_of_samples=10 ** 5, errors=[2.0], **kwargs):
    RV_dist = synthetic_RV_distribution(number_of_stars=number_of_samples, **kwargs)
    many_dists = np.random.choice(RV_dist, (number_of_samples, number_of_samples)) + \
                 np.random.normal(0, errors, (number_of_samples, number_of_stars))
    return many_dists


def ks(synthesized, observed):
    """
    Performs a Kolmogorov-Smirnov test to compare the measured distribution to the synthesized distribution.
    """
    ksval, pval = stats.ks_2samp(observed, synthesized)
    return ksval, pval


def fbin_search(velocities=[], fmin=0, fmax=1, Npoints=100, Nstars=10 ** 4, Nsamples=1000, errors=[], **kwargs):
    """
    Loops through binary fractions to find the best fitting fraction for the observed radial velocities. Fitting done
    by means of KS test. Will do some bootstrapping if observation errors are supplied (the bootstrapping might be
    questionable, not sure)
    :param velocities:  (array) observed radial velocities in km/s
    :param fmin:        (scalar) lowest binary fraction to be tested
    :param fmax:        (scalar) higest binary fraction to be tested
    :param Npoints:     (int) number of binary fractions to be tested
    :param Nstars:      (int) number of stars per synthetic sample does not have to be equal to the sample size
    :param Nsamples:    (int) number of bootstrap samples incase errors are specified
    :param errors:      (array) observational errors in km/s
    :param kwargs:      Keyword arguments passed to synthetic_RV_distribution
    :return:
    """

    binary_fractions = np.linspace(fmin, fmax, Npoints, endpoint=True)
    print binary_fractions

    print "Generating distributions"
    synth_dists = np.zeros((Npoints, Nstars))
    for i, fbin in enumerate(binary_fractions):
        synth_dists[i] = synthetic_RV_distribution(binary_fraction=fbin, number_of_stars=Nstars, **kwargs)
        print "Progress: %i %%" % i

    if type(errors) != type(None):

        ksvals = np.zeros((Npoints, Nsamples))
        pvals = np.zeros((Npoints, Nsamples))

        if len(errors) != len(velocities):
            print "Use the same amount of errors and velocities, dipshit!"
            print "Velocities: %i, Errors: %i" % (len(velocities, len(errors)))

        print "Generating random velocities"
        bootstrap_vels = np.zeros((len(velocities), Nsamples))
        for i, vel in enumerate(velocities):
            bootstrap_vels[i] = stats.norm.rvs(size=Nsamples, loc=vel, scale=errors[i])

        print "Doing KS tests"

        for sample_index in range(Nsamples):
            print "Sample number: %i" % sample_index
            for i, fbin in enumerate(binary_fractions):
                # print "fbin: %.3g" % fbin
                ksval, pval = ks(synth_dists[i], bootstrap_vels[:, sample_index], **kwargs)
                ksvals[i, sample_index] = ksval
                pvals[i, sample_index] = pval

        np.save("pvals", pvals)
        np.save("ksvals", ksvals)
        np.save("binary_fraction", binary_fractions)

        print "plotting the stuff"

        fig, axarr = plt.subplots(1, 2, figsize=(8, 6))
        fractions = []
        for i in range(Nsamples):
            arg = np.argmax(pvals[:, i])
            axarr[0].scatter(binary_fractions[arg], pvals[arg, i])
            axarr[1].scatter(binary_fractions[arg], ksvals[arg, i])
            fractions.append(binary_fractions[arg])

        fractions = np.sort(fractions)
        sigma = [fractions[int(len(fractions) * 0.16)], fractions[int(len(fractions) * 0.84)]]
        axarr[0].fill_between(sigma, 0, 1, alpha=0.5, facecolor='red')

        plt.show()

        plt.hist(fractions, bins=np.linspace(fmin, fmax, 20))
        plt.axvline(sigma[0], lw=2, color='red', alpha=0.5)
        plt.axvline(sigma[1], lw=2, color='red', alpha=0.5)
        plt.show()

        print sigma

    else:
        ksvals = np.zeros(Npoints)
        pvals = np.zeros(Npoints)

        for i, fbin in enumerate(binary_fractions):
            # print "fbin: %.3g" % fbin
            ksval, pval = ks(synth_dists[i], velocities, **kwargs)
            ksvals[i] = ksval
            pvals[i] = pval

        fig = plt.figure(figsize=(7.5, 7.5))
        plt.plot(binary_fractions, pvals)
        plt.title("Binary fraction probablility distribution no error")
        plt.ylim([0, 1])
        plt.xlabel("Binary fraction")
        plt.ylabel("p-value")
        fig.savefig("Binary fraction probablility distribution no error sig=2.pdf")
        plt.show()


def fbin_period_search(fmin, fmax, pmin, pmax, Npoints=100, velocities=[], **kwargs):
    """
    Same as fbin_search, but without the option of bootstrapping and in both period as binary fraction
    :param fmin:
    :param fmax:
    :param pmin:
    :param pmax:
    :param Npoints:
    :param velocities:
    :param kwargs:
    :return:
    """

    binary_fractions = np.linspace(fmin, fmax, Npoints)
    # period_range = np.linspace(pmin, pmax, Npoints)
    period_range = np.logspace(np.log10(pmin), np.log10(pmax), Npoints)

    ksvals = np.zeros((Npoints, Npoints))
    pvals = np.zeros((Npoints, Npoints))

    print "Generating distributions"
    # synth_dists = np.zeros((Npoints, Npoints, kwargs["number_of_stars"]))
    for i, p_cutoff in enumerate(period_range):
        print "Progress: %i%%" % i
        for j, fbin in enumerate(binary_fractions):
            synth_dist = synthetic_RV_distribution(binary_fraction=fbin, min_period=p_cutoff, **kwargs)

            # for i, fbin in enumerate(binary_fractions):
            #     print "step %.3g \t fbin %.3g" % (i / float(Npoints), fbin)
            #     for j, pmin in enumerate(period_range):
            # print "fbin: %.3g \n pmin: %.3g" % (fbin, pmin)
            ksval, pval = ks(synth_dist, velocities)

            ksvals[i, j] = ksval
            pvals[i, j] = pval

    fig, axarr = plt.subplots(1, 2, figsize=(12, 5))

    axarr[0].set_title("p-value")
    cm1 = axarr[0].pcolormesh(binary_fractions, period_range, pvals)
    axarr[0].set_xlabel("Binary fraction")
    axarr[0].set_ylabel("Period cutoff (only P > [days])")
    axarr[0].set_yscale('log')
    axarr[0].set_yticks([1, 3, 10, 30])
    axarr[0].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    cb1 = fig.colorbar(cm1, ax=axarr[0])
    cb1.set_label("p-value")

    axarr[1].set_title("ks-value")
    cm2 = axarr[1].pcolormesh(binary_fractions, period_range, ksvals)
    axarr[1].set_xlabel("Binary fraction")
    axarr[1].set_ylabel("Period cutoff (only P > [days])")
    axarr[1].set_yscale('log')
    axarr[1].set_yticks([1, 3, 10, 30])
    axarr[1].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    cb2 = fig.colorbar(cm1, ax=axarr[1])
    cb2.set_label("ks-value")

    plt.tight_layout()

    np.save("period", period_range)
    np.save("binary_fraction", binary_fractions)
    np.save("ksvals", ksvals)
    np.save("pvals", pvals)

    fig.savefig("fbin_pmin_exploration.pdf")
    fig.savefig("fbin_pmin_exploration.png")
    plt.show()


def std_search(velocities=[], Nsamples=10 ** 5, fmin=0.0, fmax=1, Npoints=100, Pmin=1.5, **kwargs):
    """
    Uses the randomness of the observed sample to determine the range of possible binary fractions.
    """
    sample_size = len(velocities)
    binary_fractions = np.linspace(fmin, fmax, Npoints)

    obs_std = np.std(velocities)

    stds = np.zeros((Npoints, Nsamples))

    for i, fbin in enumerate(binary_fractions):
        print "%.3g%%" % (float(i) / Npoints * 100)
        for j in range(Nsamples):
            stds[i, j] = np.std(
                synthetic_RV_distribution(number_of_stars=sample_size, binary_fraction=fbin, min_period=Pmin, **kwargs))

    np.save("stds.npy", stds)
    np.save("fbins", binary_fractions)

    low_bound_fbin = 1
    high_bound_fbin = 0
    low_i = 1
    high_i = 0

    for i, fbin in enumerate(binary_fractions):
        std_array = np.sort(stds[i])
        sigma = [std_array[int(len(std_array) * 0.16)], std_array[int(len(std_array) * 0.84)]]
        if sigma[0] < obs_std < sigma[1]:
            if fbin < low_bound_fbin:
                low_bound_fbin = fbin
                low_i = i
                print "lowest: ", fbin
            if fbin > high_bound_fbin:
                high_bound_fbin = fbin
                high_i = i
                print "highest: ", fbin

    # Plotting
    bins = np.linspace(0, 50, 100)
    plt.hist(stds[low_i], bins=bins, histtype="step", label="$f_{bin} = %.2g$" % low_bound_fbin, density=True)
    plt.hist(stds[high_i], bins=bins, histtype="step", label="$f_{bin} = %.2g$" % high_bound_fbin, density=True)
    plt.axvline(obs_std, lw=2, color='red', alpha=0.5, label="Observed")
    plt.xlabel("$\sigma_{1D}$ [km s$^{-1}$]")
    plt.ylabel("Frequency [(km s$^{-1}$)$^{-1}$]")
    plt.xlim([0, 50])
    plt.legend()
    plt.show()


def simple_std_plot(number_of_samples=5 * 10 ** 5, measured_errors=[], **kwargs):
    """
    Remake the plot of the M17 paper. This uses the fast method (drawing elements from a big sample)
    :param number_of_samples:
    :param kwargs:
    :return:
    """

    nbins = 1000
    print measured_errors.shape
    fig, (fax, pax) = plt.subplots(1, 2, figsize=(6, 3))

    nstars = 12

    RV_dist = synthetic_RV_distribution(number_of_stars=number_of_samples, binary_fraction=0.7, **kwargs)

    sig1D1 = np.std(np.random.choice(RV_dist, (number_of_samples, nstars)) +
                    np.random.normal(0, measured_errors, (number_of_samples, nstars)), axis=1)

    RV_dist = synthetic_RV_distribution(number_of_stars=number_of_samples, binary_fraction=0.28, **kwargs)
    sig1D2 = np.std(np.random.choice(RV_dist, (number_of_samples, nstars)) +
                    np.random.normal(0, measured_errors, (number_of_samples, nstars)), axis=1)

    RV_dist = synthetic_RV_distribution(number_of_stars=number_of_samples, binary_fraction=0.12, **kwargs)

    sig1D3 = np.std(np.random.choice(RV_dist, (number_of_samples, nstars)) +
                    np.random.normal(0, measured_errors, (number_of_samples, nstars)), axis=1)

    # sig1D1 = np.array([np.std(synthetic_RV_distribution(nstars, binary_fraction=0.7, **kwargs)) for i in range(number_of_samples)])
    # sig1D2 = np.array([np.std(synthetic_RV_distribution(nstars, binary_fraction=0.28, **kwargs)) for i in range(number_of_samples)])
    # sig1D3 = np.array([np.std(synthetic_RV_distribution(nstars, binary_fraction=0.12, **kwargs)) for i in range(number_of_samples)])

    fax.hist(sig1D1, histtype="step", bins=np.linspace(0, 500, nbins), density=True, label=r"f$_{\rm bin}$=0.70")
    fax.hist(sig1D2, histtype="step", bins=np.linspace(0, 500, nbins), density=True, label=r"f$_{\rm bin}$=0.28")
    fax.hist(sig1D3, histtype="step", bins=np.linspace(0, 500, nbins), density=True, label=r"f$_{\rm bin}$=0.12")
    fax.legend()
    fax.set_xlim([0, 50])
    fax.set_ylim([0, 0.1])
    fax.set_xlabel(r"$\sigma_{\rm 1D}$ [km s$^{-1}$]")
    fax.set_ylabel(r"Frequency [(km s$^{-1}$)$^{-1}$]")

    # sig1D1 = np.array([np.std(synthetic_RV_distribution(nstars, min_period=1.4, **kwargs)) for i in range(number_of_samples)])
    # sig1D2 = np.array([np.std(synthetic_RV_distribution(nstars, min_period=30, **kwargs)) for i in range(number_of_samples)])
    # sig1D3 = np.array([np.std(synthetic_RV_distribution(nstars, min_period=8*365, **kwargs)) for i in range(number_of_samples)])

    RV_dist = synthetic_RV_distribution(number_of_stars=number_of_samples, min_period=1.4, **kwargs)
    sig1D1 = np.std(np.random.choice(RV_dist, (number_of_samples, nstars)) + np.random.normal(0, measured_errors, (
    number_of_samples, nstars)), axis=1)

    RV_dist = synthetic_RV_distribution(number_of_stars=number_of_samples, min_period=30, **kwargs)
    sig1D2 = np.std(np.random.choice(RV_dist, (number_of_samples, nstars)) + np.random.normal(0, measured_errors, (
    number_of_samples, nstars)), axis=1)

    RV_dist = synthetic_RV_distribution(number_of_stars=number_of_samples, min_period=8 * 365, **kwargs)
    sig1D3 = np.std(np.random.choice(RV_dist, (number_of_samples, nstars)) + np.random.normal(0, measured_errors, (
    number_of_samples, nstars)), axis=1)

    pax.hist(sig1D1, histtype="step", bins=np.linspace(0, 500, nbins), density=True, label=r"P$_{\rm cutoff}$=1.4 d")
    pax.hist(sig1D2, histtype="step", bins=np.linspace(0, 500, nbins), density=True, label=r"P$_{\rm cutoff}$=30 d")
    pax.hist(sig1D3, histtype="step", bins=np.linspace(0, 500, nbins), density=True, label=r"P$_{\rm cutoff}$=8 yr")

    pax.legend()
    pax.set_xlim([0, 50])
    pax.set_ylim([0, 0.1])
    pax.set_xlabel(r"$\sigma_{\rm 1D}$ [km s$^{-1}$]")
    pax.set_ylabel(r"Frequency [(km s$^{-1}$)$^{-1}$]")

    plt.tight_layout()
    plt.savefig("sigma1D_test_with_rttot.pdf")
    plt.show()


def plot_dists(inclination, eccentric_anomaly, primary_mass, period, mass_ratio, orbit_rotation, eccentricity,
               cumulative=False):
    """
    Plot the distributions.
    :param inclination:
    :param eccentric_anomaly:
    :param primary_mass:
    :param period:
    :param mass_ratio:
    :param orbit_rotation:
    :param eccentricity:
    :return:
    """

    N = len(inclination)
    fig, axarr = plt.subplots(3, 3, figsize=(8, 8))

    axarr[0][0].hist(inclination, bins=int(N ** 0.333), density=True, cumulative=cumulative)
    axarr[0][0].set_xlabel("inclination")

    axarr[0][1].hist(eccentric_anomaly, bins=int(N ** 0.333), density=True, cumulative=cumulative)
    axarr[0][1].set_xlabel("eccentric anomaly")

    axarr[0][2].hist(primary_mass/2e30, bins=int(N ** 0.333), density=True, cumulative=cumulative)
    axarr[0][2].set_xlabel("primary_mass (M$_{\odot})$")

    axarr[1][0].hist(period / (24 * 3600), bins=np.geomspace(np.min(period/(24 * 3600)), np.max(period/(24 * 3600)), int(N ** 0.333)),
                     density=True, cumulative=cumulative)
    axarr[1][0].set_xlabel("period (days)")
    axarr[1][0].set_xscale("log")

    axarr[1][1].hist(mass_ratio, bins=int(N ** 0.333), density=True, cumulative=cumulative)
    axarr[1][1].set_xlabel("mass_ratio")

    axarr[1][2].hist(orbit_rotation, bins=int(N ** 0.333), density=True, cumulative=cumulative)
    axarr[1][2].set_xlabel("orbit_rotation")

    axarr[2][0].hist(eccentricity, bins=int(N ** 0.333), density=True, cumulative=cumulative)
    axarr[2][0].set_xlabel("eccentricity")

    plt.tight_layout()
    plt.show()


def compare_one_big_sample_vs_many_small_samples(number_of_samples=10 ** 5, sample_size=12, big_sample_size=10 ** 5,
                                                 **kwargs):
    import time

    print "Starting small samples"
    start = time.time()
    sig1D1 = np.std([synthetic_RV_distribution(sample_size, **kwargs) for i in range(number_of_samples)], axis=1)
    print "It took %.3g seconds" % (time.time() - start)

    # This is the fast way!
    print "\nStarting big sample"
    start = time.time()
    RV_dist = synthetic_RV_distribution(number_of_stars=big_sample_size, **kwargs)
    sig1D2 = np.std(np.random.choice(RV_dist, (number_of_samples, sample_size)), axis=1)
    print "It took %.3g seconds" % (time.time() - start)

    nbins = 1000
    plt.hist(sig1D1, histtype="step", bins=np.linspace(0, 500, nbins), density=True, label=r"many small samples")
    plt.hist(sig1D2, histtype="step", bins=np.linspace(0, 500, nbins), density=True, label=r"one big sample")
    plt.legend()
    plt.xlim([0, 50])
    plt.ylim([0, 0.1])
    plt.xlabel(r"$\sigma_{\rm 1D}$ [km s$^{-1}$]")
    plt.ylabel(r"Frequency [(km s$^{-1}$)$^{-1}$]")
    plt.show()


def simple_std_plot_bigSample(number_of_samples=10 ** 5, sample_size=12, big_sample_size=10 ** 5,
                              measured_errors=RV_errors, **kwargs):
    """
    Remake the plot of the M17 paper generating a big sample and drawing several samples of the same size of the number of stars.
    :param number_of_samples:
    :param sample_size:
    :param big_sample_size:
    :param kwargs:
    :return:
    """
    import time

    nbins = 1000

    fig, (fax, pax) = plt.subplots(1, 2, figsize=(6, 3))

    print "Starting big samples"
    start = time.time()

    RV_dist1 = synthetic_RV_distribution(number_of_stars=big_sample_size, binary_fraction=0.7, **kwargs)
    RV_dist2 = synthetic_RV_distribution(number_of_stars=big_sample_size, binary_fraction=0.28, **kwargs)
    RV_dist3 = synthetic_RV_distribution(number_of_stars=big_sample_size, binary_fraction=0.12, **kwargs)
    sig1D1 = np.std(np.random.choice(RV_dist1, (number_of_samples, sample_size)) +
                    np.random.normal(0, measured_errors, (number_of_samples, sample_size)), axis=1, ddof=1)
    sig1D2 = np.std(np.random.choice(RV_dist2, (number_of_samples, sample_size)) +
                    np.random.normal(0, measured_errors, (number_of_samples, sample_size)), axis=1, ddof=1)
    sig1D3 = np.std(np.random.choice(RV_dist3, (number_of_samples, sample_size)) +
                    np.random.normal(0, measured_errors, (number_of_samples, sample_size)), axis=1, ddof=1)

    fax.hist(sig1D1, histtype="step", bins=np.linspace(0, 500, nbins), density=True, label=r"f$_{\rm bin}$=0.70")
    fax.axvline(np.median(sig1D1))
    fax.hist(sig1D2, histtype="step", bins=np.linspace(0, 500, nbins), density=True, label=r"f$_{\rm bin}$=0.28")
    fax.axvline(np.median(sig1D2))
    fax.hist(sig1D3, histtype="step", bins=np.linspace(0, 500, nbins), density=True, label=r"f$_{\rm bin}$=0.12")
    fax.axvline(np.median(sig1D3))
    fax.legend()
    fax.set_xlim([0, 50])
    fax.set_ylim([0, 0.1])
    fax.set_xlabel(r"$\sigma_{\rm 1D}$ [km s$^{-1}$]")
    fax.set_ylabel(r"Frequency [(km s$^{-1}$)$^{-1}$]")

    fax.vlines(5.6, 0, 0.05, 'k')
    fax.axvspan(5.1, 6.1, color='k', alpha=0.5, lw=0)


    RV_dist1 = synthetic_RV_distribution(number_of_stars=big_sample_size, min_period=1.4, **kwargs)
    RV_dist2 = synthetic_RV_distribution(number_of_stars=big_sample_size, min_period=30, **kwargs)
    RV_dist3 = synthetic_RV_distribution(number_of_stars=big_sample_size, min_period=8 * 365, **kwargs)
    sig1D1 = np.std(np.random.choice(RV_dist1, (number_of_samples, sample_size)) + np.random.normal(0, measured_errors,
                                                                                                    (number_of_samples,
                                                                                                     sample_size)),
                    axis=1, ddof=1)
    sig1D2 = np.std(np.random.choice(RV_dist2, (number_of_samples, sample_size)) + np.random.normal(0, measured_errors,
                                                                                                    (number_of_samples,
                                                                                                     sample_size)),
                    axis=1, ddof=1)
    sig1D3 = np.std(np.random.choice(RV_dist3, (number_of_samples, sample_size)) + np.random.normal(0, measured_errors,
                                                                                                    (number_of_samples,
                                                                                                     sample_size)),
                    axis=1, ddof=1)

    print "It took %.3g seconds" % (time.time() - start)

    pax.hist(sig1D1, histtype="step", bins=np.linspace(0, 500, nbins), density=True, label=r"P$_{\rm cutoff}$=1.4 d")
    pax.axvline(np.median(sig1D1))
    pax.hist(sig1D2, histtype="step", bins=np.linspace(0, 500, nbins), density=True, label=r"P$_{\rm cutoff}$=30 d")
    pax.axvline(np.median(sig1D2))
    pax.hist(sig1D3, histtype="step", bins=np.linspace(0, 500, nbins), density=True, label=r"P$_{\rm cutoff}$=8 yr")
    pax.axvline(np.median(sig1D3))
    pax.legend()
    pax.set_xlim([0, 50])
    pax.set_ylim([0, 0.1])
    pax.set_xlabel(r"$\sigma_{\rm 1D}$ [km s$^{-1}$]")
    pax.set_ylabel(r"Frequency [(km s$^{-1}$)$^{-1}$]")

    pax.vlines(5.6, 0, 0.05, 'k')
    pax.axvspan(5.1, 6.1, color='k', alpha=0.5, lw=0)

    plt.tight_layout()
    plt.savefig("sigma1D_test_fast1.pdf")
    plt.show()


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def poly(x, a, b, c, d, e):
    return a + b * x + c * x ** 2 + d * x ** 3 + e * x ** 4


def LogLikelihood_Pois_Integ(parm, model, yval, edges):
    """
    Calculates the loglikelihood using poisson
    """
    mod = lambda x: model(x, *parm)
    ymod = np.zeros(len(yval))
    for i in range(len(edges[:-1])):
        ymod[i], ymoderr = quad(mod, edges[i], edges[i + 1])
        # we don't normalise by bin width since the rate parameter
        # is set by the model and needs to be counts per bin

    pd = stats.poisson(ymod)  # we define our Poisson distribution
    return -sum(np.log(pd.pmf(yval)))


def binned_pdf(edges, model, parm):
    """
    Turn a smooth pdf into a stepped pdf
    Only works with frozen scipy stats distributions (or distributions defined for compatibilty)
    """
    binned_model = np.zeros(len(edges) - 1)
    mod = lambda x: model(x, *parm)
    for i in range(len(edges[:-1])):
        binned_model[i], ymoderr = quad(mod, edges[i], edges[i + 1])

    # binned_model = (dist.cdf(edges[1:]) - dist.cdf(edges[:-1])) / (edges[1:] - edges[:-1])
    xvals = np.repeat(edges[:], 2)
    modvals = np.repeat(binned_model, 2)
    modvals = np.insert(modvals, 0, 0.)
    modvals = np.append(modvals, 0.)
    return xvals, modvals


def find_best_period(pmin=1.4, pmax=5500, Npoints=100, measured_errors=RV_errors, measured_sigma=5.5, **kwargs):
    """
    Loops through minimum periods to find the best fitting pmin for the observed radial velocities.
    :param obs_dispersion (scalar) observed velocity dispersion of the cluster
    :param dispersion_error (scalar) error on the measured velocity dispersion
    :param pmin:        (scalar) minimum period to be tested
    :param Npoints:     (int) number of minumum periods to be tested
    :param Nstars:      (int) number of stars per synthetic sample does not have to be equal to the sample size
    :param sample_size (int) number of stars used to calculate the observed velocity dispersion
    :param kwargs:      Keyword arguments passed to synthetic_RV_distribution
    :return:
    """

    nbins = 4001
    temp_perc16_diff = 1e5
    temp_perc05_diff = 1e5
    temp_perc84_diff = 1e5
    temp_sigma_diff = 1e5
    start = time.time()

    for period in np.logspace(np.log10(pmin), np.log10(pmax), Npoints):  # np.array([365*9, 365*12, 365*20]):#
        RV_dist = synthetic_RV_distribution(number_of_stars=big_sample_size, binary_fraction=0.7,
                                            min_period=period, sigma_dyn=2.0, **kwargs)

        sig1D = np.std(np.random.choice(RV_dist, (number_of_samples, sample_size))
                       + np.random.normal(0, measured_errors, (number_of_samples, sample_size)), axis=1)

        bin_edges = np.linspace(0, 500, nbins, endpoint=True)
        n, bins = np.histogram(sig1D, bin_edges)

        mode = bins[np.argmax(n)]
        median = np.median(sig1D)

        diff = np.abs(mode - measured_sigma)
        if diff < temp_sigma_diff:
            best_sig1D = sig1D
            best_period = period
            best_mode = mode
            temp_sigma_diff = diff

        fract = 0
        fract_array = []
        for b, f, i in zip(bins, n / float(np.sum(n)), range(len(bins))):
            fract += f
            fract_array.append(fract)

        perc16 = bins[fract_array.index(find_nearest(fract_array, 0.16))]
        perc84 = bins[fract_array.index(find_nearest(fract_array, 0.84))]
        perc05 = bins[fract_array.index(find_nearest(fract_array, 0.05))]

        diff_perc16 = np.abs(perc16 - measured_sigma)
        if diff_perc16 < temp_perc16_diff:
            perc16_sig1D = sig1D
            perc16_period = period
            best_perc16 = perc16
            temp_perc16_diff = diff_perc16

        diff_perc05 = np.abs(perc05 - measured_sigma)
        if diff_perc05 < temp_perc05_diff:
            perc05_sig1D = sig1D
            perc05_period = period
            best_perc05 = perc05
            temp_perc05_diff = diff_perc05

        print period, median, mode, perc16, perc05

    print best_period, best_mode
    print perc16_period, best_perc16
    print perc05_period, best_perc05

    print period, "it took %.4g s" % -(start - time.time())

    fig, (pax) = plt.subplots(1, 1, figsize=(6, 3))

    if best_period > 2 and best_period < 30:
        label_period = int(best_period)
        pax.hist(best_sig1D, histtype="step", bins=np.linspace(0, 500, nbins), density=True,
                 label=r"P$_{\rm "r"cutoff}$= %.1f d" % (label_period))
    elif best_period > 30 and best_period < 365:
        label_period = int(best_period / 30)
        pax.hist(best_sig1D, histtype="step", bins=np.linspace(0, 500, nbins), density=True,
                 label=r"P$_{\rm "r"cutoff}$= %.0f m" % (label_period))
    elif best_period > 365:
        label_period = int(best_period / 365)
        pax.hist(best_sig1D, histtype="step", bins=np.linspace(0, 500, nbins), density=True,
                 label=r"P$_{\rm "r"cutoff}$= %.0f y" % (label_period))

    pax.hist(perc16_sig1D, histtype="step", bins=np.linspace(0, 500, nbins), density=True,
             label=r"P$_{\rm "r"cutoff}$= %.1f m" % (int(perc16_period / 31)))

    pax.hist(perc05_sig1D, histtype="step", bins=np.linspace(0, 500, nbins), density=True,
             label=r"P$_{\rm "r"cutoff}$= %.0f d" % (perc05_period))

    pax.axvline(measured_sigma, ymax=0.2)

    pax.legend()
    pax.set_xlim(0, 50)
    pax.set_ylim(0, 0.2)
    pax.set_xlabel(r"$\sigma_{\rm 1D}$ [km s$^{-1}$]")
    pax.set_ylabel(r"Frequency [(km s$^{-1}$)$^{-1}$]")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # simple_std_plot_bigSample()
    # plot_dists(*initialize_parameters(), cumulative=True)
    # find_best_period()

    simple_std_plot_bigSample(measured_errors=RV_errors, min_mass=6)

    # compare_one_big_sample_vs_many_small_samples(binary_fraction=0.7)

    # std_search(velocities=rad_v_1, Pmin=1.5, max_period=100, Npoints=22)

    # inclination, eccentric_anomaly, primary_mass, period, mass_ratio, orbit_rotation, eccentricity = initialize_parameters()
    # plot_dists(inclination, eccentric_anomaly, primary_mass, period, mass_ratio, orbit_rotation, eccentricity, cumulative=True)

    # # plt.hist(np.log10(period/(24*3600)), cumulative=True, bins=50)
    # plt.hist(eccentricity, bins=50, cumulative=True, histtype='step', label='1')
    # # plt.hist(period, cumulative=True, bins=50)
    # plt.legend()
    # plt.show()

    # fbin_period_search(0, 1, 1.001, 500, number_of_stars=1 * 10**5, Npoints=200)

    # fbin_search(Nsamples=5000, Nstars=10**5, Npoints=100, fmin=0.5, fmax=1, errors=rad_v_1_err, velocities=rad_v_1, max_period=100)# number_of_stars=1000)
    # cdf_plot(fbin=1)
