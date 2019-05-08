import numpy as np
import matplotlib.pyplot as plt
from astropy.constants import G
from scipy.optimize import brentq
from scipy import stats
# from matplotlib.animation import FuncAnimation
from matplotlib.animation import FuncAnimation
import matplotlib.ticker


def initialize_parameters(number_of_stars=100000, min_mass=3, max_mass=20, min_period=1.4, max_period=3500):
    """
    Create the random samples following their respective distributions.
    :param number_of_stars:     (int) number of stars (duh)
    :param min_mass:            (scalar) minimum mass in solar masses
    :param max_mass:            (scalar) maximum mass in solar masses
    :param min_period:          (scalar) minimum period in days
    :param max_period:          (scalar) maximum period in days.
    :return:    Note returns the masses in kg and period in seconds.
    """
    inclination = np.random.uniform(size=number_of_stars) * np.pi * 0.5
    a = 1.3  # IMF powerlaw index -1 because of sample method
    primary_mass = np.random.uniform(min_mass**(-a), max_mass**(-a), size=number_of_stars) ** -(1. / a) * 2e30

    period = 10**(np.random.uniform(np.log10(min_period)**0.5, np.log10(max_period)**0.5, number_of_stars)**2) * 24 * 3600
    mass_ratio = np.random.uniform(0.1, 1, number_of_stars)
    # In case you want a mass ratio distribution as f(q) ~ q**-0.1
    # mass_ratio = np.random.uniform(0.1 ** (1. / 1.1), 1, number_of_stars) ** 1.1

    orbit_rotation = np.random.uniform(0, 1, size=number_of_stars) * 2 * np.pi
    eccentricity = np.random.uniform(0, 1, size=number_of_stars)**2
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
        eccentric_anomaly[i] = brentq(anomaly, 0, np.pi * 2, args=(eccentricity[i], time[i], period[i]), xtol=10**-2)
    return eccentric_anomaly


def max_orbital_velocity(primary_mass, mass_ratio, eccentricity, semi_major_axes):
    return mass_ratio / (1 + mass_ratio) * (G.value * primary_mass * (1 + mass_ratio) * (1 + eccentricity) /
                                            ((1 - eccentricity) * semi_major_axes))**0.5 / 1000.


def binary_radial_velocity(v_max, i, e, w, E):
    return v_max * np.sin(i) * (e * np.cos(w) + np.cos(2 * np.arctan(((1 + e) / (1 - e))**0.5 * np.tan(0.5 * E)) + w))


def semi_major_axis(period, primary_mass, mass_ratio):
    return (4 * np.pi ** 2 / (G.value * (1 + mass_ratio) * primary_mass * period ** 2)) ** -0.333


def synthetic_RV_distribution(number_of_stars=12, min_mass=6, max_mass=20, binary_fraction=0.7, min_period=1.4, max_period=3500):
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

    # minimum and maximum period in days.

    RV = np.zeros(number_of_stars)

    # The number of binaries is randomly determined based on the binary fraction. It can also be a fixed number:
    # number_of_binaries = int(binary_fraction * number_of_stars)
    number_of_binaries = np.sum(np.random.uniform(0, 1, number_of_stars) < binary_fraction)

    # Normally distributed cluster velocities around 0, currently with sigma_1D_cluster = 2 km/s
    cluster_velocities = np.random.normal(0.0, 2.0, size=number_of_stars)

    # generate orbital parameters and stellar properties. Note: only for the binary stars!
    inclination, eccentric_anomaly, primary_mass, period, mass_ratio, orbit_rotation, eccentricity = \
        initialize_parameters(number_of_stars=number_of_binaries, min_mass=min_mass, max_mass=max_mass,
                              min_period=min_period, max_period=max_period)

    # Get the radial velocities of the binaries
    semi_major_axes = semi_major_axis(period, primary_mass, mass_ratio)
    v_orb_max = max_orbital_velocity(primary_mass, mass_ratio, eccentricity, semi_major_axes)
    RV_binary = binary_radial_velocity(v_orb_max, inclination, eccentricity, orbit_rotation, eccentric_anomaly)

    RV[:number_of_binaries] = RV_binary + cluster_velocities[:number_of_binaries]
    RV[number_of_binaries:] = cluster_velocities[number_of_binaries:]

    return RV


def plot_RV_dist(RV_array):
    for RV in RV_array:
        plt.hist(RV, bins=np.linspace(-400, 400, 50), histtype="step", normed=True)  # , log=True)
    plt.xlabel("Radial velocity (km s$^-1$)")
    plt.ylabel("$f\ (RV)$")
    plt.xlim([-400, 400])
    plt.ylim([0, 0.004])
    plt.show()


def ks(synthesized, observed, **kwargs):
    """
    Performs a Kolmogorov-Smirnov test to compare the measured distribution to the synthesized distribution.
    """
    # print "fbin: ", fbin
    # np.random.seed(1234)  # Use seed for consistency? (THIS MAKES REPETITIONS USELESS!)
    # synthesized = synthetic_RV_distribution(binary_fraction=fbin, **kwargs)
    ksval, pval = stats.ks_2samp(observed, synthesized)
    # print "ks_val: ", ks_val
    return ksval, pval


def fbin_search(velocities=[], fmin=0, fmax=1, Npoints=100, Nstars=10**4, Nsamples=1000, errors=[], **kwargs):
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
                ksval, pval = ks(synth_dists[i], bootstrap_vels[:,sample_index], **kwargs)
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

            ksvals[i,j] = ksval
            pvals[i,j] = pval

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


def cdf_plot(fbin=0.7):
    """
    Make a cdf of a distribution for a given binary fraction, currently requires global variables rad_v_1 and rad_v_2 at the moment...
    :param fbin:
    :return:
    """

    synthesized1 = synthetic_RV_distribution(number_of_stars=10**5, min_period=1.5, binary_fraction=fbin)
    #synthesized2 = synthetic_RV_distribution(number_of_stars=10 ** 5, min_period=10, binary_fraction=0.75)

    bins = np.linspace(-300, 300, 10**4)
    plt.hist(rad_v_1, histtype="step", cumulative=True, bins=bins, lw=2, color="green", label="Sample 1", normed=True)
    plt.hist(rad_v_2, histtype="step", cumulative=True, bins=bins, lw=2, color="blue", label="Sample 2", normed=True)
    plt.hist(synthesized1, histtype="step", cumulative=True, bins=bins, lw=2, color="red", label="Synthetic", normed=True)
    # plt.hist(synthesized2, histtype="step", cumulative=True, bins=bins, lw=2, color="orange", label="Synthetic, fbin=0.4",
    #          normed=True)
    plt.xlim([-75, 15])
    plt.xlabel("$v_r$ [km s$^{-1}$]")
    plt.ylabel("Cumulative distribution")
    plt.legend(loc="upper left")
    plt.show()
    

def std_search(velocities=[], Nsamples=10**5, fmin=0.0, fmax=1, Npoints=100, Pmin=1.5, **kwargs):
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
            stds[i,j] = np.std(synthetic_RV_distribution(number_of_stars=sample_size, binary_fraction=fbin, min_period=Pmin, **kwargs))

    np.save("stds.npy", stds)
    np.save("fbins", binary_fractions)

    low_bound_fbin = 1
    high_bound_fbin = 0
    low_i = 1
    high_i=0

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
    plt.hist(stds[low_i], bins=bins, histtype="step", label="$f_{bin} = %.2g$" % low_bound_fbin, normed=True)
    plt.hist(stds[high_i], bins=bins, histtype="step", label="$f_{bin} = %.2g$" % high_bound_fbin, normed=True)
    plt.axvline(obs_std, lw=2, color='red', alpha=0.5, label="Observed")
    plt.xlabel("$\sigma_{1D}$ [km s$^{-1}$]")
    plt.ylabel("Frequency [(km s$^{-1}$)$^{-1}$]")
    plt.xlim([0,50])
    plt.legend()
    plt.show()


def simple_std_plot(number_of_samples=10**5, **kwargs):
    """
    Remake the plot of the M17 paper.
    :param number_of_samples:
    :param kwargs:
    :return:
    """

    nbins=1000

    fig, (fax, pax) = plt.subplots(1, 2, figsize=(6,3))

    nstars = 12

    sig1D1 = np.array([np.std(synthetic_RV_distribution(nstars, binary_fraction=0.7, **kwargs)) for i in range(number_of_samples)])
    sig1D2 = np.array([np.std(synthetic_RV_distribution(nstars, binary_fraction=0.28, **kwargs)) for i in range(number_of_samples)])
    sig1D3 = np.array([np.std(synthetic_RV_distribution(nstars, binary_fraction=0.12, **kwargs)) for i in range(number_of_samples)])

    fax.hist(sig1D1, histtype="step", bins=np.linspace(0,500,nbins), density=True, label=r"f$_{\rm bin}$=0.70")
    fax.hist(sig1D2, histtype="step", bins=np.linspace(0,500,nbins), density=True, label=r"f$_{\rm bin}$=0.28")
    fax.hist(sig1D3, histtype="step", bins=np.linspace(0,500,nbins), density=True, label=r"f$_{\rm bin}$=0.12")
    fax.legend()
    fax.set_xlim([0,50])
    fax.set_ylim([0,0.1])
    fax.set_xlabel(r"$\sigma_{\rm 1D}$ [km s$^{-1}$]")
    fax.set_ylabel(r"Frequency [(km s$^{-1}$)$^{-1}$]")

    sig1D1 = np.array([np.std(synthetic_RV_distribution(nstars, min_period=1.4, **kwargs)) for i in range(number_of_samples)])
    sig1D2 = np.array([np.std(synthetic_RV_distribution(nstars, min_period=30, **kwargs)) for i in range(number_of_samples)])
    sig1D3 = np.array([np.std(synthetic_RV_distribution(nstars, min_period=8*365, **kwargs)) for i in range(number_of_samples)])

    pax.hist(sig1D1, histtype="step", bins=np.linspace(0,500,nbins), density=True, label=r"P$_{\rm cutoff}$=1.4 d")
    pax.hist(sig1D2, histtype="step", bins=np.linspace(0,500,nbins), density=True, label=r"P$_{\rm cutoff}$=30 d")
    pax.hist(sig1D3, histtype="step", bins=np.linspace(0,500,nbins), density=True, label=r"P$_{\rm cutoff}$=8 yr")
    pax.legend()
    pax.set_xlim([0,50])
    pax.set_ylim([0,0.1])
    pax.set_xlabel(r"$\sigma_{\rm 1D}$ [km s$^{-1}$]")
    pax.set_ylabel(r"Frequency [(km s$^{-1}$)$^{-1}$]")

    plt.tight_layout()
    plt.savefig("sigma1D_test1.pdf")
    plt.show()


def plot_dists(inclination, eccentric_anomaly, primary_mass, period, mass_ratio, orbit_rotation, eccentricity):
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
    fig, axarr = plt.subplots(3,3, figsize=(8,8))

    axarr[0][0].hist(inclination, bins=int(N**0.333), density=True)
    axarr[0][0].set_xlabel("inclination")

    axarr[0][1].hist(eccentric_anomaly, bins=int(N**0.333), density=True)
    axarr[0][1].set_xlabel("eccentric anomaly")

    axarr[0][2].hist(primary_mass, bins=int(N**0.333), density=True)
    axarr[0][2].set_xlabel("primary_mass")

    axarr[1][0].hist(period, bins=np.geomspace(np.min(period), np.max(period), int(N**0.333)), density=True)
    axarr[1][0].set_xlabel("period")
    axarr[1][0].set_xscale("log")

    axarr[1][1].hist(mass_ratio, bins=int(N**0.333), density=True)
    axarr[1][1].set_xlabel("mass_ratio")

    axarr[1][2].hist(orbit_rotation, bins=int(N**0.333), density=True)
    axarr[1][2].set_xlabel("orbit_rotation")

    axarr[2][0].hist(eccentricity, bins=int(N**0.333), density=True)
    axarr[2][0].set_xlabel("eccentricity")

    plt.tight_layout()
    plt.show()


def compare_one_big_sample_vs_many_small_samples(number_of_samples=10**5, sample_size=12, big_sample_size=10**5, **kwargs):
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
    plt.xlim([0,50])
    plt.ylim([0,0.1])
    plt.xlabel(r"$\sigma_{\rm 1D}$ [km s$^{-1}$]")
    plt.ylabel(r"Frequency [(km s$^{-1}$)$^{-1}$]")
    plt.show()

# simple_std_plot(min_mass=6)

compare_one_big_sample_vs_many_small_samples(binary_fraction=0.3)


# std_search(velocities=rad_v_1, Pmin=1.5, max_period=100, Npoints=22)


# fbin_period_search(0, 1, 1.001, 500, number_of_stars=1 * 10**5, Npoints=200)

# fbin_search(Nsamples=5000, Nstars=10**5, Npoints=100, fmin=0.5, fmax=1, errors=rad_v_1_err, velocities=rad_v_1, max_period=100)# number_of_stars=1000)
# cdf_plot(fbin=1)
