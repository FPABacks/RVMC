import numpy as np
import matplotlib.pyplot as plt
from astropy.constants import G
from scipy.optimize import brentq
from scipy import stats
# from matplotlib.animation import FuncAnimation
from matplotlib.animation import FuncAnimation
import matplotlib.ticker

# FuncAnimation = animation.FuncAnimation

# https://www.overleaf.com/12363089pdxxmvywvxgk#/47036402/

v_bulk = -32.99  # km / s (from Frinchaboy 2008)

# velocities = np.array([-19.4, -17.7, -1.7, -39.8, -7.9, -34.1, -35.1, -44.1, 10.9, -32.2, 13.0, -50.4, -45.4, -33.5, -44.1, -26.5, -27.6, -35.5, -33.6, -30.2, -36.8, -43.2])

# errors = np.array([20.2, 8.4, 1.5, 8.3, 5.6, 0.1, 7.7, 6.5, 8.9, 7.9, 12.8, 25.0, 8.0, 3.3, 9.5, 8.5, 0.11, 14.6, 10.6, 4.1, 8.3, 7.9])

rad_v_1 = np.array([6.3, -19.6, -1.7, -42.2, -32.9, -34.1, -36.4, -45.8, 8.5, -35.0, -19.9, -40.9, -46.1, -33.9, -42.6, -28.5, -27.59, -39.3, -35.0, -29.5, -36.8, -42.0])
rad_v_1_err = np.array([1.0, 1.6, 1.5, 1.3, 0.5, 0.1, 1.0, 0.9, 1.1, 1.0, 1.9, 6.3, 1.5, 0.5, 2.0, 0.5, 0.11, 4.1, 2.0, 0.4, 1.4, 1.8])

rad_v_2 = np.array([7.1, -19.6, -1.7, -30.5, -10.2, -34.7, -36.4, -45.8, 8.5, -58.6, -19.9, -42.9, -31.9, -34.1, -42.6, -28.5, -27.9, -39.3, -35.0, -29.5, -36.8, -41.0])
rad_v_2_err = np.array([0.7, 1.6, 1.5, 1.3, 0.5, 0.2, 1.0, 0.9, 1.1, 1.5, 1.9, 4.3, 1.6, 0.4, 2.0, 0.5, 0.09, 4.1, 2.0, 0.4, 1.4, 1.8])


def initialize_parameters(number_of_stars=100000, min_mass=3, max_mass=20, min_period=1.4, max_period=3500):
    inclination = np.random.uniform(size=number_of_stars) * np.pi * 0.5
    # phase = np.cos(np.random.uniform(size=number_of_stars) * np.pi * 2)
    a = 1.3  # IMF powerlaw index -1 because of sample method
    primary_mass = np.random.uniform(min_mass**(-a), max_mass**(-a), size=number_of_stars) ** -(1. / a) * 2e30

    # period = 10 ** np.random.uniform(0.15 ** -2, 3.5 ** -2, size=number_of_stars) ** -0.5 * 24 * 3600
    # print np.log10(min_period)**-2,  np.log10(max_period)**-2

    period = 10**(np.random.uniform(np.log10(min_period)**0.5, np.log10(max_period)**0.5, number_of_stars)**2) * 24 * 3600

    # period = np.array([])
    #
    # while len(period) < number_of_stars:
    #     period = np.concatenate([period, f_P(min_period, max_period, max([number_of_stars, 100])) * 3600 * 24])
    #     # print period.shape, number_of_stars, len(period) < number_of_stars
    # period = period[:number_of_stars]


    # mass_ratio = np.random.uniform(0.1**(1./100), 1, size=number_of_stars)**100
    mass_ratio = np.random.uniform(0.1, 1, number_of_stars)
    # mass_ratio = np.random.uniform(0.1 ** (1. / 1.1), 1, number_of_stars) ** 1.1

    orbit_rotation = np.random.uniform(0, 1, size=number_of_stars) * 2 * np.pi
    eccentricity = np.random.uniform(0, 1, size=number_of_stars)**2
    time = np.random.uniform(0, 1, size=number_of_stars) * period
    eccentric_anomaly = find_eccentric_anomaly(eccentricity, time, period)

    return inclination, eccentric_anomaly, primary_mass, period, mass_ratio, orbit_rotation, eccentricity


def f_P(pmin, pmax, N):
    """
    Rejection sampling for the Period, simplest way to do it...  
    """
    # x = np.linspace(pmin, pmax, 10000)
    # y = np.log10(x) ** -0.5
    # norm = (np.sum(y) * (x[1] - x[0]))
    y_max = np.log10(pmin) ** -0.5
    P = np.random.uniform(pmin, pmax, size=N * 2)
    f = np.random.uniform(0, y_max, size=N * 2)
    return P[f < np.log10(P)**-0.5]


def anomaly(E, e, t, P):
    return E - e * np.sin(E) - t * 2 * np.pi / P


def find_eccentric_anomaly(eccentricity, time, period):
    eccentric_anomaly = np.zeros(len(time))
    for i in range(len(time)):
        eccentric_anomaly[i] = brentq(anomaly, 0, np.pi * 2, args=(eccentricity[i], time[i], period[i]))
    return eccentric_anomaly


def max_orbital_velocity(primary_mass, mass_ratio, eccentricity, semi_major_axes):
    return mass_ratio / (1 + mass_ratio) * (G.value * primary_mass * (1 + mass_ratio) * (1 + eccentricity) /
                                            ((1 - eccentricity) * semi_major_axes))**0.5 / 1000.


def binary_radial_velocity(v_max, i, e, w, E):
    return v_max * np.sin(i) * (e * np.cos(w) + np.cos(2 * np.arctan(((1 + e) / (1 - e))**0.5 * np.tan(0.5 * E)) + w))


def semi_major_axis(period, primary_mass, mass_ratio):
    return (4 * np.pi ** 2 / (G.value * (1 + mass_ratio) * primary_mass * period ** 2)) ** -0.333


def synthetic_RV_distribution(number_of_stars=12, min_mass=6, max_mass=20, binary_fraction=0.7, min_period=1.4, max_period=3500):

    # minimum and maximum period in days.

    RV = np.zeros(number_of_stars)

    # Should the number of binaries be random?? I am trying to fit this number, now there is a random deviation in it.
    # binaries = np.random.uniform(size=number_of_stars) > (1 - binary_fraction)
    # number_of_binaries = int(binary_fraction * number_of_stars)
    number_of_binaries = np.sum(np.random.uniform(0, 1, number_of_stars) < binary_fraction)

    cluster_velocities = np.random.normal(0.0, 2.0, size=number_of_stars) + v_bulk

    inclination, eccentric_anomaly, primary_mass, period, mass_ratio, orbit_rotation, eccentricity = \
        initialize_parameters(number_of_stars=number_of_binaries, min_mass=min_mass, max_mass=max_mass,
                              min_period=min_period, max_period=max_period)

    semi_major_axes = semi_major_axis(period, primary_mass, mass_ratio)

    v_orb_max = max_orbital_velocity(primary_mass, mass_ratio, eccentricity, semi_major_axes)

    RV_binary = binary_radial_velocity(v_orb_max, inclination, eccentricity, orbit_rotation, eccentric_anomaly)

    RV[:number_of_binaries] = RV_binary + cluster_velocities[:number_of_binaries]
    RV[number_of_binaries:] = cluster_velocities[number_of_binaries:]

    # print np.sum(np.abs(RV - v_bulk) > 50) / float(number_of_stars)
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


def fbin_search(velocities=rad_v_1, fmin=0, fmax=1, Npoints=100, Nstars=10**4, Nsamples=1000, errors=rad_v_1_err, **kwargs):

    binary_fractions = np.linspace(fmin, fmax, Npoints, endpoint=True)
    print binary_fractions


    # fig, axarr = plt.subplots(1, 2, sharex=True)
    # axarr[0].set_xlabel("Binary fraction")
    # axarr[0].set_ylabel("Probability")
    # axarr[0].set_title("Pvalues")
    #
    # axarr[1].set_title("KS value")
    # axarr[1].set_xlabel("Binary fraction")
    # axarr[1].set_ylabel("KS value")

    print "Generating distributions"
    synth_dists = np.zeros((Npoints, Nstars))
    for i, fbin in enumerate(binary_fractions):
        synth_dists[i] = synthetic_RV_distribution(binary_fraction=fbin, number_of_stars=Nstars, **kwargs)
        print "Progress: %i %%" % i

    # fig = plt.figure()
    # hist = plt.hist(synth_dists[i], bins=np.linspace(-400, 400, 500), histtype="step", normed=True, cumulative=True)  # , log=True)
    # plt.xlabel("Radial velocity (km s$^-1$)")
    # plt.ylabel("$f\ (RV)$")
    # plt.xlim([-400, 400])
    # # plt.ylim([0, 0.02])
    #
    # ani = FuncAnimation(fig, update_hist, interval=100, frames=Npoints, repeat=True, fargs=(synth_dists,))
    # # ani.save("hist.mp4", fps=60)
    # plt.show()

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
        
        #fig = plt.figure()
        #global ax
        #ax = plt.plot(binary_fractions, pvals[:,0])
        #plt.ylim([0, 1])
        #plt.xlabel("Binary fraction")
        #plt.ylabel("p-value")

        #ani = FuncAnimation(fig, update, frames=Nsamples, interval=50, fargs=(pvals,binary_fractions))
        #plt.show()

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


def fbin_period_search(fmin, fmax, pmin, pmax, Npoints=100, velocities=rad_v_1, **kwargs):

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
    
    fig.savefig("fancy_plaatje_Np200_P500.pdf")
    fig.savefig("fancy_plaatje_Np200_P500.png")
    plt.show()


def update_hist(i, data):
    plt.cla()
    plt.hist(data[i], bins=np.linspace(-400, 400, 500), histtype="step", normed=True, cumulative=True)  # , log=True)
    plt.xlabel("Radial velocity (km s$^-1$)")
    plt.ylabel("$f\ (RV)$")
    plt.xlim([-400, 400])
    # plt.ylim([0, 0.02])


def update(i, pvals, bf):
    # print "hoi"
    # print bf, pvals[:,i]
    ax[0].set_data(bf, pvals[:,i])
    return ax,


def cdf_plot(fbin=0.7):

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
    

def std_search(velocities=rad_v_1, Nsamples=10**5, fmin=0.0, fmax=1, Npoints=100, Pmin=1.5, **kwargs):
    """
    Uses the randomness of the observed sample to determine the range of possible binaryt fractions.
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


simple_std_plot(min_mass=6)



# std_search(velocities=rad_v_1, Pmin=1.5, max_period=100, Npoints=22)


# fbin_period_search(0, 1, 1.001, 500, number_of_stars=1 * 10**5, Npoints=200)

# fbin_search(Nsamples=5000, Nstars=10**5, Npoints=100, fmin=0.5, fmax=1, errors=rad_v_1_err, velocities=rad_v_1, max_period=100)# number_of_stars=1000)
# cdf_plot(fbin=1)


"""
period = 2 * 24 * 3600
primary_mass = 5e30
mass_ratio = 0.5
eccentricity = 0.9
phase = np.zeros(10000)
#inclination, phase, primary_mass, period, mass_ratio, orbit_rotation, eccentricity = \
#    initialize_parameters(number_of_stars=10000)

#for i in range(10000):
#    phase[i] = make_phase(period, primary_mass, mass_ratio, eccentricity)


v2 = orbital_velocity2(period, primary_mass, mass_ratio, eccentricity, phase)
v3 = orbital_velocity3(period, primary_mass, mass_ratio, eccentricity, phase)

#plt.plot(phase1, 1000. / 2 / np.pi / np.sum(1/v1) / v1)
a = semi_major_axis(period, primary_mass, mass_ratio)
r = distance_r(phase, a, eccentricity)

plt.hist(r)
plt.show()

plt.hist((1 / np.sum(r / v2)) * r / v2, histtype="step")
plt.hist((1 / np.sum(r / v3)) * r / v3, histtype="step")
#plt.plot(v2)
plt.show()

plt.hist(phase)
plt.show()

"""

# Mathematica ding
# # Integrate[Sqrt[(a^2 + a Cos[x])/((1 - a^2) (1 + a Cos[x])^2)], {x, 0, 2pi}]
#
# RV1 = synthetic_RV_distribution(number_of_stars=10**5, min_mass=3, binary_fraction=0.7)
# ##RV2 = synthetic_RV_distribution(number_of_stars=100000, min_mass=6, binary_fraction=0.3)
# # Data3 = np.array([-60.52,-32.96,-25.69,51.8,-8.35,-37.62,-34.45,45.38,-44.85,-28.9,-32.88,-24.72,-51.17,-45.23,\
# #                  10.46,-10.66,-33.26])
# print "Stanard deviation of velocities: %.3g" % np.std(RV1, ddof=1)
#
# #plot_RV_dist([RV1])
# RV_std = []
# for i in range(10000):
#     RV_std.append(np.std(synthetic_RV_distribution(10, min_mass=3, binary_fraction=0.7), ddof=1))
#
# print np.mean(RV_std), np.std(RV_std, ddof=1)


# parameter_check()
# peak at 0.22


