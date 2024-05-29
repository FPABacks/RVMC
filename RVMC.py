"""
Script for simulating radial velocity measurements of clusters of stars.
By Frank Backs (frank.backs@kuleuven.be)
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.constants import G
from scipy.optimize import newton, brentq
from scipy import stats


def anomaly_function(E, e, M):
    """
    The equation to solve to obtain the eccentric anomaly.
    :param E:   The eccentric anomaly
    :param e:   The eccentricity
    :param M:   The mean anomaly /the time in the orbit times 2 * pi divided by the period
    :return:
    """
    return E - e * np.sin(E) - M


class RVMC:
    """
    A tool for randomly generating radial velocity distributions of binary and single stars in clusters
    """

    def __init__(self,
                 number_of_stars=10**5,
                 min_mass=6.,
                 max_mass=20.,
                 min_period=10.**0.15,
                 max_period=10.**3.5,
                 period_pi: float = -0.5,
                 mass_ratio_kappa: float = 0,
                 mass_power: float = -2.35,
                 min_q: float = 0.1,
                 max_q: float = 1.0,
                 e_power: float = -0.5,
                 mass_dist=None,
                 period_dist=None,
                 sigma_dyn=2.0,
                 fbin=0.7):
        """
        Initialize the properties of the single and binary stars in the sample. Note that mass_dist and period_dist
        should be supplied in SI units and that the size must be equal to int(number_of_stars * fbin)
        :param number_of_stars: Number of stars (default=10**5)
        :param min_mass:        Lowest mass of the stars in the cluster in solar masses (default=6)
        :param max_mass:        Highest mass of star in the cluseter in solar masses (default=20)
        :param min_period:      Shortest period of binaries in the cluster in days (default=10**0.15)
        :param max_period:      Longest period of binaries in the cluster in days (default=10**3.5)
        :param period_pi:       Power of the powerlaw sampling of the periods (of log(P)**pi in days)
        :param mass_ratio_kappa Power of the mass ratio sampling
        :param mass_power:      Power of the mass sampling (default -2.35, i.e. Kroupa)
        :param min_q:           Minimum mass ratio
        :param max_q:           Maximum mass ratio
        :param e_power:         Power of the eccentricity sampling power law.
        :param mass_dist:       An alternative mass distribution of the stars in the cluster in kg (default=None)
        :param period_dist:     An alternative period distribution of the stars in the cluster in seconds (default=None)
        :param fbin:            Binary fraction of the stars in the cluster (default=0.7)
        :param sigma_dyn:       Standard deviation of velocities due to cluster dynamics in km/s (default=2.)

        """
        self.fbin = fbin
        self.min_mass = min_mass
        self.max_mass = max_mass
        self.min_period = min_period
        self.max_period = max_period
        self.sigma_dyn = sigma_dyn
        self.number_of_stars = number_of_stars

        self.period_pi = period_pi
        self.mass_ratio_kappa = mass_ratio_kappa
        self.mass_power = mass_power
        self.min_q = min_q
        self.max_q = max_q
        self.e_power = e_power

        if type(mass_dist) == type(None):
            self.primary_mass = self.sample_masses()
        else:
            self.primary_mass = mass_dist

        if type(period_dist) == type(None):
            self.period = self.sample_period()
        else:
            self.period = period_dist

        self.time = self.sample_time()
        self.mass_ratio = self.sample_mass_ratio()
        self.eccentricity = self.sample_eccentricity()
        self.eccentric_anomaly = self.calculate_eccentric_anomaly()

        self.inclination = self.sample_inclination()
        self.orbit_rotation = self.sample_orbit_rotation()

        self.rv_binary = None
        self.radial_velocities = np.zeros(self.number_of_stars, dtype=float)
        self.sigma_1d = None

        self.cluster_velocities = np.random.normal(0.0, self.sigma_dyn, size=self.number_of_stars)

    def sample_powerlaw(self, min_val, max_val, power, size=None):
        """
        Samples a powerlaw between the min_val, max_val with power distribution power.
        :return:
        """
        if size is None:
            size = self.number_of_stars
        pmax = (max_val / min_val)**(power + 1)
        p = np.random.uniform(pmax, 1, size=size)
        return p**(1. / (power + 1)) * min_val

    def sample_inclination(self):
        """
        Returns inclinations for each binary as a np array in radians
        :return:
        """
        return np.arccos(np.random.random(size=self.number_of_stars))

    def sample_masses(self):
        """
        Returns an array of primary masses sample from a kroupa distribution in kg
        :return:
        """
        return self.sample_powerlaw(self.min_mass, self.max_mass, self.mass_power) * 2.e30

    def sample_period(self):
        """
        Returns a distribution of periods in seconds
        Distribution described by pdf ~ log(P)**-0.5 with 0.15 < log(P) < 3.5 with P in days
        :return:
        """
        return 24 * 3600 * 10**self.sample_powerlaw(np.log10(self.min_period), np.log10(self.max_period), self.period_pi)

    def sample_mass_ratio(self):
        """
        Returns an array of mass ratios
        """
        return self.sample_powerlaw(self.min_q, self.max_q, self.mass_ratio_kappa)

    def sample_orbit_rotation(self):
        """
        Returns an a array of random orbit rotations in radians
        """
        return np.random.random(size=self.number_of_stars) * 2 * np.pi

    def sample_eccentricity(self):
        """
        Returns an array of random eccentricities based on the period of the binaries
        For periods < 4 days e = 0
        for periods between 4 and 6 days e < 0.5
        for periods > 6 days e < 0.9
        """
        eccentricity = np.zeros(self.number_of_stars)
        more_than_6 = self.period > (6 * 24 * 3600)
        between_4_and_6 = (self.period > (4 * 24 * 3600)) * np.logical_not(more_than_6)
        eccentricity[more_than_6] = self.sample_powerlaw(1e-5, 0.9, self.e_power, size=np.sum(more_than_6))
        eccentricity[between_4_and_6] = self.sample_powerlaw(1e-5, 0.5, self.e_power, size=np.sum(between_4_and_6))
        # eccentricity = self.sample_powerlaw(1e-5, 0.9, self.e_power) # For testing purposes.
        return eccentricity

    def sample_time(self):
        """
        Samples random moments in the orbit of the binaries by sampling a time between 0 and period
        :return:
        """
        return np.random.random(size=self.number_of_stars) * self.period

    def calculate_eccentric_anomaly(self):
        """
        Calculates the eccentric anomaly. This has to be done by solving keplers equation numerically as there
        is no analytical solution.
        :return:
        """
        M = np.pi * 2 * self.time / self.period
        return newton(anomaly_function,
                      M,
                      args=(self.eccentricity, M),
                      tol=10**-6)

    def calculate_semi_major_axis(self):
        """
        Returns the total semi major axis of all binary systems in meters
        :return:
        """
        return ((4 * np.pi**2) / (G.value * (1 + self.mass_ratio) * self.primary_mass * self.period**2))**-(1. / 3.)

    def max_orbital_velocity(self, semi_major_axes):
        """
        Calculates the velocity of the primary in perihelion in km/s
        equation:
        v_0 = q / (1 + q) * ((G * (M + m) * (1 + e)) / ((1 - e) * a))**0.5
        :param semi_major_axes: Total semi major axis of the system
        :return: The maximum velocity of the primary. (that is at periastron.
        """
        return (self.mass_ratio * ((G.value * self.primary_mass * (1 + self.eccentricity)) /
                   ((1 + self.mass_ratio) * (1 - self.eccentricity) * semi_major_axes)) ** 0.5) * 0.001

    def binary_radial_velocity(self, v_max):
        """
        Calculates the radial velocity of the primary star of a binary given the eccentricity, inclination, orbit
        rotation (rotation of the semi major axis relative to the observer), eccentric anomaly and the maximal velocity
        of the primary. Note that in the equation here the velocity amplitude is obtained by multiplying the maximal
        orbital velocity by 1./(1 + e).
        :param v_max:
        :return:
        """
        return (1 / (1 + self.eccentricity)) * v_max * np.sin(self.inclination) *\
               (self.eccentricity * np.cos(self.orbit_rotation) +
                np.cos(2 * np.arctan((((1 + self.eccentricity) / (1 - self.eccentricity)) ** 0.5) *
                                     np.tan(0.5 * self.eccentric_anomaly)) + self.orbit_rotation))

    def calculate_binary_velocities(self):
        """
        Calculates the radial velocities of the cluster
        :return:
        """

        semi_major_axes = self.calculate_semi_major_axis()
        max_orbital_vels = self.max_orbital_velocity(semi_major_axes)
        rv_binary = self.binary_radial_velocity(max_orbital_vels)

        rv_binary += np.random.normal(0.0, self.sigma_dyn, size=self.number_of_stars)
        self.rv_binary = rv_binary
        return rv_binary

    def calculate_radial_velocity(self):
        """
        Combines single and binary stars to create a sample representing a cluster of stars with fbin binaries.
        :return:
        """
        number_of_binaries = int(self.number_of_stars * self.fbin)
        self.radial_velocities[:number_of_binaries] = np.random.choice(self.rv_binary, size=number_of_binaries, replace=False)
        self.radial_velocities[number_of_binaries:] = np.random.choice(self.cluster_velocities,
                                                                       size=self.number_of_stars - number_of_binaries)

    def subsample(self, sample_size=12, measured_errors=1.2, number_of_samples=10**5, fbin=None, weighted=False):
        """
        Takes many subsamples of the big sample cluster. Puts the resulting distribution in sigma_1d
        :param sample_size:         Number of stars in observed sample to compare to (integer, default=12)
        :param measured_errors:     Either the typical error on the measured radial velocity or each of the measured
                                    radial velocity errors, default=1.2.
        :param number_of_samples:   The number of subsamples to take
        :param fbin:                The binary fraction of the simulated cluster.
        :param weighted:            Adds turns the starndard deviation into a weighted standard deviation if set to True
                                    Only meaningful if an array of measured errors is provided. weights are
                                    1 / measured_error*2
        :return: An array of velocity dispersions, array with size=number_of_samples
        """
        if not isinstance(measured_errors, float):
            if sample_size != len(measured_errors):
                print("Changing the subsample size from %i to %i to match the measured errors!" %
                      (sample_size, len(measured_errors)))

        if not isinstance(fbin, type(None)):
            self.fbin = fbin

        self.calculate_radial_velocity()

        # Alternative way of calculating the
        if weighted:
            weights = 1 / measured_errors**2
            rvs = (np.random.choice(self.radial_velocities, (number_of_samples, sample_size)) +
                   np.random.normal(0, measured_errors, (number_of_samples, sample_size)))
            rv_mean = np.sum(rvs * weights, axis=1) / np.sum(weights)
            # self.sigma_1d = (np.sum(weights * (rvs.T - rv_mean).T**2, axis=1) / (((sample_size - 1.) / sample_size) * np.sum(weights)))**0.5
            self.sigma_1d = (np.sum(weights * (rvs.T - rv_mean).T**2, axis=1) / (np.sum(weights)))**0.5
        else:
            self.sigma_1d = np.std(np.random.choice(self.radial_velocities, (number_of_samples, sample_size)) +
                               np.random.normal(0, measured_errors, (number_of_samples, sample_size)), axis=1, ddof=1)

        return self.sigma_1d

    def check_initialization(self):
        """
        Plots the cumulative distributions of the orbit parameters to check if they look correct.
        :return:
        """
        fig, ((ax1, ax2, ax3),
              (ax4, ax5, ax6),
              (ax7, ax8, ax9)) = plt.subplots(3,3, figsize=(6,6))

        ax8.remove()
        ax9.remove()

        nbins = int(self.number_of_stars**0.333)

        ax1.hist(self.primary_mass / 2.e30, histtype="step", cumulative=True,
                 density=True, bins=np.geomspace(self.min_mass, self.max_mass, nbins))
        ax1.set_xlabel("$M_1 [$M$_\odot]$")
        ax1.set_title("Primary mass")
        ax1.set_xscale("log")

        ax2.hist(self.period / (3600. * 24), histtype="step", cumulative=True,
                 density=True, bins=np.geomspace(self.min_period, self.max_period, nbins))
        ax2.set_xlabel("$P$ [days]")
        ax2.set_title("Period")
        ax2.set_xscale("log")

        ax3.hist(self.mass_ratio, histtype="step", cumulative=True, density=True,
                 bins=np.linspace(0, 1, nbins))
        ax3.set_xlabel("q [-]")
        ax3.set_title("Mass ratio")

        ax4.hist(self.eccentricity, histtype="step", cumulative=True, density=True,
                 bins=np.linspace(0, 1, nbins))
        ax4.set_xlabel("e [-]")
        ax4.set_title("Eccentricity")

        ax5.hist(self.inclination, histtype="step", cumulative=True, density=True,
                 bins=np.linspace(0, np.pi * 0.5, nbins))
        ax5.set_xlabel("$i$ [radians]")
        ax5.set_title("Inclination")

        ax6.hist(self.orbit_rotation, histtype="step", cumulative=True, density=True,
                 bins=np.linspace(0, 2 * np.pi, nbins))
        ax6.set_xlabel("$\omega$ [radians]")
        ax6.set_title("Orbit rotation")

        ax7.hist(self.eccentric_anomaly, histtype="step", cumulative=True, density=True,
                 bins=np.linspace(0, 2 * np.pi, nbins))

        ax7.set_xlabel("$E$ [radians]")
        ax7.set_title("Eccentric anomaly")

        plt.tight_layout()
        plt.show()


def plot_sigma_1d(sigma1d, labels, colors=None, hide_median=False):
    """
    Plots the distribution of
    :param sigma1d:
    :param labels:
    :return:
    """
    if type(colors) == type(None):
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        if len(sigma1d) > 10:
            print("Please specify your own colors if more than 10 distributions are to be plotted")
            return

    n_distributions = len(sigma1d)
    if len(labels) != n_distributions:
        print("Warning! Number of labels does not match number of distributions")
        return

    fig, ax = plt.subplots(1,1, figsize=(6,5))

    for i, dist in enumerate(sigma1d):
        ax.hist(dist, histtype="step", bins=np.linspace(0, 100, 200), label=labels[i], density=True, color=colors[i], lw=1.5)
        if len(sigma1d) < 1000 and not hide_median:
            ax.axvline(np.median(dist), color=colors[i], zorder=0.1, ls="--")

    ax.set_xlabel(r"$\sigma_{\rm 1D}$ [km s$^{-1}$]")
    ax.set_ylabel(r"Frequency [(km s$^{-1}$)$^{-1}$]")

    if len(labels) <= 10:
        ax.legend()
    ax.set_xlim([0, 50])
    # ax.set_ylim([0, 0.1])
    plt.tight_layout()
    # plt.show()
    return fig, ax


if __name__ == "__main__":
    # Example of generating many sigma1d distributions for many binary fractions.
    import matplotlib.cm as cmx
    from matplotlib.colors import Normalize
    from tqdm import tqdm
    import time

    # Some text settings I like to use.
    # plt.rcParams['text.usetex'] = True
    # plt.rcParams["font.family"] = "serif"
    # plt.rcParams["font.serif"] = "Latin Modern Math"

    print("Generating sample")
    start = time.time()
    rv = RVMC(number_of_stars=10**6)
    rv.calculate_binary_velocities()
    print(f"It took {time.time() - start:.2f} seconds to generate radial velocities")

    print("Generating subsamples")
    fbins = np.linspace(0, 1, 11)
    dists = [rv.subsample(fbin=fbin, number_of_samples=10**6) for fbin in tqdm(fbins)]
    labels = [r"$f_{\rm bin}$ = %.3g" % fbin for fbin in fbins]

    norm = Normalize(0, 1)
    scalarmap = cmx.ScalarMappable(norm=norm, cmap="viridis")
    scalarmap._A = []
    colors = [scalarmap.to_rgba(fbin) for fbin in fbins]

    print("Plotting")
    fig, ax = plot_sigma_1d(dists, labels, colors=colors)
    cb = plt.colorbar(scalarmap, ax=ax)
    cb.set_label("Binary fraction")
    plt.xlim([0, 100])
    plt.tight_layout()
    plt.show()

    # Testing eccentricities, runs the code for differing eccentricity distributions.
    sigma1d = []
    e_powers = np.linspace(-0.5, 3, 11)
    for e_power in tqdm(e_powers):
        rv = RVMC(number_of_stars=10**6, e_power=e_power)#, min_period=10)
        rv.calculate_binary_velocities()
        sigma1d.append(rv.subsample(fbin=0.7, number_of_samples=10 ** 6))

    labels = [rf"e power = {e_power:.3g}" for e_power in e_powers]

    norm = Normalize(e_powers[0], e_powers[-1])
    scalarmap = cmx.ScalarMappable(norm=norm, cmap="viridis")
    scalarmap._A = []
    colors = [scalarmap.to_rgba(e_power) for e_power in e_powers]

    print("Plotting")
    fig, ax = plot_sigma_1d(sigma1d, labels, colors=colors)
    cb = plt.colorbar(scalarmap, ax=ax)
    cb.set_label("e power")
    plt.xlim([0, 60])
    # plt.savefig("eccentricity test.pdf")
    plt.show()
