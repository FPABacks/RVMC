import numpy as np
import matplotlib.pyplot as plt
from astropy.constants import G
from scipy.optimize import brentq
from scipy import stats
import matplotlib.ticker


def anomaly_function(E, e, t, P):
    """
    The equation to solve to obtain the eccentric anomaly.
    :param E:   The eccentric anomaly
    :param e:   The eccentricity
    :param t:   The time in the orbit
    :param P:   The period of the binary
    :return:
    """
    return E - e * np.sin(E) - t * 2 * np.pi / P


class RVMC:
    """
    A tool for randomly generating radial velocity distributions of binary and single stars in clusters
    """

    def __init__(self, number_of_stars=10**5, min_mass=6, max_mass=20, min_period=10.**0.15, max_period=10.**3.5,
                 mass_dist=None, period_dist=None, sigma_dyn=2.0, numerical_orbits=False, fbin=0.7):
        """
        Initialize the properties of the single and binary stars in the sample. Note that mass_dist and period_dist
        should be supplied in SI units and that the size must be equal to int(number_of_stars * fbin)
        :param number_of_stars:
        :param min_mass:
        :param max_mass:
        :param min_period:
        :param max_period:
        :param mass_dist:
        :param period_dist:
        :param fbin:
        :param sigma_dyn:
        """
        self.fbin = fbin
        # self.number_of_stars = number_of_stars
        self.min_mass = min_mass
        self.max_mass = max_mass
        self.min_period = min_period
        self.max_period = max_period
        self.sigma_dyn = sigma_dyn
        self.numerical_orbits = numerical_orbits
        self.number_of_stars = number_of_stars

        if not numerical_orbits:
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
        a = 2.3
        # CDF sampling of the IMF powerlaw
        pmin = (self.min_mass / self.min_mass) ** (-a + 1)
        pmax = (self.max_mass / self.min_mass) ** (-a + 1)
        p = np.random.uniform(pmax, pmin, size=self.number_of_stars)
        primary_mass = p ** (1. / (-a + 1)) * self.min_mass * 2e30
        return primary_mass

    def sample_period(self):
        """
        Returns a distribution of periods in seconds
        Distribution described by pdf ~ log(P)**0.5 with 0.15 < log(P) < 3.5 with P in days
        :return:
        """
        return 10 ** (np.random.uniform(np.log10(self.min_period) ** 0.5, np.log10(self.max_period) ** 0.5,
                                        self.number_of_stars) ** 2) * 24 * 3600

    def sample_mass_ratio(self):
        """
        Returns an array of mass ratios
        """
        return np.random.uniform(0.1, 1., self.number_of_stars)

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
        index6d = (self.period > (4 * 24 * 3600)) & (self.period < (6 * 24 * 3600))
        index_rest = self.period > (6 * 24 * 3600)
        eccentricity[index6d] = np.random.uniform(0 ** 0.5, 0.5 ** 0.5, np.sum(index6d)) ** 2
        eccentricity[index_rest] = (np.random.uniform(0 ** 0.5, 0.9 ** 0.5, np.sum(index_rest)) ** 2)
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
        eccentric_anomaly = np.zeros(self.number_of_stars)
        for i in range(self.number_of_stars):
            eccentric_anomaly[i] = brentq(anomaly_function, 0, np.pi * 2,
                                          args=(self.eccentricity[i], self.time[i], self.period[i]),
                                          xtol=10**-2)
        return eccentric_anomaly

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

        if self.numerical_orbits:  # Get velocities from pre-calculated file
            try:
                data = np.load("Numerical_RV/radial_velocities.npy")
            except IOError as err:
                print err
                print "Cannot find the file with the binary velocities, did you make it first?"
                print "You might need to run numerical_2_body.py in Numerical_RV/"
                exit()

            # number_of_binaries = int(self.number_of_stars * self.fbin)
            t_steps = np.random.choice(np.arange(data.shape[0]), size=self.number_of_stars, replace=True)
            stars = np.random.choice(np.arange(data.shape[1]), size=self.number_of_stars, replace=True)

            rv_binary = data[t_steps, stars, 0] * np.cos(self.orbit_rotation) +\
                        data[t_steps, stars, 1] * np.sin(self.orbit_rotation)
            rv_binary *= np.sin(self.inclination) * 0.001

        else:
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

    def subsample(self, sample_size=12, measured_errors=1.2, number_of_samples=10**5, fbin=0.7):
        """
        Takes many subsamples of the big sample cluster. Puts the resulting distribution in sigma_1d
        :param sample_size:
        :param measured_errors:
        :param number_of_samples:
        :param fbin:
        :return:
        """
        if type(measured_errors) != float:
            if sample_size != len(measured_errors):
                print "Changing the subsample size from %i to %i to match the measured errors!" % \
                      (sample_size, len(measured_errors))

        self.fbin = fbin
        self.calculate_radial_velocity()

        sigma_1d = np.std(np.random.choice(self.radial_velocities, (number_of_samples, sample_size)) +
                        np.random.normal(0, measured_errors, (number_of_samples, sample_size)), axis=1, ddof=1)
        self.sigma_1d = sigma_1d
        return sigma_1d

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


def plot_sigma_1d(sigma1d, labels, colors=None):
    """
    Plots the distribution of
    :param sigma1d:
    :param labels:
    :return:
    """
    if type(colors) == type(None):
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        if len(sigma1d) > 10:
            print "Please specify your own colors if more than 10 distributions are to be plotted"
            return

    n_distributions = len(sigma1d)
    if len(labels) != n_distributions:
        print "Warning! Number of labels does not match number of distributions"
        return

    fig, ax = plt.subplots(1,1, figsize=(6,5))

    for i, dist in enumerate(sigma1d):
        ax.hist(dist, histtype="step", bins=np.linspace(0, 100, 200), label=labels[i], density=True, color=colors[i], lw=1.5)
        if len(sigma1d) < 10:
            ax.axvline(np.median(dist), color=colors[i])

    ax.set_xlabel(r"$\sigma_{\rm 1D}$ [km s$^{-1}$]")
    ax.set_ylabel(r"Frequency [(km s$^{-1}$)$^{-1}$]")

    if len(labels) < 10:
        ax.legend()
    ax.set_xlim([0, 50])
    ax.set_ylim([0, 0.1])
    plt.tight_layout()
    # plt.show()
    return fig, ax


if __name__ == "__main__":

    # initialize the generation of the parameters
    rv = RVMC(fbin=0.7)

    # Check if the distributions are as desired
    rv.check_initialization()

    # Change a parameter
    rv.eccentricity[:] = 0
    rv.eccentric_anomaly = rv.calculate_eccentric_anomaly()

    # Check again
    rv.check_initialization()

    # Calculate the velocities of the binaries once parameter distributions are as desired
    rv.calculate_binary_velocities()

    # Generate many subsamples
    rv.subsample()

    fig, ax = plot_sigma_1d([rv.sigma_1d], ["fbin=0.7"])
    ax.set_xlim([0, 50])
    # ax.set_ylim([0, 0.1])
    plt.show()

    # Use the numerically calculated orbits file to check the results
    rv_num = RVMC(numerical_orbits=True, fbin=0.7)
    rv_num.calculate_binary_velocities()

    RV_errors = np.array([0.8, 2.3, 1.0, 1.8, 0.8, 1.0, 1.4, 2.5, 2.0, 0.4, 1.5, 0.6])
    # Generate subsamples, note that the size of the sub-samples can be varied as well as the binary fraction
    rv_num.subsample(sample_size=12, fbin=0.7, measured_errors=RV_errors)
    rv.subsample(sample_size=12, fbin=0.7, measured_errors=RV_errors)
    data_sana = np.loadtxt("data_sana/fbin_0.70.dat").T

    fig, ax = plot_sigma_1d([rv.sigma_1d, rv_num.sigma_1d, data_sana], ["Standard", "Numerical orbits", "Sana (2017)"])
    ax.set_xlim([0, 50])
    # ax.set_ylim([0, 0.1])
    plt.show()

    # Example of generating many sigma1d distributions for many binary fractions.
    import matplotlib.cm as cmx
    from matplotlib.colors import Normalize
    from tqdm import tqdm

    print "Generating sample"
    rv = RVMC()
    rv.calculate_binary_velocities()

    print "Generating subsamples"
    fbins = np.linspace(0, 1, 100)
    dists = [rv.subsample(fbin=fbin) for fbin in tqdm(fbins)]
    labels = [r"$f_{\rm bin}$ = %.3g" % fbin for fbin in fbins]

    norm = Normalize(0, 1)
    scalarmap = cmx.ScalarMappable(norm=norm, cmap="viridis")
    scalarmap._A = []
    colors = [scalarmap.to_rgba(fbin) for fbin in fbins]

    print "Plotting"
    plot_sigma_1d(dists, labels, colors=colors)
    plt.show()


