"""
Script for simulating radial velocity measurements of stars.
By Frank Backs (frank.backs@kuleuven.be)
"""
import numpy as np
from astropy.constants import G
from scipy.optimize import newton
import itertools


def anomaly_function(E, e, M):
    """
    The equation to solve to obtain the eccentric anomaly.
    :param E:   The eccentric anomaly
    :param e:   The eccentricity
    :param M:   The mean anomaly /the time in the orbit times 2 * pi divided by the period
    :return:
    """
    return E - e * np.sin(E) - M


class BiasCorr:
    """
    A tool for determining a bias free distribution of Delta RV values based on
    a set of periods and input parameters.

    Usage example:

------------------------------------------------
    # Input Parameters
    nstars = 10
    nepochs = 5
    nsamples = 5 * 10**4
    epoch_intervals = np.random.uniform(0, np.pi * 10**7, (nstars, nepochs))

    masses = np.random.uniform(10, 50, nstars) * 2e30
    mass_err = masses * 0.1

    # Running
    bc = BiasCorr(epoch_intervals, masses=masses, mass_errors=mass_err, number_of_samples=nsamples)
    bc.calculate_radial_velocities()
    bc.generate_single_delta_rv()
    all_delta_rv = bc.get_all_delta_rv()

    # Checking the result
    plt.plot(np.sort(bc.delta_rv.ravel()), np.linspace(0,1, nstars * nsamples)[::-1], label="Binaries")
    plt.plot(np.sort(bc.delta_rv_single.ravel()), np.linspace(0,1, bc.delta_rv_single.size)[::-1], label="Single stars")
    plt.plot(np.sort(all_delta_rv), np.linspace(0,1, len(all_delta_rv))[::-1], label="All")
    plt.xscale("log")
    plt.xlim([1000, 0.1])
    plt.xlabel("Delta RV [km/s]")
    plt.ylabel("Cumulative distribution")
    plt.legend()
    plt.show()
------------------------------------------------
    """
    def __init__(self,
                 t_obs: (np.ndarray, list),
                 masses: (np.ndarray, list) = None,
                 mass_errors: (np.ndarray, list) = None,
                 number_of_samples: int = 10**4,
                 min_mass: float = 6,
                 max_mass: float = 20,
                 min_period: float = 10**0.15,
                 max_period: float = 10**3.5,
                 period_pi: float = -0.5,
                 mass_ratio_kappa: float = 0,
                 mass_power: float = -2.35,
                 rv_errors: (np.ndarray, list, float) = None,
                 fbin: float = 0.7,
                 period_dist: (np.ndarray, list) = None,
                 mass_dist: (np.ndarray, list) = None,
                 min_q: float = 0.1,
                 max_q: float = 1.0,
                 e_power: float = -0.5):
        """
        Initializes the synthetic population of stars, with their period distribution, mass distribution, mass ratio,
        inclination, orbit rotation, and eccentricity. Also gives the stars a random start time, putting them in a
        random initial position in the orbit.
        Required input is the times since the first observation in seconds (including the first epoch).
        
        :param t_obs:             (required, <array of arrays>)
                                  Time since first observation in shape: (number of stars, number of epochs)
                                  Units are seconds.

        :param masses:            (optional, <array>, default=None)
                                  The (approximate) mass of the objects related to the observing epochs, so
                                  must be same length as number of stars.
                                  Units are kg.

        :param mass_errors:       (optional, <array>, default=None)
                                  Uncertainty on the mass of the stars, same shape as masses.
                                  Units are kg.

        :param number_of_samples: (optional, <int>, default=10**4_
                                  Number of samples to take for each star. If mass or period distributions are specified
                                  this parameter needs to be supplied and equal to the second dimension of the array.

        :param min_mass:          (optional, <float>, default=6)
                                  Lowest mass of the stars to consider if no other masses are specified. An IMF will be
                                  sampled from min_mass to max_mass.
                                  Units are solar masses.

        :param max_mass:          (optional, <float>, default=20)
                                  higest mass of the stars to consider if no other masses are specified. An IMF will be
                                  sampled from min_mass to max_mass.
                                  Units are solar masses.

        :param min_period:        (optional, <float>, default=10**0.15)
                                  Shortest period of binaries to consider.
                                  Units are days.

        :param max_period:        (optional, <float>, default=10**3.5)
                                  Longest period of binaries to consider.
                                  Units are days.

        :param period_pi:         (optional, <float>, default=-0.5)
                                  The power of the power law that is used to sample the period as N(logP) ~ (logP)^pi
                                  Unitless

        :param mass_ratio_kappa:  (optional, <float>, default=0)
                                  Power of the power law that is used to sample the mass ratio.
                                  Unitless

        :param mass_power:        (optional, <float>, default=-2.35)
                                  The power of the powerlaw that is used to sample the masses only a min mass and max
                                  are specified.
                                  Unitless

        :param rv_errors:         (optional, <float or array>, default=None)
                                  The measured uncertainty on the radial velocities. This can be one typical value for
                                  all stars (float), or a separate value for each star (array), or a value for each
                                  epoch (array of arrays). If not None a random velocity drawn from a normal
                                  distribution with a width of rv_error will be added to the simulated radial velocity.
                                  Units are in km/s.

        :param mass_dist:         (optional, <array>, default=None)
                                  A self determined mass distribution to use rather than the mass distribution generated
                                  or estimated here. Shape has to match inputs, so (number of stars, number of samples),
                                  where number of stars is defined by len(t_obs).
                                  Units are kg.

        :param period_dist:       (optional, <array>, default=None)
                                  A self determined period distribution to use rather than the period distribution
                                  generated. Shape has to match inputs, so (number of stars, number of samples), where
                                  number of stars is defined by len(t_obs).
                                  Units are seconds.

        :param min_q:             (optional, <float>, default=0.1)
                                  The low end of the mass ratio distribution.
                                  Unitless

        :param max_q:             (optinal, <float>, default=1.0)
                                  The high end of the mass ratio distribution.
                                  Unitless
        :param e_power:           (optinal, <float>, default=-0.5)
                                  The power of the eccentricity powerlaw distribution
                                  Unitless
        """
        # Just an internal parameter, the star for which the radial velocities are being calculated.
        self.current_star = 0

        self.min_q = min_q
        self.max_q = max_q
        self.kappa = mass_ratio_kappa
        self.pi = period_pi
        self.mass_power = mass_power

        self.number_of_samples = number_of_samples
        self.t_obs = t_obs
        self.number_of_stars = len(t_obs)
        self.e_power = e_power

        # Masses sampling
        if not isinstance(masses, type(None)):
            if len(masses) != self.number_of_stars:
                raise ValueError(f"The number of masses ({len(masses)}) and epoch "
                                 f"sequences ({len(t_obs)}) are not equal!")
            
            # Sample masses normally distributed based on supplied errors
            # Reshaping stuff to be able to do it all in one go for efficiency
            elif not isinstance(mass_errors, type(None)):
                self.primary_mass = np.random.normal(np.reshape(masses, (-1, 1)),
                                                     np.reshape(mass_errors, (-1, 1)),
                                                     size=(self.number_of_stars, number_of_samples))
                self.min_mass = np.min(self.primary_mass) / 2.e30
                self.max_mass = np.max(self.primary_mass) / 2.e30
            # If no uncertainty take masses at face value.
            else:
                self.primary_mass = np.array(masses).reshape(-1, 1)
                self.min_mass = np.min(self.primary_mass) / 2.e30
                self.max_mass = np.max(self.primary_mass) / 2.e30

        # if no masses are supplied generate masses based on the IMF 
        elif not isinstance(mass_dist, type(None)):
            if np.shape(mass_dist) == (len(t_obs), number_of_samples):
                self.min_mass = np.min(mass_dist) / 2.e30
                self.max_mass = np.max(mass_dist) / 2.e30
                self.primary_mass = mass_dist
            else:
                raise ValueError(
                    f"The shape of mass distributions ({np.shape(mass_dist)})"
                    f" not as expected {(len(t_obs), number_of_samples)}.")
        else:
            self.min_mass = min_mass
            self.max_mass = max_mass
            self.primary_mass = self.sample_masses()

        if not isinstance(period_dist, type(None)):

            if np.shape(period_dist) == (len(t_obs), number_of_samples):
                self.min_mass = np.min(period_dist) / (3600 * 24)  # in days for consistency
                self.max_mass = np.max(period_dist) / (3600 * 24)  # in days for consistency
                self.period = period_dist
            else:
                raise ValueError(
                    f"The shape of period distribution ({np.shape(period_dist)})"
                    f" not as expected: {(len(t_obs), number_of_samples)}.")

        else:
            self.min_period = min_period  # Here still in days
            self.max_period = max_period  # Here still in days
            self.period = self.sample_period()  # Seconds now

        if isinstance(rv_errors, type(None)):
            self.rv_errors = None
        else:
            if isinstance(rv_errors, (float, int)):
                self.rv_errors = rv_errors
            elif isinstance(rv_errors, (list, np.ndarray)):
                if len(rv_errors) == self.number_of_stars:
                    self.rv_errors = rv_errors
                else:
                    raise ValueError(f"The number of radial velocity uncertainties ({len(rv_errors)}) does not match"
                                     f"the number of stars ({self.number_of_stars}).")

        self.time = self.sample_time()  # in seconds, used as random offset
        
        self.mass_ratio = self.sample_mass_ratio()
        self.eccentricity = self.sample_eccentricity()

        self.inclination = self.sample_inclination()
        self.orbit_rotation = self.sample_orbit_rotation()

        # Calculated later
        self.semi_major_axis = None
        self.max_orbital_velocity = None

        # Empty list for all the stars
        self.radial_velocities = [[]] * self.number_of_stars
        self.delta_rv = np.zeros(self.time.shape)

        self.fbin = fbin
        self.delta_rv_single = None

    def sample_powerlaw(self, min_val, max_val, power, size=None):
        """
        Samples a powerlaw between the min_val, max_val with power distribution power.
        :return:
        """
        if isinstance(size, type(None)):
            size = (self.number_of_stars, self.number_of_samples)
        pmax = (max_val / min_val) ** (power + 1)
        p = np.random.uniform(pmax, 1, size=size)
        return p**(1. / (power + 1)) * min_val

    def sample_masses(self):
        """
        Returns an array of primary masses sample from a kroupa distribution in kg
        :return:
        """
        primary_mass = self.sample_powerlaw(self.min_mass, self.max_mass, self.mass_power) * 2e30
        return primary_mass
    
    def sample_inclination(self):
        """
        Returns inclinations for each binary as a np array in radians
        :return:
        """
        return np.arccos(np.random.random(size=(self.number_of_stars, self.number_of_samples)))

    def sample_period(self):
        """
        Returns a distribution of periods in seconds
        Distribution described by pdf ~ log(P)**-0.5 with 0.15 < log(P) < 3.5 with P in days
        :return:
        """
        return 24 * 3600 * 10**self.sample_powerlaw(np.log10(self.min_period), np.log10(self.max_period), self.pi)

    def sample_period_old(self):
        return 10 ** (np.random.uniform(np.log10(self.min_period)**0.5,
                                        np.log10(self.max_period)**0.5,
                                        (self.number_of_stars, self.number_of_samples))**2) * 24 * 3600

    def sample_mass_ratio(self):
        """
        Returns an array of mass ratios
        """
        return self.sample_powerlaw(self.min_q, self.max_q, self.kappa)
        # return np.random.uniform(self.min_q, self.max_q, (self.number_of_stars, self.number_of_samples))

    def sample_orbit_rotation(self):
        """
        Returns an array of random orbit rotations in radians
        """
        return np.random.random(size=(self.number_of_stars, self.number_of_samples)) * 2 * np.pi

    def sample_eccentricity(self):
        """
        Returns an array of random eccentricities based on the period of the binaries
        For periods < 4 days e = 0
        for periods between 4 and 6 days e < 0.5
        for periods > 6 days e < 0.9
        """
        eccentricity = np.zeros((self.number_of_stars, self.number_of_samples))
        for i in range(self.number_of_stars):
            more_than_6 = (self.period[i] > (6 * 24 * 3600))
            between_4_and_6 = (self.period[i] > (4 * 24 * 3600)) * np.logical_not(more_than_6)
            eccentricity[i, more_than_6] = self.sample_powerlaw(1e-5, 0.9, self.e_power, size=np.sum(more_than_6))
            eccentricity[i, between_4_and_6] = self.sample_powerlaw(1e-5, 0.5, self.e_power, size=np.sum(between_4_and_6))
        return eccentricity

    def sample_time(self):
        """
        Samples random moments in the orbit of the binaries by sampling a time between 0 and period
        :return:
        """
        return np.random.random(size=(self.number_of_stars, self.number_of_samples)) * self.period

    def calculate_semi_major_axis(self):
        """
        Returns the total semi major axis of all binary systems in meters
        :return:
        """
        self.semi_major_axis = ((4 * np.pi**2) /
                                (G.value * (1 + self.mass_ratio) * self.primary_mass * self.period**2))**-(1. / 3.)

    def calculate_max_orbital_velocity(self):
        """
        Calculates the velocity of the primary in perihelion in km/s
        equation:
        v_0 = q / (1 + q) * ((G * (M + m) * (1 + e)) / ((1 - e) * a))**0.5
        :return: The maximum velocity of the primary. (that is at periastron.
        """
        self.max_orbital_velocity = (self.mass_ratio * ((G.value * self.primary_mass * (1 + self.eccentricity)) /
                                     ((1 + self.mass_ratio) * (1 - self.eccentricity) * self.semi_major_axis)) ** 0.5) * 0.001

    def calculate_eccentric_anomaly(self, time_deltas):
        """
        Calculates the eccentric anomaly. This has to be done by solving keplers equation numerically as there
        is no analytical solution.
        :return:
        """
        return newton(anomaly_function,
                      np.pi * 2 * time_deltas / self.period[self.current_star],
                      args=(self.eccentricity[self.current_star],
                            2 * np.pi * time_deltas % self.period[self.current_star] / self.period[self.current_star]),
                      tol=10**-6)

    def binary_radial_velocity(self, eccentric_anomalies_i):
        """
        Calculates the radial velocity of the primary star of a binary given the eccentricity, inclination, orbit
        rotation (rotation of the semi major axis relative to the observer), eccentric anomaly and the maximal velocity
        of the primary. Note that in the equation here the velocity amplitude is obtained by multiplying the maximal
        orbital velocity by 1./(1 + e) and sin i.
        :return:
        """
        i = self.current_star
        return (1 / (1 + self.eccentricity[i])) * self.max_orbital_velocity[i] * np.sin(self.inclination[i]) *\
               (self.eccentricity[i] * np.cos(self.orbit_rotation[i]) +
                np.cos(2 * np.arctan((((1 + self.eccentricity[i]) / (1 - self.eccentricity[i])) ** 0.5) *
                                     np.tan(0.5 * eccentric_anomalies_i)) + self.orbit_rotation[i]))

    def check_error_shape(self, number_of_epochs):
        """
        Checks the shape of the errors and returns an array or float compatible with the data set for multiplication
        :param number_of_epochs:
        :return:
        """
        if isinstance(self.rv_errors, (np.ndarray, list)):
            if isinstance(self.rv_errors[self.current_star], (np.ndarray, list)):
                if len(self.rv_errors[self.current_star]) == number_of_epochs:
                    use_err = np.reshape(self.rv_errors[self.current_star], (-1, 1))
                else:
                    raise ValueError(f"The number of radial velocity uncertainties "
                                     f"({len(self.rv_errors[self.current_star])}) does not match the number of epochs,"
                                     f"({number_of_epochs})")
            else:
                use_err = self.rv_errors[self.current_star]
        else:
            use_err = self.rv_errors

        return use_err

    def generate_rv_errors(self, number_of_epochs, number_of_samples):
        """
        Simulates measurement errors on the radial velocity for the current star.
        Check first how the errors have been specified and selects based on that what (shape) value is used.
        :param: number_of_epochs    The amount of epochs to generate errors for
        :param: number_of_samples   The number of times an error needs to be generated.
        :return: np.ndarray with shape: (number of epochs, number of samples)
        """
        use_err = self.check_error_shape(number_of_epochs)
        return np.random.normal(0, use_err, size=(number_of_epochs, number_of_samples))

    def calculate_radial_velocities(self):
        """
        Calculates the radial velocities for all specified epochs for all stars
        Can include single stars
        :return: None, but the delta_rv values are updated. The shape of delta_rv is (number of stars, number of samples)
        """
        self.calculate_semi_major_axis()
        self.calculate_max_orbital_velocity()
        for i in range(self.number_of_stars):
            self.current_star = i
            time_deltas = np.reshape(self.t_obs[i], (-1, 1))
            virtual_times = time_deltas + self.time[i]
            eccentric_anomalies_i = self.calculate_eccentric_anomaly(virtual_times)
            self.radial_velocities[i] = self.binary_radial_velocity(eccentric_anomalies_i)
            if not isinstance(self.rv_errors, type(None)):
                self.radial_velocities[i] += self.generate_rv_errors(len(self.t_obs[i]), self.number_of_samples)
            self.delta_rv[i] = self.radial_velocities[i].max(axis=0) - self.radial_velocities[i].min(axis=0)

    def generate_single_delta_rv(self):
        """
        Generates radial velocities for the single stars. These do no have any delta RV from the orbit, but they do have
        measurement uncertainties (if supplied), which can if large enough result in significant delta RV. This can be
        taken into account.
        :return: Nothing, but the self.delta_rv_single is updated.
        """
        needed_singles = int((self.number_of_samples - self.fbin * self.number_of_samples) / self.fbin)
        # print(f"Will need to add {needed_singles} single stars to get the desired binary fraction")
        self.delta_rv_single = np.zeros((self.number_of_stars, needed_singles))
        if not isinstance(self.rv_errors, type(None)):
            for i in range(self.number_of_stars):
                self.current_star = i
                single_rvs = self.generate_rv_errors(len(self.t_obs[i]), len(self.delta_rv_single[i]))
                self.delta_rv_single[i] = single_rvs.max(axis=0) - single_rvs.min(axis=0)
        # If no uncertainties are known just add zeros
        else:
            print("Cannot generate single star radial velocities as there are no uncertainties to work with."
                  "So all are zero. ")

    def get_all_delta_rv(self):
        """
        Combines the samples of single and binary star delta rv values
        :return: combined set of single and binary delta RV values.
        """
        return np.concatenate([self.delta_rv, self.delta_rv_single], axis=1).ravel()

    def get_detected_binaries(self, delta_RV=20, n_sigma_RV=4):
        """
        Checks for each simulated observation set if it is detected as a binary.
        :param delta_RV:     The minimum delta RV required
        :param n_sigma_RV:   The minimum significance of the difference in radial velocities.
        :return: Array of bools with detection (True) or non detection (False) for each star.
        """
        detected = np.zeros((self.number_of_stars, self.number_of_samples), dtype=bool)
        for i, rvs in enumerate(self.radial_velocities):
            # Fix the uncertainties
            self.current_star = i
            err = self.check_error_shape(len(rvs))
            if isinstance(err, (float, int)):
                err = [err] * len(rvs)
            elif isinstance(err, (np.ndarray, list)):
                err = np.ravel(err)

            # Loop over all combinations of radial velocities.
            combinations = itertools.combinations(range(len(rvs)), 2)
            for v1, v2 in combinations:
                detected_req1 = np.abs(rvs[v1,:] - rvs[v2,:]) / np.sqrt(err[v1]**2 + err[v2]**2) > n_sigma_RV
                detected_req2 = np.abs(rvs[v1,:] - rvs[v2,:]) > delta_RV
                detected_req1 *= detected_req2
                detected[i] = np.logical_or(detected[i], detected_req1)

        return detected



