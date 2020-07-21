import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

G = 6.67408e-11
one_sixth = 1./6.


def a_grav(bodies, masses):
    """
    Calculates change in velocity and position of two gravitional bodies
    :param bodies: Bodies is an np array of n body pairs in the shape (N, 2, 2, 2, 2) (pairs, x-coords, y-coords, x-vels, y-vels)
    :param masses: shape (N, 2) array of masses of the bodies. Note: arranged as (N pairs, M_1, mass ratio)
    :return:
    """
    dbodies = np.zeros(bodies.shape)

    dx = bodies[:,0, 0] - bodies[:, 0, 1]
    dy = bodies[:,1, 0] - bodies[:, 1, 1]

    Gr3 = G * masses[:, 0] * (dx**2 + dy**2)**-1.5
    # r3 = r2**-1.5

    #Gr3 = G * masses[:, 0] * r3

    # Move primary
    dbodies[:, 0, 0] = bodies[:, 2, 0]# * time_steps
    dbodies[:, 1, 0] = bodies[:, 3, 0]# * time_steps
    # Move secondary
    dbodies[:, 0, 1] = bodies[:, 2, 1]# * time_steps
    dbodies[:, 1, 1] = bodies[:, 3, 1]# * time_steps

    # Accelerate primary
    dbodies[:, 2, 0] = -dx * Gr3 * masses[:, 1]  # * time_steps  # note this is multiplied by the mass ratio
    dbodies[:, 3, 0] = -dy * Gr3 * masses[:, 1]  # * time_steps  # note this is multiplied by the mass ratio
    # Accelerate secondary
    dbodies[:, 2, 1] = dx * Gr3  # * time_steps
    dbodies[:, 3, 1] = dy * Gr3  # * time_steps

    return dbodies


def rk4(function, x, masses, dt):
    """
    Applies a fourth order runge-kutta numerical integration method to a
    variable x with differtation function (?) "function" with a time step of dt
    Returns the new value for "x"
    """
    k1 = function(x, masses)
    k2 = function(x + k1 * dt * 0.5, masses)
    k3 = function(x + k2 * dt * 0.5, masses)
    k4 = function(x + k3 * dt, masses)
    x += one_sixth * (k1 + 2 * k2 + 2 * k3 + k4) * dt
    return x


def semi_major_axis(period, primary_mass, mass_ratio):
    """
    Calculates the semi major axis of the binary (that is the total semi major axis).
    All input in SI units
    :param period:
    :param primary_mass:
    :param mass_ratio:
    :return: The semi major axis
    """
    return ((4 * np.pi ** 2) / (G * (1 + mass_ratio) * primary_mass * period ** 2))**-(1./3.)


def initialize_parameters(number_of_stars=1000, min_mass=6, max_mass=20, min_period=10.**0.15, max_period=10.**3.5):
    """
    Create the random samples following their respective distributions.
    :param number_of_stars:     (int) number of stars (duh)
    :param min_mass:            (scalar) minimum mass in solar masses
    :param max_mass:            (scalar) maximum mass in solar masses
    :param min_period:          (scalar) minimum period in days
    :param max_period:          (scalar) maximum period in days.
    :return:    Note returns the masses in kg and period in seconds.
    """
    a = 2.3

    # CDF sampling of the IMF powerlaw
    Pmin = (min_mass / min_mass) ** (-a + 1)
    Pmax = (max_mass / min_mass) ** (-a + 1)
    P = np.random.uniform(Pmax, Pmin, size=number_of_stars)
    primary_mass = P ** (1. / (-a + 1)) * min_mass * 2e30

    period = 10**(np.random.uniform(np.log10(min_period)**0.5, np.log10(max_period)**0.5, number_of_stars)**2) * 24 * 3600
    mass_ratio = np.random.uniform(0.1, 1., number_of_stars)
    # mass_ratio = np.zeros(number_of_stars) + 0.01

    eccentricity = np.zeros(number_of_stars)
    index6d = (period > (4 * 24 * 3600)) & (period < (6 * 24 * 3600))
    index_rest = period > (6 * 24 * 3600)
    eccentricity[index6d] = np.random.uniform(0**0.5, 0.5**0.5, np.sum(index6d))**2
    eccentricity[index_rest] = (np.random.uniform(0**0.5, 0.9**0.5, np.sum(index_rest))**2)

    semi_major_axes = semi_major_axis(period, primary_mass, mass_ratio)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(6,6))

    ax1.hist(primary_mass / 2e30, cumulative=True, bins=np.geomspace(min_mass, max_mass, 50))
    ax2.hist(period / (3600 * 24), cumulative=True, bins=np.geomspace(min_period, max_period, 50))
    ax3.hist(mass_ratio, cumulative=True, bins=50)
    ax4.hist(eccentricity, cumulative=True, bins=50)

    ax1.set_xscale("log")
    ax2.set_xscale("log")

    ax1.set_title("Primary mass")
    ax2.set_title("Period")
    ax3.set_title("Mass ratio")
    ax4.set_title("Eccentricity")
    plt.tight_layout()
    plt.show()

    return primary_mass, period, mass_ratio, eccentricity, semi_major_axes




def initialize_bodies(Npairs, Nsteps):

    primary_mass, period, mass_ratio, eccentricity, semi_major_axes = initialize_parameters(number_of_stars=Npairs)
    # primary_mass, period, mass_ratio, eccentricity, semi_major_a = np.array([[1*2e30], [365*3600*24], [1.], [0.0], [semi_major_axis(365*3600*24,  1*2e30, 0.0000005)]])


    np.save("binary_properties.npy", [primary_mass, period, mass_ratio, eccentricity, semi_major_axes])
    #
    # print "Primary mass: \t %.3g Msun" % (primary_mass / 2e30)
    # print "Period: \t %.3g days" % (period / (3600 * 24))
    # print "Mass ratio: \t %.3g" % mass_ratio
    # print "Eccentricity: \t %.3g" % eccentricity
    # print "Semi ma ax: \t %.3g m" % semi_major_a
    q = mass_ratio
    e = eccentricity

    # initialize the bodies at the periastron
    # Secondary
    bodies = np.zeros((Npairs, 4, 2))
    bodies[:, 0, 1] = ((1. - e) / (1 + q)) * semi_major_axes
    bodies[:, 3, 1] = (1. / (1 + q) * ((1 + e) / (1 - e))**0.5 *
                      ((G * (primary_mass * (1 + q))) / semi_major_axes) ** 0.5)

    # Primary
    bodies[:, 0, 0] = -q * bodies[:, 0, 1]
    bodies[:, 3, 0] = -q * bodies[:, 3, 1]

    masses = np.array(zip(primary_mass, mass_ratio))
    time_steps = period / Nsteps

    print bodies

    return bodies, masses, time_steps


def run(Npairs, Nsteps, save_interval=10, test=False):
    """
    Runs a set of N gravitational body pairs. Note this could possible be optimized further by only simulating half an
    orbit and mirroring it such that a complete orbit is covered.
    :param Npairs:          The number of binary pairs
    :param Nsteps:          The number of steps to simulate (affects the accuracy and speed of the simulation)
    :param save_interval:   The interval (number of steps) at which to save the velocities (affects the size of the saved file.)
    :return:
    """

    bodies, masses, timesteps = initialize_bodies(Npairs, Nsteps)
    if test:
        all_steps = np.zeros((Nsteps / save_interval, Npairs, 5))
    else:
        all_steps = np.zeros((Nsteps / save_interval, Npairs, 2))

    # timesteps *= 10
    repeated_timesteps = timesteps.repeat(8).reshape(Npairs, 4, 2)

    for i in tqdm(range(Nsteps)):

        # if (i + 1) % (Nsteps / 10) == 0:
        #     print "progress %i%%" % int((float(i + 1) / Nsteps) * 100)

        if i % save_interval == 0:

            # The test case
            if test:
                all_steps[i / save_interval,:,0] = bodies[:, 0, 0]
                all_steps[i / save_interval,:,1] = bodies[:, 1, 0]
                all_steps[i / save_interval,:,2] = bodies[:, 2, 0]
                all_steps[i / save_interval,:,3] = bodies[:, 3, 0]
                all_steps[i / save_interval,:,4] = timesteps * i

            # the normal case
            else:
                all_steps[i / save_interval,:,0] = bodies[:, 2, 0]
                all_steps[i / save_interval,:,1] = bodies[:, 3, 0]

        bodies = rk4(a_grav, bodies, masses, repeated_timesteps)

    np.save("radial_velocities.npy", all_steps)


if __name__ == "__main__":
    run(10**5, 10000)




























