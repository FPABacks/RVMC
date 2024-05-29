from RVMC import RVMC, plot_sigma_1d
import matplotlib.pyplot as plt
import numpy as np

# initialize the cluseter
cluster = RVMC(number_of_stars=10**5)  # note that there are a few optional parameters that can be used

# plot the distributions
cluster.check_initialization()

# Note that the distributions can be changed if you wish:
# cluster.time = np.zeros(10**5)  # Start all stars at the moment of maximum velocity
# cluster.eccentric_anomaly = cluster.calculate_eccentric_anomaly()
# cluster.check_initialization()

# Calculate the velocities of the binaries once parameter distributions are as desired
cluster.calculate_binary_velocities()

# Can check the distribution of radial velocities:
cluster.calculate_radial_velocity()
plt.hist(cluster.radial_velocities, bins=50)
plt.xlabel("Radial velocity")
plt.ylabel("Number of stars")
plt.show()

# Generate many subsamples that simulate the observed cluster
cluster.subsample(sample_size=10, number_of_samples=10**6)
RV_dispersion_10 = cluster.sigma_1d

# Do it again with maybe some different parameters
cluster.subsample(sample_size=30, number_of_samples=10**6)
RV_dispersion_30 = cluster.sigma_1d

cluster.subsample(sample_size=50, number_of_samples=10**6)
RV_dispersion_50 = cluster.sigma_1d

cluster.subsample(sample_size=100, number_of_samples=10**6)
RV_dispersion_100 = cluster.sigma_1d

fig, ax = plot_sigma_1d([RV_dispersion_10, RV_dispersion_30, RV_dispersion_50, RV_dispersion_100],
                        ["10 stars", "30 stars", "50 stars", "100 stars"])
ax.set_xlim([0, 75])
plt.show()
