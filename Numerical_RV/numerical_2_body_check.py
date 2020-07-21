import numpy as np
import matplotlib.pyplot as plt
# from main_RVMC import *

# run(1000, 1000)

def check_test_output_velocities():
    """
    Checks if the output from the orbit simulations are reasonable
    :return: 
    """
    data = np.load("test_sim.npy")
    bin_props = np.load("binary_properties.npy")

    selection = bin_props[0,:] > 15 * 2e30
    selection *= bin_props[1,:] < 10 * 3600 * 24
    # selection *= bin_props[2,:] > 0.5
    selection *= bin_props[3,:] < 0.05

    print np.sum(selection)

    indices = np.argwhere(selection)

    for i in indices:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6,3))
        ax1.plot(data[:,i,0], data[:,i,1])
        ax2.plot(data[:,i,4] / (3600 * 24),  data[:,i,2] * 0.001, label="vx")
        ax2.plot(data[:,i,4] / (3600 * 24),  data[:,i,3] * 0.001, label="vy")
        for j in range(10):
            ax2.axvline(j * 2, color="k", alpha=0.5)
        plt.show()


    # selection = bin_props[0,:] > 15. * 2e30
    selection = bin_props[1,:] > 1000 * 3600 * 24
    # selection *= bin_props[2,:] > 0.5
    selection *= bin_props[3,:] > 0.5

    print np.sum(selection)

    indices = np.argwhere(selection)

    for i in indices:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6,3))
        ax1.plot(data[:,i,0], data[:,i,1])
        ax2.plot(data[:,i,4] / (3600 * 24),  data[:,i,2] * 0.001, label="vx")
        ax2.plot(data[:,i,4] / (3600 * 24),  data[:,i,3] * 0.001, label="vy")
        plt.show()




    print np.shape(data)

    max_vels_x = np.max(data[:,:,0], axis=0)
    min_vels_x = np.min(data[:,:,0], axis=0)

    print max_vels_x.shape

    abs_diffs_x = np.abs(min_vels_x) - max_vels_x

    max_vels_y = np.max(data[:,:, 1], axis=0)
    min_vels_y = np.min(data[:,:, 1], axis=0)

    print max_vels_y.shape

    abs_diffs_y = np.abs(min_vels_y) - max_vels_y

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(3, 6))

    ax1.hist(abs_diffs_x, histtype="step", bins=40, label="x")
    ax2.hist(abs_diffs_y, histtype="step", bins=40, label="y")
    ax1.legend()
    ax2.legend()
    plt.show()

    print abs_diffs_x.max(), abs_diffs_y.max()

    props = bin_props[:, np.argmax(abs_diffs_y)]

    # [primary_mass, period, mass_ratio, eccentricity, semi_major_a])
    print props[0] / 2e30
    print props[1] / (3600 * 24)
    print props[2]
    print props[3]

    print "travelled distance"
    print "x", np.trapz(data[:, np.argmax(abs_diffs_y),0], dx=props[1] / 1000.)
    print "y", np.trapz(data[:, np.argmax(abs_diffs_y),1], dx=props[1] / 1000.)

    plt.plot(data[:, np.argmax(abs_diffs_y),0], label='x')
    plt.plot(data[:, np.argmax(abs_diffs_y),1], label="y")
    plt.legend()
    plt.show()


def check_vxvy():

    data = np.load("binary_velocities.npy")
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(6,3))
    for i, dat in enumerate(data[[0, 999]]):
        ax1.hist(dat[:,0], histtype="step", label="%i" %i, bins=100)
        ax2.hist(dat[:,1], histtype="step", label="%i" %i, bins=100)
    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    plt.show()


def animated_vx(data):
    """
    copy pasted animated histogram from matplotlib website
    :return:
    """
    # import numpy as np
    #
    # import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.path as path
    import matplotlib.animation as animation

    # Fixing random state for reproducibility
    # np.random.seed(19680801)

    pre_bins = np.linspace(-300, 300, 100)

    # histogram our data with numpy
    n, bins = np.histogram(data[0,:,0], pre_bins)

    # get the corners of the rectangles for the histogram
    left = np.array(bins[:-1])
    right = np.array(bins[1:])
    bottom = np.zeros(len(left))
    top = bottom + n
    nrects = len(left)

    nverts = nrects * (1 + 3 + 1)
    verts = np.zeros((nverts, 2))
    codes = np.ones(nverts, int) * path.Path.LINETO
    codes[0::5] = path.Path.MOVETO
    codes[4::5] = path.Path.CLOSEPOLY
    verts[0::5, 0] = left
    verts[0::5, 1] = bottom
    verts[1::5, 0] = left
    verts[1::5, 1] = top
    verts[2::5, 0] = right
    verts[2::5, 1] = top
    verts[3::5, 0] = right
    verts[3::5, 1] = bottom

    patch = None

    def animate(i):
        # simulate new data coming in
        # data = np.random.randn(1000)
        n, bins = np.histogram(data[i,:,0], pre_bins)
        top = bottom + n
        verts[1::5, 1] = top
        verts[2::5, 1] = top
        return [patch, ]

    fig, ax = plt.subplots()
    barpath = path.Path(verts, codes)
    patch = patches.PathPatch(barpath, facecolor='green', edgecolor='yellow', alpha=0.5)
    ax.add_patch(patch)

#    ax.set_xlim(left[0], right[-1])
    ax.set_xlim(-300, 300)
    ax.set_ylim(0, 10000)
    # ax.set_ylim(bottom.min(), top.max())

    ani = animation.FuncAnimation(fig, animate, 1000, repeat=False, blit=True, interval=20)
    plt.show()


def v_t(data):

    plt.plot(data[:,:10,0])
    plt.show()


# v_t(np.load("radial_velocities.npy") * 0.001)
# check_vxvy()
# # check_test_output_velocities()
animated_vx(np.load("binary_velocities.npy") * 0.001)

