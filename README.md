# RVMC

This is a very incomplete readme file of the RVMC scripts included here.
The purpose of this script is generating radial velocities of clusters or groups of stars. This is done based on input distributions of orbital parameters. The fault distribution is mainly based on Sana et al. (2012). A very basic example of usage can be seen in the RVMC_example.py script, and at the bottom of the RVMC.py script. 

There is now also a RVMC_bias_corr.py script. This one allows multi epoch observations of the same system with specified time intervals. It can be useful to check the probability of detecting significant radial velocity shifts given the expected properties of a binary system. Usage example is for now just in the docstring in the beginning of the class. Need to still improve on this. 