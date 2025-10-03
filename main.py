import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

from linear_velocity import LinearVelocity
from velocity_field import VelocityField
from linearmoc import LinearMoc
from accretion import calculate_accretion_efficiency
import surface_density as sd
import torque as tq
from plot import streamline_plot, surface_density_plot

from OL18 import epsilon

def moc_abc(r, phi, maxm, vel_field):
    '''Brute force calculation of the velocities and source term for the Method Of Lines computation'''
    u, v, dudr = vel_field.calc_2D_sum(r, phi, maxm, grid_output=False, full_output=True)

    return np.real(u), np.real(v/r) + r**(-1.5) - 1.0, np.real(dudr)

def moc_abc_interpolate(r, phi, vel_field, min_dist=0.0):
    '''Interpolating velocities and source term for the Method of Lines computation'''

    # Ignore a circle around the planet (needed if pebble accretion)
    fac = np.ones(len(r))
    dist_to_planet = np.sqrt(r*r + 1 - 2*r*np.cos(phi))
    fac[np.asarray(dist_to_planet < min_dist).nonzero()] = 0

    # For convenience, remap phi to [-pi, pi]
    phi = (phi + np.pi) % (2*np.pi) - np.pi
    points = np.array((r, phi)).T

    # Return vr, Omega, source = -dvrdr - dOmega/dphi
    return fac*np.real(vel_field.u(points)), fac*(np.real(vel_field.v(points)/r) + np.abs(r)**(-1.5) - 1.0), fac*np.real(vel_field.src(points))

def equilibrium_velocities(param_dict):
    '''Calculate equilibrium (i.e. no planet) radial velocity and angular momentum.
       Note that these are in fact power laws: u = u0*(r/r0)**(-0.5) and L = L0*(r/r0)**(0.5).'''
    if (param_dict['zero_velocity_background'] is True):
        return 0.0, 1.0

    func = lambda x: -0.5*x*x - (1-param_dict['eta'])**2/(1 + 0.5*param_dict['taus']*x)**2 + 1 + x/param_dict['taus']
    u0 = sp.optimize.fsolve(func, 0.0)[0]
    L0 = (1-param_dict['eta'])/(1 + 0.5*u0*param_dict['taus'])

    return u0, L0

# Dictionary with parameters
param_dict = {
    'soft' : 0.005,                       # Potential softening (0.007)
    'q' : 3e-5,                           # Planet/star mass ratio
    'taus': 0.1,                          # Stokes number
    'eta': 0.001875,                      # Radial pressure gradient parameter
    'zero_velocity_background' : True     # Take bkg radial velocity into perturbation?
}

########################
# ACCRETION EFFICIENCY #
########################

# Create velocity field from linear equations
lin_vel = LinearVelocity(param_dict)
vel_field = VelocityField.from_linear(lin_vel, 4000, approx=False)
# Could also create from FARGO3D snapshot
#vel_field = VelocityField.from_fargo('/Users/sjp/Downloads/snapshot_10M_e_St_01/', 131)

#r_con = 1.5
#ret = sp.integrate.quad(lambda x: vel_field.u([r_con, x]), 0, 2*np.pi)
#print('eta = ', 0.25*ret[0]*np.sqrt(r_con)/np.pi/0.01)

#exit()

# Method of Characteristics object
moc = LinearMoc.from_single(lambda x,y: moc_abc_interpolate(x, y, vel_field, min_dist=0.0))

# Calculate accretion efficiency from streamlines
#print(calculate_accretion_efficiency(moc, param_dict, from_streamlines=True))
#print(epsilon(mode='2dset', times_eta=False, qp=param_dict['q'], tau=param_dict['taus'], eta=param_dict['eta']))
#exit()

# Data collected
#taus =            [0.0100, 0.0110, 0.0120, 0.0150, 0.0200, 0.0300, 0.0500, 0.1000, 0.2000, 0.5000, 1.0000]
#acc_eff_soft7_3 = [0.3203, 0.3320, 0.3164, 0.3086, 0.2969, 0.2813, 0.2734, 0.2148, 0.1875, 0.1484, 0.1250]
#acc_eff_soft1_2 = [0.1719, 0.1953, 0.1836, 0.1914, 0.2070, 0.2148, 0.2305, 0.2148, 0.1758, 0.1445, 0.1172]
#acc_eff_soft5_3 = [0.4023, 0.4023, 0.3867, 0.3711, 0.3438, 0.3125, 0.2930, 0.2461, 0.1953, 0.1523, 0.1250]
#acc_liu =         [0.5529, 0.5355, 0.5202, 0.4828, 0.4385, 0.3830, 0.3230, 0.2564, 0.2035, 0.1499, 0.1190]

#plt.xscale('log')
#plt.yscale('log')
#plt.xlim([0.006,2])
#plt.ylim([0.01,1])

#plt.plot(taus, acc_eff_soft5_3, label=r'$\epsilon=0.005$')
#plt.plot(taus, acc_eff_soft7_3, label=r'$\epsilon=0.007$')
#plt.plot(taus, acc_eff_soft1_2, label=r'$\epsilon=0.01$')
#plt.plot(taus, acc_liu, label='Liu+Ormel')

#plt.title('Accretion efficiency and portential softening')
#plt.xlabel('St')
#plt.ylabel('efficiency')

#plt.legend()

#plt.show()

###################
# STREAMLINE PLOT #
###################

# Streamline plot, based on MOC surface density calculation.

print('Hill sphere: ', (param_dict['q']/3)**(1/3))

# Streamlines are calculated backwards from r=0.8. Total number: 64, of which a fraction is started
# at a circle of radius 0.01 around the planet (accreting streamlines).
sigma = sd.SurfaceDensity.from_moc(param_dict, moc, r_start=0.8, N=64, r_circle=0.01, dt=0.2*np.pi)

# Plot the streamlines in the current axes
streamline_plot(sigma)(plt.gca(), stride=1)

plt.xlim([0.8,1.2])
plt.ylim([-0.1,0.1])

plt.xlabel(r'$r$')
plt.ylabel(r'$\varphi$')

plt.show()
