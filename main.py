import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

from linear_velocity import LinearVelocity
from velocity_field import VelocityField
from linearmoc import LinearMoc
from accretion import calculate_accretion_efficiency
import surface_density.surface_density as sd
import torque.torque as tq
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
    'soft' : 0.04,                         # Potential softening (0.007)
    'q' : 3e-05,                           # Planet/star mass ratio
    'taus': 1.0,                           # Stokes number
    'eta': 0.001875,                       # Radial pressure gradient parameter
    'zero_velocity_background' : False     # Take bkg radial velocity into perturbation?
}

##print('Pablo planet masses: ', np.logspace(-6,np.log10(3.0e-5),10))
#print('Pablo Stokes numbers: ', np.logspace(-2,1,20))
#exit()

########################
# ACCRETION EFFICIENCY #
########################

# Create velocity field from linear equations
#lin_vel = LinearVelocity(param_dict)
#vel_field = VelocityField.from_linear(lin_vel, 4000, approx=False)
# Could also create from FARGO3D snapshot
#vel_field = VelocityField.from_fargo('/Users/sjp/Downloads/Archive/static/fargo3d/outputs/lui_ormel/', 20)
# Method of Characteristics object
#moc = LinearMoc.from_single(lambda x,y: moc_abc_interpolate(x, y, vel_field, min_dist=0.0))

# Calculate accretion efficiency from streamlines
#eff, r, phi = calculate_accretion_efficiency(moc, param_dict, from_streamlines=True, N=512, r_start=1.4)
#print('Efficiency: ', eff)

# Data collected
#taus =            [0.0100, 0.0110, 0.0120, 0.0150, 0.0200, 0.0300, 0.0500, 0.1000, 0.2000, 0.5000, 1.0000]
#acc_eff_soft7_3 = [0.3203, 0.3320, 0.3164, 0.3086, 0.2969, 0.2813, 0.2734, 0.2148, 0.1875, 0.1484, 0.1250]
#acc_eff_soft1_2 = [0.1719, 0.1953, 0.1836, 0.1914, 0.2070, 0.2148, 0.2305, 0.2148, 0.1758, 0.1445, 0.1172]
#acc_eff_soft5_3 = [0.4023, 0.4023, 0.3867, 0.3711, 0.3438, 0.3125, 0.2930, 0.2461, 0.1953, 0.1523, 0.1250]
#acc_liu =         [0.5529, 0.5355, 0.5202, 0.4828, 0.4385, 0.3830, 0.3230, 0.2564, 0.2035, 0.1499, 0.1190]

def mass_flux(param_dict, stokes_range):
    stokes_range = np.asarray(stokes_range)
    scalar_input = False
    if stokes_range.ndim == 0:
        stokes_range = stokes_range[None]  # Makes x 1D
        scalar_input = True

    # Accretion efficiency according to Ormel & Liu
    acc_eff = lambda x: epsilon(mode='2dset',qp=param_dict['q'], tau=x, eta=param_dict['eta'])

    ret = 0*stokes_range
    for i in range(0, len(ret)):
        param_dict['taus'] = stokes_range[i]
        # Equilibrium radial velocity pebbles
        u0 = equilibrium_velocities(param_dict)[0]
        # Accretion rate in units of Sigma_p*r_p^2*Omega_p
        ret[i] = 2*np.pi*u0*acc_eff(stokes_range[i])

    # Convert to Earth masses per year: surface density is 1.53e-3 for M_disc = 0.01 M_sun
    dust_to_gas_ratio = 0.01
    Sigmap = dust_to_gas_ratio*1.0e-4 #1.53e-3

    # Omega_p = 2*pi/yr to make it Earth masses per year
    ret = ret*Sigmap*2.0*np.pi/3.0e-6

    if scalar_input:
        return np.squeeze(ret)

    return ret

def total_mass_flux(param_dict, stokes_min, stokes_max):
    a = stokes_min
    b = stokes_max
    return sp.integrate.quad(lambda x: mass_flux(param_dict, x)*0.5/np.sqrt(x)/(np.sqrt(b) - np.sqrt(a)), a, b)[0]

planet_mass = np.linspace(1, 30, 100)
acc = 0*planet_mass
for i in range(0, len(acc)):
    param_dict['q'] = planet_mass[i]*3.0e-6
    acc[i] = total_mass_flux(param_dict, 0.01, 1.0)

plt.plot(planet_mass, -acc)

acc_lyra = 3*(1.0/0.1)**(2/3)*2*np.pi*(planet_mass*1.0e-6)**(2/3)*0.01*1.0e-4/(14-3*3.5)/3.0e-6
plt.plot(planet_mass, acc_lyra)

#plt.xscale('log')
plt.yscale('log')
#plt.xlim([0.006,2])
plt.ylim([0.00003,0.02])

#plt.plot(taus, acc_eff_soft5_3, label=r'$\epsilon=0.005$')
#plt.plot(taus, acc_eff_soft7_3, label=r'$\epsilon=0.007$')
#plt.plot(taus, acc_eff_soft1_2, label=r'$\epsilon=0.01$')
#plt.plot(taus, acc_liu, label='Liu+Ormel')

#plt.title('Accretion efficiency and portential softening')
#plt.xlabel('St')
#plt.ylabel('efficiency')

#plt.legend()

plt.show()
exit()

###################
# STREAMLINE PLOT #
###################

# Streamline plot, based on MOC surface density calculation.

#print('Hill sphere: ', (param_dict['q']/3)**(1/3))

# Streamlines are calculated backwards from r=0.8. Total number: 64, of which a fraction is started
# at a circle of radius 0.01 around the planet (accreting streamlines).
sigma = sd.SurfaceDensity.from_moc(param_dict, moc, r_start=0.8, N=256, r_circle=0.02, dt=0.02*np.pi)

# Plot the streamlines in the current axes
streamline_plot(sigma)(plt.gca(), stride=1)

#plt.xlim([0.975,1.025])
#plt.ylim([-0.025,0.025])

plt.xlabel(r'$r$')
plt.ylabel(r'$\varphi$')
plt.title('Static gas')

plt.show()
exit()

########################
# SURFACE DENSITY PLOT #
########################

#fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

#param_dict['zero_velocity_background'] = True
#lin_vel = LinearVelocity(param_dict)
#vel_field = VelocityField.from_linear(lin_vel, 4000, approx=False)
#moc = LinearMoc.from_single(lambda x,y: moc_abc_interpolate(x, y, vel_field, min_dist=0.0))
#sigma = sd.SurfaceDensity.from_moc(param_dict, moc, r_start=0.8, N=2048, r_circle=-0.01, dt=0.02*np.pi)
#surface_density_plot(sigma)(plt.gca())

#msingle = 6
param_dict['zero_velocity_background'] = False
lin_vel = LinearVelocity(param_dict)
#r = np.linspace(0.8, 1.2, 2049)
#phi = np.linspace(-np.pi, np.pi, 1000, endpoint=False)
#sigma = sd.SurfaceDensity.from_fourier(param_dict, r, phi, lin_vel, mrange=[0,100])
#surface_density_plot(sigma)(plt.gca())


stokes = np.logspace(-2,1,30)
sigma_c = np.zeros_like(stokes)

for i in range(0, len(stokes)):
    param_dict['taus'] = stokes[i]
    lin_vel = LinearVelocity(param_dict)
    rc = np.power(lin_vel.Omega0, 2/3)
    r = np.asarray([rc,1.2])
    phi = np.asarray([0.0])
    #sigma = sd.SurfaceDensity.from_fourier(param_dict, r, phi, lin_vel, mrange=[0,400], method='approx')
    #sigma_c[i] = np.abs(sigma.Sigma[0,0]-1.0)

print(stokes, sigma_c)
plt.plot(stokes, sigma_c, marker='o')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('St')
plt.ylabel(r'$\Sigma_c$')
plt.show()

#plt.show()
#exit()

###############
# TORQUE PLOT #
###############

from torque.fourier import total_torque_sum


#stokes_plot = np.linspace(3.0, 20.0, 2)
#torque = np.zeros_like(stokes_plot)
#for i in range(0, len(torque)):
#    param_dict['taus'] = stokes_plot[i]
#    lin_vel = LinearVelocity(param_dict)
#    torque[i] = total_torque_sum(param_dict, lin_vel, mrange=[1,100])
#    print(i, stokes_plot[i], torque[i])

#plt.plot(stokes_plot, torque)
#plt.xlabel('St')
#plt.ylabel(r'$\Gamma$')
#plt.show()

# soft = 0.04, eta=0.001875
stokes = [0.01, 68.19435108524077,
          0.05, 256.0940238432398,
          0.075, 309.72977467089646,
          0.1, 325.6291438024326,
          0.12, 318.7214205227355,
          0.16, 257.0703231048056,
          0.2, 154.0564736995673,
          0.325, -273.08415669181574,
          0.55, -984.1903003736019,
          0.775, -1314.4518056946458,
          0.9, -1377.102405438828,
          1.0, -1373.8862912833038,
          1.1, -1348.9110958742126,
          1.2, -1309.322613334745,
          1.5, -1066.1091111687615,
          2.0, -715.7586092854831]
plt.plot(stokes[0::2], stokes[1::2])
plt.xscale('log')
plt.xlabel('St')
plt.ylabel(r'$\Gamma$')
plt.show()


stokes = [0.01,        0.01268961,  0.01610262,  0.0204336,   0.02592944,  0.03290345,
          0.04175319,  0.05298317,  0.06723358,  0.08531679,  0.10826367,  0.13738238,
          0.17433288,  0.22122163,  0.28072162,  0.35622479,  0.45203537,  0.57361525,
          0.72789538,  0.92367086,  1.1721023,   1.48735211,  1.88739182,  2.39502662,
          3.03919538,  3.85662042,  4.89390092,  6.21016942,  7.88046282,  10.        ]
sigma_approx = [0.41941178,  0.47868111,  0.54761356,  0.62851387,  0.72471797,  0.84127204,
                0.98605952,  1.1715871,   1.41762615,  1.75477769,  2.22879302,  2.9052549,
                3.87395267,  5.25115511,  7.17415933,  9.77374769, 13.09719933, 16.95498717,
                20.73675073, 23.4264737,  24.0904033,  22.57170885, 19.6002763,  16.19050473,
                13.0645178,  10.52554955,  8.59806671,  7.18521942,  6.16277292,  5.42024576]
#plt.plot(stokes, sigma_c, label='numerical')
plt.plot(stokes, sigma_approx, label='approx')
plt.plot(stokes, 0.65*np.sqrt(np.asarray(stokes)/stokes[0]), label=r'$\propto {\rm St}^{1/2}$')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('St')
plt.ylabel(r'$|\Sigma\'(r_c)|$')
plt.title('Planet mass $q=1\cdot 10^{-6}$, Softening $b/r_0=r_h=0.007$')
plt.legend()
plt.show()
