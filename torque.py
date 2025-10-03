import numpy as np
import scipy as sp

from potential import PotentialComponent
from surface_density import calc_surface_density_1D

def nodes_weights(a, b, n=5):
    '''Gauss-Legendre nodes and weights for arbitrary interval'''
    x, w = np.polynomial.legendre.leggauss(n)

    x = 0.5*(b - a)*x + 0.5*(b + a)
    w = 0.5*(b - a)*w

    return x, w

def fixed_quad(func, a, b, n=5):
    '''Gauss-Legendre quadrature but return also the function evaluated at the nodes'''
    x, w = nodes_weights(a, b, n=n)
    res = func(x)
    return np.sum(w*res), x, res

def elementwise_possible(func):
    '''Decorator for functions allowing the first argument to be either scalar or ndarray'''
    def wrapper(*args, **kwargs):
        if np.isscalar(args[0]):
            arg_list = list(args)
            arg_list[0]=np.asarray([arg_list[0]])
            args = tuple(arg_list)
            return func(*args, **kwargs)[0]
        else:
            return func(*args, **kwargs)
    return wrapper

def calc_torque_1D(r, m, u0, param_dict):
    sigma = calc_surface_density_1D(r, m, u0, param_dict)

    # Planet potential component
    Phi = PotentialComponent(m, param_dict['soft'], method='Bessel')
    if m < 150:
        Phi = PotentialComponent(m, param_dict['soft'], method='Hyper')

    pot = 0*r
    for i in range(0, len(r)):
        pot[i] = param_dict['q']*Phi(r[i])

    # indirect term
    if m == 1:
        pot = pot + param_dict['q']*r

    return 2*np.pi*r*np.imag(sigma)*pot

@elementwise_possible
def calc_torque_sum(r, max_m, u0, param_dict):
    '''Calculate the total torque on the disc as a sum over m'''
    torque = 0.0*r

    for m in range(1, max_m):
        torque += calc_torque_1D(r, m, u0, param_dict)

    return torque

def calculate_torque_density(r, Sigma, param_dict, N=None):
    # Sigma(r, phi) should give surface density at (r,phi)
    rr = np.atleast_1d(r)

    # Return 2D torque distribution
    torque_2D = np.zeros((len(rr),N+20))

    r_hill = (param_dict['q']/3)**(1/3)
    r_cut = 0.1*r_hill

    r_acc = (param_dict['taus']*param_dict['q']/3)**(1/3)
    p_bound = 0.5*r_acc*(0.75*r_acc + param_dict['eta'])/param_dict['eta']/param_dict['taus']/0.8
    print('p_bound: ', p_bound, 0.1*np.pi)

    ret = 0.0*rr
    for i in range(0, len(rr)):
        # Mask: 1 if located further from the planet than Hill, 0 otherwise (i.e. Hill cutoff)
        mask_out = lambda pp: np.heaviside(np.sqrt(rr[i]*rr[i] + 1 - 2*rr[i]*np.cos(pp)) - r_cut - 0.5*r_hill, 0.5)
        mask_in = lambda pp: np.heaviside(-np.sqrt(rr[i]*rr[i] + 1 - 2*rr[i]*np.cos(pp)) + r_cut, 0.5)
        mask_mid = lambda pp: 1 - mask_out(pp) - mask_in(pp)
        mask = lambda pp: mask_out(pp) + mask_mid(pp)*np.sin(np.pi*(np.sqrt(rr[i]*rr[i] + 1 - 2*rr[i]*np.cos(pp)) - r_cut)/r_hill)**2

        #mask = lambda pp: np.heaviside(np.sqrt(rr[i]*rr[i] + 1 - 2*rr[i]*np.cos(pp)) - r_hill, 0.5)

        # Potential derivative
        dpot = lambda pp: rr[i]*np.sin(pp)*(rr[i]*rr[i] + 1 - 2*rr[i]*np.cos(pp)+param_dict['soft']**2)**(-1.5)
        # 2D torque distribution
        func = lambda x: Sigma(rr[i], x)*dpot(x)*mask(x)


        #p_bound = np.pi/10

        tq1, p1, f1 = fixed_quad(func, -np.pi, -p_bound, n=10)
        tq2, p2, f2 = fixed_quad(func, -p_bound, p_bound, n=N)
        tq3, p3, f3 = fixed_quad(func, p_bound, np.pi, n=10)
        tq_dens = tq1 + tq2 + tq3

        #import matplotlib.pyplot as plt
        #plt.plot(p1, f1)
        #plt.plot(p2, f2)
        #plt.plot(p3, f3)
        #plt.title('r = {}'.format(r[i]))
        #plt.show()

        #plt.plot(p2, Sigma(rr[i], p2))
        #plt.show()

        phi_ret = np.concatenate([p1,p2,p3], axis=None)
        torque_2D[i,:] = np.concatenate([f1,f2,f3], axis=None)
        ret[i] = tq_dens

        print(i, rr[i], ret[i])

    if np.isscalar(r):
        return ret[0], phi_ret, torque_2D[0,:]
    return ret, phi_ret, torque_2D

def calculate_total_torque(torque_density, param_dict, Nbase=10):
    r_acc = (param_dict['taus']*param_dict['q']/3)**(1/3)

    # Split integration interval into 3
    x0, w0 = nodes_weights(0.9, 1-2*r_acc, n=Nbase)
    x1, w1 = nodes_weights(1-2*r_acc, 1+2*r_acc, n=Nbase)
    x2, w2 = nodes_weights(1+2*r_acc, 1.1, n=Nbase)

    x = np.concatenate([x0,x1,x2], axis=None)
    w = np.concatenate([w0,w1,w2], axis=None)

    tq_dens, phi_ret, torque_2D = torque_density(x)
    return np.sum(w*tq_dens), x, phi_ret, tq_dens, torque_2D

def chrenko_torque(stokes, mass_ratio, h=0.05):
    tau = stokes #np.atleast_1d(stokes)
    q = mass_ratio #np.atleast_1d(mass_ratio)

    xi1_lo = 2.275e-2*tau**(2.645)
    xi2_lo = -0.479 + 0.2*np.log(tau)
    xi1_hi = 9.506e-7*tau**(-12)
    xi2_hi = -1.241 - 0.868*np.log(tau)
    xi1 = np.where(tau < 0.45, xi1_lo, xi1_hi)
    xi2 = np.where(tau < 0.45, xi2_lo, xi2_hi)

    #xi1 = np.minimum(xi1, 1.928e-3*np.ones(len(xi1)))
    #xi2 = np.minimum(xi2, -0.673*np.ones(len(xi2)))
    xi1 = np.minimum(xi1, 1.928e-3)
    xi2 = np.minimum(xi2, -0.673)


    ret = q*q*xi1*q**(xi2)/(0.01*h*h)
    #ret = xi1*q**(xi2)

    #if np.isscalar(stokes):
    #    return ret[0]

    return ret

def guilera_torque(stokes, mass_ratio, h=0.05):
    q = np.asarray([0.333, 0.486, 0.709, 1.03, 1.51, 2.2, 3.22, 4.69, 6.85, 10.0])*3.0e-6
    tau = np.asarray([0.010, 0.014, 0.021, 0.030, 0.043, 0.062, 0.089, 0.127, 0.183, 0.264, 0.379, 0.546, 0.785, 1.129])

    torque = [[0.283, 0.309, 0.264, 0.201, 0.142, 0.093, 0.056, 0.029, 0.006, -0.010],
              [0.371, 0.394, 0.415, 0.333, 0.239, 0.160, 0.098, 0.053, 0.017, -0.008],
	          [0.739, 0.720, 0.618, 0.475, 0.382, 0.276, 0.172, 0.095, 0.041, -0.002],
	          [1.638, 1.332, 1.102, 0.851, 0.625, 0.434, 0.275, 0.185, 0.087, 0.021],
	          [3.254, 2.614, 1.961, 1.476, 1.078, 0.757, 0.503, 0.316, 0.174, 0.085],
	          [5.776, 4.579, 3.428, 2.461, 1.790, 1.266, 0.850, 0.546, 0.326, 0.167],
	          [9.672, 7.297, 5.313, 3.760, 2.620, 1.872, 1.301, 0.880, 0.562, 0.304],
	          [13.684, 10.191, 7.337, 5.145, 3.471, 2.374, 1.725, 1.209, 0.805, 0.483],
	          [16.058, 11.276, 7.866, 5.231, 3.178, 1.372, 1.740, 1.493, 1.073, 0.701],
	          [-1.521, -18.135, -8.446, -0.425, 1.971, 1.746, 2.237, 1.823, 1.351, 0.910],
	          [-32.124,	-7.348, 2.307, 4.517, 4.752, 3.232, 2.803, 2.108, 1.499, 1.016],
	          [5.523, 8.831, 9.054, 7.882, 5.860, 3.741, 2.716, 1.953, 1.398, 0.986],
	          [10.492, 9.803, 7.187, 5.859, 4.314, 3.278, 2.465, 1.821, 1.326, 0.943],
	          [11.861, 9.572, 7.850, 5.807, 4.538, 3.367, 2.463, 1.799, 1.299, 0.916]]
    torque = np.asarray(torque).T

    points = np.array((mass_ratio,stokes)).T
    ret = sp.interpolate.RegularGridInterpolator((q, tau), torque, bounds_error=False, fill_value=None)(points)

    ret = ret*100*mass_ratio*mass_ratio/h/h

    if np.isscalar(stokes):
        return ret[0]

    return ret

def calc_torque_density_arm(r, u0, param_dict):
    acc_rad = (param_dict['taus']*param_dict['q']/3)**(1/3)

    phi1 = ((1-param_dict['eta'])*np.log(r) - 2*(r**(3/2)-1)/3)/u0 - 2*acc_rad
    phi2 = ((1-param_dict['eta'])*np.log(r) - 2*(r**(3/2)-1)/3)/u0 + 2*acc_rad

    pot1 = -1/np.sqrt(r*r + 1 - 2*r*np.cos(phi1) + param_dict['soft']**2)
    pot2 = -1/np.sqrt(r*r + 1 - 2*r*np.cos(phi2) + param_dict['soft']**2)

    return np.sqrt(r)*(pot2 - pot1)*np.heaviside(1-r,0)

def calc_torque_density_approx(r, param_dict, square=False):
    r_hill = (param_dict['q']/3)**(1/3)
    racc = (param_dict['taus']*param_dict['q'])**(1/3)

    u0 = -2*param_dict['eta']*param_dict['taus']

    s_star = racc
    s_cut = 0.4*r_hill
    sigma_gap = 0.5
    torque_reduc = 1 - sigma_gap

    s = r - 1.0

    if square is True:
        phi1 = (0.75*s_star*s_star + param_dict['eta']*s_star)/u0
        phi2 = s_cut

        pot1 = -1/np.sqrt(r*r + 1 - 2*r*np.cos(phi1) + param_dict['soft']**2)
        pot2 = -1/np.sqrt(r*r + 1 - 2*r*np.cos(phi2) + param_dict['soft']**2)

        return torque_reduc*((pot1 - pot2)*np.heaviside(s_star - np.abs(s), 0))*np.heaviside(s, 0)

    if np.isscalar(s):
        phi_dagger = 0.0
        if np.abs(s) < s_cut:
            phi_dagger = np.acos(0.5*((1+s)**2+1-s_cut**2)/(1+s))
    else:
        phi_dagger = np.where(np.abs(s) < s_cut, np.acos(0.5*((1+s)**2+1-s_cut**2)/(1+s)), np.zeros(len(s)))

    #phi = -phi_dagger - 0.75*(s*s - s_star*s_star)/u0
    phi = -(0.75*(s*s - s_star*s_star) + param_dict['eta']*(s-s_star))/u0
    if s_star < s_cut:
        phi = -np.acos(0.5*((1+s_star)**2+1-s_cut**2)/(1+s_star)) - (0.75*(s*s - s_star*s_star) + param_dict['eta']*(s-s_star))/u0

    pot1 = -1/np.sqrt(r*r + 1 - 2*r*np.cos(phi) + param_dict['soft']**2)
    pot2 = -1/np.sqrt(r*r + 1 - 2*r*np.cos(phi_dagger) + param_dict['soft']**2)

    #phi_arm = np.acos(1 - 0.5*s_cut**2) - 0.75*s*s/u0
    phi_arm = np.acos(1 - 0.5*s_cut**2) - (0.75*s*s + param_dict['eta']*s)/u0
    pot3 = -1/np.sqrt(r*r + 1 - 2*r*np.cos(phi_arm) + param_dict['soft']**2)

    s_trans = -4*param_dict['eta']/3

    #print(-0.75*s_star**2/u0)

    #plt.plot(s, phi*np.heaviside(s_star - s, 0))
    #plt.plot(s, phi_dagger)
    #plt.plot(s, phi_arm*np.heaviside(-s, 0))
    ##plt.xlim([-s_star,s_star])
    #plt.ylim([2*np.min(phi), -2*np.min(phi)])
    #plt.show()

    return torque_reduc*(pot1*np.heaviside(s_star - s, 0) - pot2*np.heaviside(s_star - s, 0)*np.heaviside(s - s_trans, 0) - pot3*np.heaviside(s_trans-s, 0))


def calc_torque_approx(param_dict):
    r_hill = (param_dict['q']/3)**(1/3)
    racc = (param_dict['taus']*param_dict['q'])**(1/3)

    u0 = -2*param_dict['eta']*param_dict['taus']

    s_star = 2*racc
    s_cut = 0.4*r_hill
    sigma_gap = 0.75
    torque_reduc = 1 - sigma_gap

    phi1 = (0.75*s_star*s_star + param_dict['eta']*s_star)/u0
    phi2 = s_cut

    num = lambda s: np.sqrt(s*s + 2*(1+s)*(1 - np.cos(phi1)) + param_dict['soft']**2) + 1 - np.cos(phi1) - s
    den = lambda s: np.sqrt(s*s + 2*(1+s)*(1 - np.cos(phi2)) + param_dict['soft']**2) + 1 - np.cos(phi2) + s

    return -torque_reduc*(np.log(num(s_star)/den(s_star)) - np.log(num(-s_star)/den(-s_star)))
