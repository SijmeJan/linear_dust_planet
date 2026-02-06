import numpy as np
import scipy as sp

from fargo import read_fargo

class VelocityField():
    def __init__(self):
        self.method = 'cubic'

    @classmethod
    def from_linear(cls, linear_vel_field, max_m, ignore_source=False, approx=False):
        ret = cls()
        r, phi = ret.nonuniform_coords(linear_vel_field.param_dict['q'], linear_vel_field.param_dict['soft'])

        u, v, src = linear_vel_field.calc_2D_sum(r, phi, max_m, grid_output=True, full_output=True, approx=approx)
        if ignore_source is True:
            src = 0.0*src

        ret.u = sp.interpolate.RegularGridInterpolator((r, phi), u, method=ret.method, bounds_error=False, fill_value=None)
        ret.v = sp.interpolate.RegularGridInterpolator((r, phi), v, method=ret.method, bounds_error=False, fill_value=None)
        ret.src = sp.interpolate.RegularGridInterpolator((r, phi), src, method=ret.method, bounds_error=False, fill_value=None)

        return ret

    @classmethod
    def from_fargo(cls, direc, number, n_fluid=1):
        ret = cls()

        r, phi, u, v = read_fargo(direc, number, n_fluid=n_fluid)

        ret.u = sp.interpolate.RegularGridInterpolator((r, phi), u.transpose(), method=ret.method, bounds_error=False, fill_value=None)
        ret.v = sp.interpolate.RegularGridInterpolator((r, phi), v.transpose(), method=ret.method, bounds_error=False, fill_value=None)
        ret.src = sp.interpolate.RegularGridInterpolator((r, phi), 0*u.transpose(), method=ret.method, bounds_error=False, fill_value=None)

        return ret


    def nonuniform_coords(self, q, soft, N_base=200):
        # Create 1D radial coordinate, clustered around the planet
        u = np.linspace(0.15,2.0,N_base)
        du = u[1] - u[0]

        # Clustering falls off on a scale of the Hill sphere
        beta = 2*(q/3)**(1/3)
        # Minimum dr is roughly 0.1 times the softening
        alpha = np.sqrt(1-0.1*soft/du)*beta
        # Function that needs to be inverted to get radial coordinates
        func = lambda x,y: x + alpha**2*np.atan((x-1)/np.sqrt(beta**2-alpha**2))/np.sqrt(beta**2-alpha**2) - y
        r = 0.0*u
        for i in range(0, len(r)):
            r[i] = sp.optimize.fsolve(lambda x: func(x,u[i]), u[i])[0]

        print('Min r, max r, min d:', r[0], r[-1], np.min(np.diff(r)))

        # Now do the same for the azimuthal coordinate
        up = np.linspace(-np.pi, np.pi, 3*N_base)
        du = up[1] - up[0]

        func = lambda x,y: x + alpha**2*np.atan(x/np.sqrt(beta**2-alpha**2))/np.sqrt(beta**2-alpha**2) - y
        phi = 0.0*up
        for i in range(0, len(phi)):
            phi[i] = sp.optimize.fsolve(lambda x: func(x,up[i]), up[i])[0]
        phi = -phi*np.pi/phi[0]
        if phi[-1] < np.pi:
            phi = phi*np.pi/phi[-1]

        return r, phi


