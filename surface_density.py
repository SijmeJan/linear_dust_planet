import numpy as np
import scipy as sp

from laplace import LaplaceCoefficient


def calc_surface_density_along_streamlines(moc, param_dict, N,
                                           r_start, phi_start=None, u_start=None,
                                           method='DOP853', reverse_flag=False, dt=0.2*np.pi, max_steps=None, r_finish=None):
    '''Calculate surface density Sigma using the Method of Lines. Follow a set of points, starting at
    r_start, phi_start, with starting u=log(r*Sigma). Returns the paths (r,phi) and the surface density along
    the streamlines.'''

    # Calculate integration time to either
    if r_finish is None:
        r_finish = 0.85
        if reverse_flag is True:
            r_finish = 1.5
    t_max = (1 + param_dict['taus']**2)*np.abs(r_finish**(1.5) - r_start**(1.5))/(3*param_dict['taus']*param_dict['eta'])
    t_max = np.max(np.atleast_1d(t_max))
    if max_steps is None:
        max_steps = int(t_max/dt)

    if len(np.atleast_1d(r_start)) != N:
        # All start at same radial coordinate
        start_x = r_start*np.ones(N)
    else:
        start_x = r_start
    # By default, distribute starting locations over phi
    if phi_start is None:
        start_y = np.linspace(-np.pi, np.pi, N, endpoint=False)
    else:
        start_y = phi_start
    # By default, start with equilibrium surface density
    if u_start is None:
        start_u = np.log(np.sqrt(start_x))
    else:
        start_u = u_start

    # Method of Lines
    r, phi, u = moc.calculate(start_x, start_y, start_u, dt=dt, max_steps=max_steps, method=method, reverse_flag=reverse_flag)

    # Convert to surface density
    Sigma = np.exp(u)/r

    # Adjust surface density, assuming unperturbed at r=r_out
    if reverse_flag is True:
        Sigma_out = Sigma[:,-1]
        r_out = r[:,-1]

        for i in range(0, N):
            if np.abs(r_out[i]-1.0) < 0.1:
                Sigma[i,:] = 0.0*Sigma[i,:]
            else:
                Sigma[i,:] = Sigma[i,:]/(np.sqrt(r_out[i])*Sigma_out[i])

    # For convenience, remap phi to [-pi,pi]
    phi = (phi + np.pi) % (2*np.pi) - np.pi

    return r, phi, Sigma

def surface_density(moc, param_dict, N=512, r_start=1.2, dt=0.2*np.pi, r_finish=None, r_circle=0.01, reverse_flag=True, streamline_plot=False):
    '''Calculate surface density using MoC, using starting points for
       streamlines that give a decent covering of the domain.
       Returns the paths (r,phi) and the surface density.'''

    # Corotation radius
    rc=(1-param_dict['eta'])**(2/3)

    # Add a circle of points around the planet
    N2 = 0
    if r_circle > 0.0:
        N2 = int(32*N/128)
        if streamline_plot is True:
            N2 = N

    # 'ordinary' starting points (i.e. not on circle)
    r_start = r_start*np.ones(N-N2)
    phi_start = np.linspace(-np.pi, np.pi, N-N2, endpoint=False)

    # Extra starting points
    s = r_circle
    theta = np.linspace(0, 2*np.pi, N2)
    r_extra = np.sqrt(s*s + rc + 2*s*rc*np.cos(theta))
    phi_extra = np.atan2(s*np.sin(theta), rc + s*np.cos(theta))

    # Check if extra starting points end on planet
    r, phi, Sigma = calc_surface_density_along_streamlines(moc, param_dict, N2,
                                                           r_extra, phi_start=phi_extra, u_start=0*phi_extra,
                                                           reverse_flag=False, dt=dt, r_finish=r_finish)
    # Only add points that end up on the planet
    d = np.sqrt(r*r + 1 - 2*r*np.cos(phi))
    for n in range(0, N2):
        if np.max(d[n,:]) == d[n,0]:
            r_start = np.append(r_start, r_extra[n])
            phi_start = np.append(phi_start, phi_extra[n])

    print('HALLO: ', np.shape(r_start), np.shape(phi_start))

    N = len(r_start)
    u_start = np.zeros(N)
    # Combine all starting points
    #r_start = np.concatenate((r_start, r_extra), axis=None)
    #phi_start = np.concatenate((phi_start,phi_extra), axis=None)

    # Calculate surface density
    r, phi, Sigma = calc_surface_density_along_streamlines(moc, param_dict, N,
                                                           r_start, phi_start=phi_start, u_start=u_start,
                                                           reverse_flag=reverse_flag, dt=dt, r_finish=r_finish)

    return r,phi,Sigma

def calc_surface_density_moc(r, phi, moc, param_dict):
    '''Calculate surface density at locations (r, phi) using MoC'''
    rr = np.atleast_1d(r)
    pp = np.atleast_1d(phi)

    # What do we want to do if rr and pp are not of the same size?
    if len(rr) == 1:
        rr = rr[0]*np.ones(len(pp))
    if len(pp) == 1:
        pp = pp[0]*np.ones(len(rr))
    if len(rr) != len(pp):
        print('ERROR: r and phi need to have the same shape')
        exit()

    # Set surface density to zero inside accretion radius
    fac = np.ones(len(rr))
    #dd = np.sqrt(rr*rr + 1.0 - 2*rr*np.cos(pp))
    #fac[np.asarray(dd < (param_dict['taus']*param_dict['q']/3)**(1/3)).nonzero()] = 0

    r0, phi0, Sigma = calc_surface_density_along_streamlines(moc, param_dict, len(rr),
                                                             rr, phi_start=pp, u_start=0*rr,
                                                             reverse_flag=True, r_finish=1.5)

    #import matplotlib.pyplot as plt
    #plt.plot(phi0[0,:], r0[0,:])
    #plt.plot(r0[0,:], Sigma[0,:])
    #plt.show()

    if np.isscalar(phi):
        return fac[0]*Sigma[0,0]
    return fac*Sigma[:,0]

def calc_surface_density_1D(r, m, u0, vel_field):
    u, v = vel_field.calc_1D_profile(r, m)

    u1, v1 = vel_field.calc_1D_profile(np.asarray([1.0]), m)

    x = r**(1.5)
    eps = 1.5*u0
    heaviside = np.asarray(x < 1).astype(int)

    ret = -u/u0 - heaviside*np.exp(-1j*m*(np.log(x)-x+1)/eps - 1j*np.pi/4)*(1j*m*v1- u1)*np.sqrt(2*np.pi*np.abs(eps))/(eps*x)

    return ret

def calc_2D_surface_density(r, phi, m, u0, vel_field, grid_output=True):
    Sigma1D = calc_surface_density_1D(r, m, u0, vel_field)

    # Add unperturbed surface density
    if m == 0:
        Sigma1D = Sigma1D + 1/np.sqrt(r)

    if grid_output is True:
        # Output on r,phi grid
        Sigma = np.zeros((len(r), len(phi)), dtype=np.complex128)

        for i in range(0, len(phi)):
            Sigma[:,i] = Sigma1D*np.exp(1j*m*phi[i])
    else:
        if np.shape(r) != np.shape(phi):
            raise TypeError("r and phi need to have the same shape for 1D output")

        Sigma = Sigma1D*np.exp(1j*m*phi)

    return Sigma

def calc_surface_density_sum(r, phi, mmax, u0, vel_field, grid_output=True):
    # Calculate m=0 component
    Sigma = calc_2D_surface_density(r, phi, 0, u0, vel_field, grid_output=grid_output)

    for m in range(1, mmax):
        Sigma = Sigma + calc_2D_surface_density(r, phi, m, u0, vel_field, grid_output=grid_output)

    return Sigma

def calc_surface_density_approx(r, phi, u0, param_dict):
    rr = np.atleast_1d(r)
    pp = np.atleast_1d(phi)

    if len(rr) == 1 and len(pp) > 1:
        rr = rr[0]*np.ones(len(phi))

    Sigma = 0*rr

    dist_to_planet = np.sqrt(rr*rr + 1.0 - 2*rr*np.cos(pp))

    for i in range(0, len(rr)):
        if dist_to_planet[i] > (param_dict['taus']*param_dict['q']/3)**(1/3):
            rsym = rr[i]
            psym = pp[i]
            if rr[i] < 1:
                # Force symmetry
                #rsym = 2 - rr[i]
                rsym = 1.0
                psym = pp[i] - ((1-param_dict['eta'])*np.log(rr[i]) - 2*(rr[i]**(1.5)-1)/3)/u0
                while psym < -np.pi:
                    psym += 2*np.pi
                while psym > np.pi:
                    psym -= 2*np.pi

            if psym > 0.0:
                psym = psym - 2*np.pi

            x_star = 1 + np.sqrt((rsym-1)**2 + 4*u0*psym/3)

            B = LaplaceCoefficient()

            eps = param_dict['soft']
            x = 1/x_star
            b  = B(x, 0.5, 0, np.sqrt(1.0 + eps*eps), 1)
            db = B.derivative(x, 0.5, 0, np.sqrt(1.0 + eps*eps), 1)
            d2b= B.second_derivative(x, 0.5, 0, np.sqrt(1.0 + eps*eps), 1)

            #print(rr[i], pp[i], x_star, x)

            #du = x**(9/2)*(3.5*b + 5.5*x*db + x*x*d2b)/(x**(1.5)-1)
            du = x**(3/2)*(0.5*b + 2.5*x*db + x*x*d2b)/(x**(1.5)-1)

            du = np.pi*param_dict['taus']*param_dict['q']*du

            if du > 0.0:
                Sigma[i] = 0.0
            else:
                Sigma[i] = np.exp(du)/np.sqrt(rr[i])

    if np.isscalar(phi):
        return Sigma[0]
    return Sigma
    #return np.exp(dlog)/np.sqrt(r)

def radial_scale(param_dict):
    B = LaplaceCoefficient()

    eps = param_dict['soft']

    def func(x):
        b  = B(x, 0.5, 0, np.sqrt(1.0 + eps*eps), 1)
        db = B.derivative(x, 0.5, 0, np.sqrt(1.0 + eps*eps), 1)
        d2b= B.second_derivative(x, 0.5, 0, np.sqrt(1.0 + eps*eps), 1)
        return param_dict['taus']*param_dict['q']*x**(9/2)*(3.5*b + 5.5*x*db + x*x*d2b)/(x**(1.5)-1) + 1

    #import matplotlib.pyplot as plt

    #x = np.linspace(0.8, 0.99, 100)
    #ret = 0*x
    #for i in range(0, len(x)):
    #    ret[i] = func(x[i])
    #plt.plot(x, ret)
    #plt.show()

    xtilde = sp.optimize.fsolve(func, 0.99999)

    return 1/xtilde[0]




class SurfaceDensity():
    def __init__(self, param_dict, r=None, phi=None, Sigma=None):
        self.param_dict = param_dict

        self.Sigma = Sigma
        self.r = r
        self.phi = phi

        return

    @classmethod
    def from_moc(cls, param_dict, moc, r_start=0.9, N=2048, r_circle=0.01, reverse_flag=True, dt=None):
        ret = cls(param_dict)
        if dt is None:
            dt = 0.2*np.pi*0.1/param_dict['taus']

        r, phi, Sigma = surface_density(moc, param_dict, r_start=r_start, N=N, dt=dt, r_finish=1.5, r_circle=r_circle, reverse_flag=reverse_flag)

        ret.Sigma = Sigma
        ret.r = r
        ret.phi = phi

        return ret

    @classmethod
    def from_approx(cls, param_dict, r, phi):
        func = lambda x: -0.5*x*x - (1-param_dict['eta'])**2/(1 + 0.5*param_dict['taus']*x)**2 + 1 + x/param_dict['taus']
        u0 = sp.optimize.fsolve(func, 0.0)[0]

        Sigma = np.zeros((len(r), len(phi)))
        for i in range(0,len(r)):
            Sigma[i,:] = calc_surface_density_approx(r[i], phi, u0, param_dict)

        return cls(param_dict, r=r, phi=phi, Sigma=Sigma)

    @classmethod
    def from_file(cls, filename):
        fd = np.load(filename)

        param_dict = {
            'soft' : fd['param'][0],                       # Potential softening (0.007)
            'q' : fd['param'][1],                          # Planet/star mass ratio
            'taus': fd['param'][2],                        # Stokes number
            'eta': fd['param'][3],
            'zero_velocity_background' : True              # Take bkg radial velocity into perturbation?
        }

        return cls(param_dict, r=fd['r'], phi=fd['phi'], Sigma=fd['Sigma'])

    def save_to_file(self, filename):
        param = np.asarray([self.param_dict['soft'],
                            self.param_dict['q'],
                            self.param_dict['taus'],
                            self.param_dict['eta']])

        np.savez(filename, r=self.r, phi=self.phi, Sigma=self.Sigma, param=param)

    def __call__(self, r, phi):
        points = list(zip(self.r.flatten(), self.phi.flatten()))
        values = self.Sigma.flatten()
        lin_int = sp.interpolate.LinearNDInterpolator(points, values, fill_value=np.nan, rescale=False)
        return lin_int(r, phi)
