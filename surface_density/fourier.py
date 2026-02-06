import numpy as np
import scipy as sp

'''Surface density is given by

(int_r^\infty source(r')K(r') dr')/(K(r) r^(3/2))

with integrating factor K(r)
'''

def phase_func(r, vel_field):
    '''Phase function f(r)'''
    return 1.5*np.log(r) - r**(1.5)/vel_field.Omega0

def integrating_factor(r, m, vel_field):
    '''Integrating factor K(r)'''
    k = -2*m/vel_field.u0/3
    return np.exp(-1j*k*phase_func(r, vel_field))/r

def source(r, m, vel_field, include_terms=[1,1,1]):
    u, v = vel_field.calc_1D_profile(r, m)
    dudr, dvdr = vel_field.calc_derivative(r, m)

    return (include_terms[0]*r*dudr + 0.5*include_terms[1]*u + include_terms[2]*1j*m*v)/vel_field.u0/np.sqrt(r)

def dsource(r, m, vel_field, include_terms=[1,1,1]):
    dr = 0.001
    src = lambda x: source(x, m, vel_field, include_terms=include_terms)
    return 0.5*(src(r + dr) - src(r - dr))/dr

def dtsource(r, m, vel_field, include_terms=[1,1,1]):
    dr = 0.001
    src = lambda x: source(x, m, vel_field, include_terms=include_terms)
    return (src(r + dr) - 2*src(r) + src(r - dr))/dr/dr

def surface_density_from_ode(r, m, vel_field, method='DOP853', rtol=1.0e-3, atol=1.0e-6):
    def rhs(t, y):
        rr = r[-1] - t
        ret = np.zeros_like(y)

        # Surface density rhs
        ret[0] = -(1j*m*(vel_field.Omega0*rr**(-1.5) - 1)*np.sqrt(rr)/vel_field.u0 - 1/rr)*y
        ret[0] = ret[0] - rr*source(rr, m, vel_field)

        return -ret

    # t = r[-1] - r : t = [0, r[-1] - r[0]]

    sol = sp.integrate.solve_ivp(rhs, [0, r[-1] - r[0]], np.zeros(1, dtype=complex), method=method, t_eval=np.flip(r[-1]-r), rtol=rtol, atol=atol)
    ret = np.flip(sol.y[0])

    return ret

def surface_density_from_ode_old(r, m, vel_field):
    t0 = r[-1]+0.01 #2.0
    y0 = 0.0
    ret = np.zeros_like(r, dtype=complex)

    def rhs(t, y):
        ret = -(1j*m*(vel_field.Omega0*t**(-1.5) - 1)*np.sqrt(t)/vel_field.u0 - 1/t)*y
        return ret - t*source(t, m, vel_field)

    integrator = sp.integrate.ode(rhs).set_integrator('zvode', method='bdf')
    integrator.set_initial_value(y0, t0)

    for i in range(0, len(r)):
        integrator.integrate(r[-1-i])
        ret[-1-i] = integrator.y

    return ret

def calc_surface_density_1D(r, m, vel_field, method='ode', outer_radius=1.2):
    '''Surface density for single potential component'''

    # No perturbation for m=0
    if m == 0:
        return 0*r

    r = np.asarray(r)
    scalar_input = False
    if r.ndim == 0:
        r = r[None]  # Makes x 1D
        scalar_input = True

    if method == 'ode':
        ret = surface_density_from_ode(r, m, vel_field)*r**(-1.5)

    if method == 'integral':
        src = lambda x: source(x, m, vel_field)
        k = -2*m/vel_field.u0/3

        integral_exact = 0*r + 1j
        integrand = lambda x: src(x)*integrating_factor(x, m, vel_field)

        rmin= 0.8
        n_fixed = int(1.5*(1/rmin - np.sqrt(rmin))*k*0.1)
        #n_fixed = int(k*np.max([np.abs(phase_func(r[0], vel_field)), np.abs(phase_func(r[-1], vel_field))]))
        # No strong oscillations for r > 1: easier integral
        if np.min(r) > 0.95:
            n_fixed = 128
        for i in range(0, len(r)):
            #integral_exact[i] = sp.integrate.quad(lambda x: np.real(integrand(x)), r[i], outer_radius, complex_func=True, limit=1000)[0]
            integral_exact[i] = sp.integrate.fixed_quad(integrand, r[i], outer_radius, n=n_fixed)[0]
        ret = r**(-1.5)*integral_exact/integrating_factor(r, m, vel_field)

    if method == 'approx':
        rc = np.power(vel_field.Omega0, 2/3)
        src = lambda x: source(x, m, vel_field)
        k = -2*m/vel_field.u0/3
        dr = 0.001
        dsrc = 0.5*(src(rc + dr) - src(rc - dr))/dr
        d2src = (src(rc + dr) - 2*src(rc) + src(rc - dr))/dr/dr
        ret = np.exp(1j*np.pi/4)*np.sqrt(2*np.pi/(9*k))*r*(src(rc) + 2j*rc*rc*d2src/(9*k)) + 4j*rc*rc*dsrc/(9*k)

    plot_flag = False
    if plot_flag is True:
        import matplotlib.pyplot as plt
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

        plt.plot(r, np.real(ret), color=colors[0])
        plt.plot(r, np.imag(ret), color=colors[1])

        rc = np.power(vel_field.Omega0, 2/3)
        src = lambda x: source(x, m, vel_field)
        k = -2*m/vel_field.u0/3
        dr = 0.001
        dsrc = 0.5*(src(rc + dr) - src(rc - dr))/dr
        d2src = (src(rc + dr) - 2*src(rc) + src(rc - dr))/dr/dr
        surf_coro = np.exp(1j*np.pi/4)*np.sqrt(2*np.pi/(9*k))*rc*(src(rc) + 2j*rc*rc*d2src/(9*k)) + 4j*rc*rc*dsrc/(9*k)
        plt.plot([rc],[np.real(surf_coro)], marker='o', linestyle='None', color=colors[0])
        plt.plot([rc],[np.imag(surf_coro)], marker='o', linestyle='None', color=colors[1])

        rr = np.linspace(rc + 2/k, r[-1], 100)
        df = 3*(1-(rr/rc)**(3/2))/(2*rr)
        d2f = -1.5*(1+0.5*(rr/rc)**(3/2))/(rr*rr)
        dsrc = np.gradient(src(rr), rr)
        surf_far = -1j*src(rr)/(k*df) - (dsrc*df - src(rr)*d2f)/(k*k*df*df*df)
        plt.plot(rr, np.real(surf_far), color=colors[0], linestyle='--')
        plt.plot(rr, np.imag(surf_far), color=colors[1], linestyle='--')

        rr = np.linspace(r[0], rc, int(10*(1-r[0])*k))
        f = 1.5*np.log(rr) - (rr/rc)**(1.5)
        df = 3*(1-(rr/rc)**(3/2))/(2*rr)
        d2f = -1.5*(1+0.5*(rr/rc)**(3/2))/(rr*rr)
        fc = 1.5*np.log(rc) - 1.0
        d2src = (src(rc + dr) - 2*src(rc) + src(rc - dr))/dr/dr
        surf_stationary = np.exp(1j*k*(f-fc) + 1j*np.pi/4)*np.sqrt(8*np.pi/(9*k))*rc*(src(rc) + 0*4j*d2src*rc*rc/(18*k))/np.sqrt(rr) + 0*1j*(src(rr) - src(rc))/df/k/np.sqrt(rr)
        plt.plot(rr, np.real(surf_stationary), color=colors[0], linestyle=':')
        plt.plot(rr, np.imag(surf_stationary), color=colors[1], linestyle=':')

        plt.ylim([-1.5*np.max(np.abs(ret)), 1.5*np.max(np.abs(ret))])
        plt.show()

    if scalar_input:
        return np.squeeze(ret)

    return ret

def calc_2D_surface_density(r, phi, m, vel_field, grid_output=True, method='ode'):
    Sigma1D = calc_surface_density_1D(r, m, vel_field, method=method)

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

def calc_surface_density_sum(r, phi, mrange, vel_field, grid_output=True, method='ode'):
    # Calculate m=0 component
    print(mrange[0])
    Sigma = calc_2D_surface_density(r, phi, mrange[0], vel_field, grid_output=grid_output, method=method)

    for m in range(mrange[0]+1, mrange[1]+1):
        print(m)
        Sigma = Sigma + calc_2D_surface_density(r, phi, m, vel_field, grid_output=grid_output, method=method)

    return Sigma

