import numpy as np
import scipy as sp

from potential import PotentialComponent
from surface_density.fourier import calc_surface_density_1D, phase_func, integrating_factor, source

def torque_density_fourier(r, param_dict, m, vel_field, method='RK45', rtol=1.0e-3, atol=1.0e-6):
    # Planet potential component
    Phi = PotentialComponent(m, param_dict['soft'], method='Bessel')
    if m < 150:
        Phi = PotentialComponent(m, param_dict['soft'], method='Hyper')

    def rhs(t, y):
        rr = r[-1] - t
        #print(t, rr)
        ret = np.zeros_like(y)

        # Surface density rhs
        ret[0] = -(1j*m*(vel_field.Omega0*rr**(-1.5) - 1)*np.sqrt(rr)/vel_field.u0 - 1/rr)*y[0]
        ret[0] = ret[0] - rr*source(rr, m, vel_field)
        # Torque density rhs
        ret[1] = 1j*np.pi*rr*m*y[0]*Phi(rr)

        return -ret

    # t = r[-1] - r : t = [0, r[-1] - r[0]]

    sol = sp.integrate.solve_ivp(rhs, [0, r[-1] - r[0]], np.zeros(2, dtype=complex), method=method, t_eval=np.flip(r[-1]-r), rtol=rtol, atol=atol)
    return -np.flip(sol.y[1])

def total_torque_fourier(param_dict, m, vel_field, method='integral'):
    if method == 'ode':
        r_in = 0.6
        if m > 5:
            r_in = 0.7
        if m > 10:
            r_in = 0.8
        if m > 30:
            r_in = 0.9
        if param_dict['taus'] < 0.021:
            r_in = 0.8
            if m > 5:
                r_in = 0.9
            if m > 10:
                r_in = 0.95

        plot_flag = True
        if plot_flag is True:
            import matplotlib.pyplot as plt
            r = np.linspace(r_in, 1.5, 1000)
            ret = torque_density_fourier(r, param_dict, m, vel_field)
            plt.plot(r, np.real(ret))
            plt.plot(r, np.imag(ret))
            plt.title('m = {}'.format(m))
            plt.show()
        else:
            ret = torque_density_fourier(np.asarray([r_in,1.2]), param_dict, m, vel_field)

        return np.real(ret[0])

    # Planet potential component
    Phi = PotentialComponent(m, param_dict['soft'], method='Bessel')
    if m < 150:
        Phi = PotentialComponent(m, param_dict['soft'], method='Hyper')

    # Multiply by integrating factor to get only the integral part \int_r^\infty g(r)*exp(-i*k*f(r)) dr
    surf_dens = lambda x: calc_surface_density_1D(x, m, vel_field, method='integral', outer_radius=1.2)*integrating_factor(x, m, vel_field)*x**(1.5)

    # This is G(r): torque density is int_0^\infty pi*m*G(r)*exp(i*k*f(r))*dr
    G = lambda x: surf_dens(x)*np.sqrt(x)*Phi(x)

    # Calculate torque by numerical integration
    if method == 'integral':
        # Integral from 0 to infinity gives total torque for this m
        integrand = lambda x: 1j*np.pi*m*G(x)/integrating_factor(x, m, vel_field)/x

        k = -2*m/(3*vel_field.u0)
        #print('k = {}, k*b*b={}'.format(k, k*param_dict['soft']**2))

        rmin= 0.8
        n_fixed = int(1.5*(1/rmin - np.sqrt(rmin))*k*0.1)

        rc = np.power(vel_field.Omega0, 2/3)

        import matplotlib.pyplot as plt
        #Gtemp = lambda x: surf_dens(x)

        src = lambda x: source(x, m, vel_field)
        dr = 0.001
        d2src = (src(rc + dr) - 2*src(rc) + src(rc - dr))/dr/dr

        r = np.linspace(0.5, 1.2, 1000)
        #ret = np.zeros_like(r, dtype=complex)
        #for i in range(0, len(r)):
        #    n_fixed = int(k*np.max([np.abs(phase_func(r[i], vel_field)), np.abs(phase_func(r[-1], vel_field))]))
        #    print(i, n_fixed)
        #
        #    ret[i] = sp.integrate.fixed_quad(integrand, r[i], r[-1], n=n_fixed)[0]
        #    #ret[i] = sp.integrate.quad(integrand, r[i], r[-1], complex_func=True, limit=1000)[0]

        #plt.plot(r, np.real(ret))
        #plt.plot(r, np.imag(ret))

        #ret = integrand(r)
        ret = torque_density_fourier(r, param_dict, m, vel_field)
        plt.plot(r, np.real(ret))
        plt.plot(r, np.imag(ret))

        #plt.show()
        #exit()
        #ret = 1j*np.pi*m*G(r)
        #plt.plot(r, np.real(ret))
        #plt.plot(r, np.imag(ret))

        #rmax = rc #- 0.02
        #G_approx = lambda x:np.exp(-1j*k*phase_func(rc, vel_field) + 1j*np.pi/4)*np.sqrt(8*np.pi/(9*k))*(src(rc) + 2j*d2src*rc*rc/(9*k))*np.heaviside(rmax-x,1)*np.sqrt(x)*Phi(x)
        #ret_approx = 1j*np.pi*m*G_approx(r)
        #plt.plot(r, np.real(ret_approx))
        #plt.plot(r, np.imag(ret_approx))

        #ret = integrand(r)
        #plt.plot(r, np.real(ret))
        #plt.plot(r, np.imag(ret))
        #for i in range(0, len(r)):
        #    print(i, len(r))
        #    ret[i] = sp.integrate.fixed_quad(integrand, r[i], 1.2, n=n_fixed)[0]
        #plt.plot(r, np.real(ret))
        #plt.plot(r, np.imag(ret))

        #integrand = lambda x: 1j*np.pi*m*G(rc)/integrating_factor(x, m, vel_field)/x
        ret = sp.integrate.fixed_quad(integrand, 0.8, 1.2, n=n_fixed)[0]
        #ret = sp.integrate.quad(integrand, 0.8, 1.2, limit=1000, complex_func=True)[0]
        print('Exact: ', ret)
        torque_app = 1j*np.pi*m*4*np.pi*(src(rc) + 2j*rc*rc*d2src/(9*k))*rc*np.sqrt(rc)*Phi(rc)/(9*k)
        print('Stationary phase: ', torque_app)
        #integrand = lambda x: 1j*np.pi*m*G_approx(x)/integrating_factor(x, m, vel_field)/x
        #ret2 = sp.integrate.fixed_quad(integrand, 0.8, rmax, n=n_fixed)[0]
        #print('Approximate G: ', ret2)

        Gc = G(rc)
        d2Gc = (G(rc + dr) - 2*Gc + G(rc - dr))/dr/dr
        torque_app = 1j*np.pi*m*(Gc - 2j*d2Gc*rc*rc/(9*k))*np.sqrt(8*np.pi/(9*k))*np.exp(1j*k*phase_func(rc, vel_field) - 1j*np.pi/4)
        print('Classic stationary phase: ', torque_app)
        plt.title('m = {}'.format(m))
        plt.show()
        #ret=0.0

        #print(m, k, ret, torque_app)

        return np.real(ret)

    k = -2*m/(3*vel_field.u0)
    rc = np.power(vel_field.Omega0, 2/3)

    dr = 0.001
    d2Gc = (G(rc+dr) - 2*G(rc) + G(rc-dr))/dr/dr
    torque_app = G(rc)*np.sqrt(8*np.pi/(9*k))*np.exp(-1j*k*phase_func(rc, vel_field) + 1j*np.pi/4)
    torque_app2 = (G(rc) + 0.5j*d2Gc*4/9/k)*np.sqrt(8*np.pi/(9*k))*np.exp(-1j*k*phase_func(rc, vel_field) + 1j*np.pi/4)
    print(m, k, torque_app, torque_app2)

    return np.imag(np.pi*m*torque_app2)

def total_torque_sum(param_dict, vel_field, mrange=[1,200]):
    tq = np.zeros(mrange[1]+1)
    for m in range(mrange[0],mrange[1]+1):
        tq[m] = total_torque_fourier(param_dict, m, vel_field, method='ode')
        print(m, tq[m], np.sum(tq))

    #import matplotlib.pyplot as plt
    #plt.xscale('log')
    #plt.plot(tq)
    #plt.show()

    #plt.plot(np.cumsum(tq))
    #plt.show()

    return np.sum(tq)
