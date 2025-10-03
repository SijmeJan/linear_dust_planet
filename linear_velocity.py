import numpy as np

from potential import PotentialComponent

class LinearVelocity():
    def __init__(self, param_dict):
        self.param_dict = param_dict

    def calc_1D_profile(self, r, m, full_output=False):
        '''Returns velocity perturbations due at azimuthal wavenumber m at specific radii'''

        # Keplerian angular velocity. Change to equilibrium?
        Omega = r**(-1.5)
        tau = self.param_dict['taus']*r**(1.5)

        # Extra term taking into account m=0 from radial pressure gradient
        Pi = 0.0
        if self.param_dict['zero_velocity_background'] is True:
            if m == 0:
                Pi = -self.param_dict['eta']*r**(-2)/self.param_dict['taus']

        # Planet potential component
        Phi = PotentialComponent(m, self.param_dict['soft'], method='Bessel')
        if m < 150:
            Phi = PotentialComponent(m, self.param_dict['soft'], method='Hyper')

        pot = 0*r
        dpot = 0*r
        for i in range(0, len(r)):
            pot[i] = self.param_dict['q']*Phi(r[i])
            dpot[i] = self.param_dict['q']*Phi.derivative(r[i])

        # indirect term
        if m == 1:
            pot = pot + self.param_dict['q']*r
            dpot = dpot + self.param_dict['q']

        # Calculate velocity perturbations
        sigma = 1j*m*(Omega - 1.0) + 1/tau
        u = -(2j*Omega*m*pot/r - 2*Omega*Pi + sigma*dpot)/(sigma*sigma + Omega*Omega)
        v = -0.5*Omega*u/sigma - 1j*m*pot/sigma/r + Pi/sigma

        if full_output is True:
            d2pot = 0*r
            for i in range(0, len(r)):
                d2pot[i] = self.param_dict['q']*Phi.second_derivative(r[i])

            dudr = -Omega*Pi - 1j*m*Omega*(pot + 2*r*dpot)/r + 1.5*(1j*m-sigma)*dpot - r*sigma*d2pot - (2j*m*Omega*pot/r + sigma*dpot)*3j*sigma*m/(sigma*sigma + Omega*Omega)
            dudr = dudr/r/(sigma*sigma + Omega*Omega)

            # In addition output MoC source term
            return u, v, -dudr-1j*m*v/r

        return u, v

    def calc_2D_component(self, r, phi, m, grid_output=True, full_output=False):
        res1D = self.calc_1D_profile(r, m, full_output=full_output)

        if full_output is False:
            u1D, v1D = res1D
        else:
            u1D, v1D, src1D = res1D

        if grid_output is True:
            # Output on r,phi grid
            u = np.zeros((len(r), len(phi)), dtype=np.complex128)
            v = np.zeros((len(r), len(phi)), dtype=np.complex128)
            if full_output is True:
                src = np.zeros((len(r), len(phi)), dtype=np.complex128)

            for i in range(0, len(phi)):
                u[:,i] = u1D*np.exp(1j*m*phi[i])
                v[:,i] = v1D*np.exp(1j*m*phi[i])
                if full_output is True:
                    src[:,i] = src1D*np.exp(1j*m*phi[i])
        else:
            if np.shape(r) != np.shape(phi):
                raise TypeError("r and phi need to have the same shape for 1D output")

            u = u1D*np.exp(1j*m*phi)
            v = v1D*np.exp(1j*m*phi)
            if full_output is True:
                src = src1D*np.exp(1j*m*phi)

        if full_output is True:
            return u, v, src

        return u, v

    def calc_2D_sum(self, r, phi, mmax, grid_output=True, full_output=False, approx=False):
        if approx is True:
            eta = self.param_dict['eta']
            eps = self.param_dict['taus']

            if grid_output is True:
                rr, pp = np.meshgrid(r, phi, indexing='ij')
            else:
                rr = r
                pp = phi

            Omega = 1/np.sqrt(rr)/rr
            s = np.sqrt(rr*rr + 1.0 - 2*rr*np.cos(pp) + self.param_dict['soft']**2)
            dsdr = (rr -np.cos(pp))/s

            dpotdr = self.param_dict['q']*(rr -np.cos(pp))/s**3
            d2potdr2 = self.param_dict['q']*(1/s**3 - 3*(rr-np.cos(pp))**2/s**5)
            dpotdp = self.param_dict['q']*rr*np.sin(pp)/s**3
            d2potdp2 = self.param_dict['q']*(rr*np.cos(pp)/s**3 - 3*rr*rr*np.sin(pp)*np.sin(pp)/s**5)
            d2potdrdp = self.param_dict['q']*(np.sin(pp)/s**3 - 3*rr*np.sin(pp)*dsdr/s**4)
            d3potdrdp2 = self.param_dict['q']*(np.cos(pp)/s**3 - 3*rr*np.cos(pp)*dsdr/s**4
                                               - 6*rr*np.sin(pp)*np.sin(pp)/s**5
                                               + 15*rr*rr*np.sin(pp)*np.sin(pp)*dsdr/s**6)

            u0 = 0.0*rr
            u1 = -2*eta*rr*Omega - dpotdr/Omega
            u2 = -2*dpotdp/rr/Omega + (Omega - 1.0)*d2potdrdp/Omega/Omega
            u3 = -2*eta*rr*Omega - dpotdr/Omega + \
                2*(Omega-1)*(2*d2potdp2/rr/Omega - (Omega - 1.0)*d3potdrdp2/Omega/Omega)/Omega + \
                (Omega - 1)**2*d2potdrdp/Omega/Omega/Omega
            #u4 = -u2 - 2*(Omega-1)*(u3)'/Omega - (Omega-1)**2*(u2)''/Omega/Omega

            u = u0 + eps*u1 + eps*eps*u2 + eps*eps*eps*u3

            v0 = -eta*rr*Omega
            v1 = -dpotdp/Omega/rr
            v2 = -0.5*u1 + 0.5*(Omega - 1.0)**2*d3potdrdp2/Omega**3
            v3 = -0.5*u2 + 0.5*(Omega-1)**2*d3potdrdp2/Omega/Omega/Omega

            v = v0 + eps*v1 + eps*eps*v2 + eps*eps*eps*v3

            if full_output is True:
                dudr = eps*(eta*Omega - 1.5*dpotdr*np.sqrt(rr) - d2potdr2/Omega)
                dvdp = -eps*d2potdp2/Omega

                return u, v, -dudr + -dvdp/rr

            return u, v

        # Calculate m=0 component
        res = self.calc_2D_component(r, phi, 0, grid_output=grid_output, full_output=full_output)

        if full_output is False:
            u, v = res
        else:
            u, v, src = res

        for m in range(1, mmax):
            res = self.calc_2D_component(r, phi, m, grid_output=grid_output, full_output=full_output)
            if full_output is False:
                u1, v1 = res
            else:
                u1, v1, src1 = res

            u = u + u1
            v = v + v1
            if full_output is True:
                src = src + src1

        if full_output is True:
            return u, v, src

        return u, v

