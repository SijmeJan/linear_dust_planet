import numpy as np

from laplace import LaplaceCoefficient

class PotentialComponent():
    def __init__(self, m, b, method='Hyper'):
        self.laplace = LaplaceCoefficient(method)
        self.m = m
        self.b = b
        self.q = 1.0
        self.fac = 1.0
        if m == 0:
            self.fac = 0.5

    def __call__(self, r):
        r = np.asarray(r)
        scalar_input = False
        if r.ndim == 0:
            r = r[None]  # Makes x 1D
            scalar_input = True

        ret = 0*r
        for i in range(0, len(r)):
            # Calculate Laplace coefficient
            if  r[i] < 1.0:
                ret[i] = self.laplace(r[i], 0.5, self.m, 1, np.sqrt(1.0 + self.b*self.b))
            else:
                ret[i] = self.laplace(1/r[i], 0.5, self.m, np.sqrt(1.0 + self.b*self.b), 1.0)/r[i]

            # Add indirect term
            if self.m == 1:
                ret[i] = ret[i] - r[i]

        # Modify m=0
        ret = -self.q*self.fac*ret

        if scalar_input:
            return np.squeeze(ret)

        return ret

    def derivative(self, r):
        if r < 1.0:
            ret = self.laplace.derivative(r, 0.5, self.m, 1, np.sqrt(1 + self.b*self.b))
        else:
            ret = -(self.laplace(1.0/r, 0.5, self.m,
                                    np.sqrt(1.0 + self.b*self.b), 1.0)/(r*r) +
                   self.laplace.derivative(1.0/r, 0.5, self.m,
                                               np.sqrt(1.0 + self.b*self.b),
                                               1.0)/(r*r*r))

        # Add indirect term
        if self.m == 1:
            ret = ret - 1.0

        # Modify m=0
        return -self.q*self.fac*ret

    def second_derivative(self, r):
        r = np.asarray(r)
        scalar_input = False
        if r.ndim == 0:
            r = r[None]  # Makes x 1D
            scalar_input = True

        ret = 0*r
        for i in range(0, len(r)):
            if r[i] < 1.0:
                ret[i] = self.laplace.second_derivative(r[i], 0.5, self.m, 1, np.sqrt(1 + self.b*self.b))
            else:
                ret[i] = (2*self.laplace(1.0/r[i], 0.5, self.m,
                                      np.sqrt(1.0 + self.b*self.b), 1.0)/(r[i]*r[i]*r[i]) +
                       self.laplace.derivative(1.0/r[i], 0.5, self.m,
                                           np.sqrt(1.0 + self.b*self.b), 1.0)/(r[i]*r[i]*r[i]*r[i]) +
                       3*self.laplace.derivative(1.0/r[i], 0.5, self.m,
                                               np.sqrt(1.0 + self.b*self.b),
                                               1.0)/(r[i]*r[i]*r[i]*r[i]) +
                       self.laplace.second_derivative(1.0/r[i], 0.5, self.m,
                                               np.sqrt(1.0 + self.b*self.b),
                                               1.0)/(r[i]*r[i]*r[i]*r[i]*r[i]))
        ret = -self.q*self.fac*ret

        if scalar_input:
            return np.squeeze(ret)

        return ret

