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
        # Calculate Laplace coefficient
        if  r < 1.0:
            ret = self.laplace(r, 0.5, self.m, 1, np.sqrt(1.0 + self.b*self.b))
        else:
            ret = self.laplace(1/r, 0.5, self.m, np.sqrt(1.0 + self.b*self.b), 1.0)/r

        # Add indirect term
        if self.m == 1:
            ret = ret - r

        # Modify m=0
        return -self.q*self.fac*ret

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
        if r < 1.0:
            ret = self.laplace.second_derivative(r, 0.5, self.m, 1, np.sqrt(1 + self.b*self.b))
        else:
            ret = (2*self.laplace(1.0/r, 0.5, self.m,
                                    np.sqrt(1.0 + self.b*self.b), 1.0)/(r*r*r) +
                   self.laplace.derivative(1.0/r, 0.5, self.m,
                                           np.sqrt(1.0 + self.b*self.b), 1.0)/(r*r*r*r) +
                   3*self.laplace.derivative(1.0/r, 0.5, self.m,
                                               np.sqrt(1.0 + self.b*self.b),
                                               1.0)/(r*r*r*r) +
                   self.laplace.second_derivative(1.0/r, 0.5, self.m,
                                               np.sqrt(1.0 + self.b*self.b),
                                               1.0)/(r*r*r*r*r))
        # Modify m=0
        return -self.q*self.fac*ret
