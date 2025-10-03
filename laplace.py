# -*- coding: utf-8 -*-
"""Class dealing with calculating generalized Laplace coefficients

Generalized Laplace coefficients can be calculated either by brute force integration (slow), hypergeometric functions (faster), or Bessel fuctions (fastest but approximate).

"""

from __future__ import print_function

import numpy as np
import scipy.integrate as integrate
import scipy.special as sp

class LaplaceCoefficient():
    """Class containing functions to calculate generalized Laplace coefficients.

    The generalized Laplace coefficients are defined by

    .. math::
        b_s^m(a) = \\frac{2}{\pi}\int_0^\pi \\frac{\cos(m\phi)d\phi}{(q^2 + p^2a^2 - 2a\cos(\phi))^s}

    Args:
        method (str): way to calculate the coefficients, either 'Brute' (brute force integration, slow but exact), 'Hyper' (hypergeometric functions, faster and exact), or 'Bessel' (Bessel functions, fastest but approximate)
    """

    def __init__(self, method='Hyper'):
        if (method != 'Bessel' and
            method != 'Hyper' and
            method != 'Brute'):
            print("Error: method should be either 'Bessel', 'Hyper', or 'Brute'")

        # Set functions according to method used
        self._calc = None
        self._calc_derivative = None
        if method == 'Bessel':
            self._calc = self._laplace_bessel
            self._calc_derivative = self._laplace_derivative_bessel
            self._calc_second_derivative = self._laplace_second_derivative_bessel
        if method == 'Hyper':
            self._calc = self._laplace_hyper
            self._calc_derivative = self._laplace_derivative_hyper
            self._calc_second_derivative = self._laplace_second_derivative_hyper
        if method == 'Brute':
            self._calc = self._laplace_brute
            self._calc_derivative = self._laplace_derivative_brute
            self._calc_second_derivative = self._laplace_second_derivative_brute

    def __call__(self, a, s, m, p, q):
        """Calculate generalized Laplace coefficient.

        Args:
            a (float): radius-like coordinate, must be smaller than unity
            s (float): power to which denominator is raised
            m (float): numerator of integrant is :math:`\cos(m\phi)`
            p (float): factor multiplying :math:`a^2` in denominator
            q (float): constant term in denominator

        """
        return self._calc(a, s, m, p, q)

    def derivative(self, a, s, m, p, q):
        """Calculate derivative with respect to :math:`a` of generalized Laplace coefficient.

        Args:
            a (float): radius-like coordinate, must be smaller than unity
            s (float): power to which denominator is raised
            m (float): numerator of integrant is :math:`\cos(m\phi)`
            p (float): factor multiplying :math:`a^2` in denominator
            q (float): constant term in denominator

        """
        return self._calc_derivative(a, s, m, p, q)

    def second_derivative(self, a, s, m, p, q):
        """Calculate second derivative with respect to :math:`a` of generalized Laplace coefficient.

        Args:
            a (float): radius-like coordinate, must be smaller than unity
            s (float): power to which denominator is raised
            m (float): numerator of integrant is :math:`\cos(m\phi)`
            p (float): factor multiplying :math:`a^2` in denominator
            q (float): constant term in denominator

        """
        return self._calc_second_derivative(a, s, m, p, q)

    def _laplace_brute(self, a, s, m, p, q):
        """Calculate generalized Laplace coefficient by brute force integration.

        Args:
            a (float): radius-like coordinate, must be smaller than unity
            s (float): power to which denominator is raised
            m (float): numerator of integrant is :math:`\cos(m\phi)`
            p (float): factor multiplying :math:`a^2` in denominator
            q (float): constant term in denominator

        """

        f = lambda x: np.cos(m*x)*np.power(p*p*a*a - 2.0*a*np.cos(x) + q*q, -s)
        res = integrate.quad(f, 0, np.pi, limit=100)
        ret = 2.0*res[0]/np.pi

        return ret

    def _laplace_derivative_brute(self, a, s, m, p, q):
        """Calculate derivative of generalized Laplace coefficient by brute force integration.

        Args:
            a (float): radius-like coordinate, must be smaller than unity
            s (float): power to which denominator is raised
            m (float): numerator of integrant is :math:`\cos(m\phi)`
            p (float): factor multiplying :math:`a^2` in denominator
            q (float): constant term in denominator

        """
        f = lambda x: -2.0*s*np.cos(m*x)* \
          np.power(p*p*a*a - 2.0*a*np.cos(x) + q*q, -s-1.0)*(p*p*a - np.cos(x))
        res = integrate.quad(f, 0, np.pi, limit=100)
        ret = 2.0*res[0]/np.pi

        return ret

    def _laplace_second_derivative_brute(self, a, s, m, p, q):
        """Calculate second derivative of generalized Laplace coefficient by brute force integration.

        Args:
            a (float): radius-like coordinate, must be smaller than unity
            s (float): power to which denominator is raised
            m (float): numerator of integrant is :math:`\cos(m\phi)`
            p (float): factor multiplying :math:`a^2` in denominator
            q (float): constant term in denominator

        """
        f = lambda x: 4.0*s*(s+1)*np.cos(m*x)* \
          np.power(p*p*a*a - 2.0*a*np.cos(x) + q*q, -s-2.0)*(p*p*a - np.cos(x))**2 -\
          2*p*p*s*np.cos(m*x)*np.power(p*p*a*a - 2.0*a*np.cos(x) + q*q, -s-1.0)
        res = integrate.quad(f, 0, np.pi, limit=100)
        ret = 2.0*res[0]/np.pi

        return ret

    def _laplace_bessel(self, a, s, m, p, q):
        """Calculate generalized Laplace coefficient using Bessel functions.

        Args:
            a (float): radius-like coordinate, must be smaller than unity
            s (float): power to which denominator is raised
            m (float): numerator of integrant is :math:`\cos(m\phi)`
            p (float): factor multiplying :math:`a^2` in denominator
            q (float): constant term in denominator

        """
        K = sp.kn(0, m*np.sqrt((q*q + p*p*a*a - 2.0*a)/a))

        return 2.0*K/(np.pi*np.sqrt(a))

    def _laplace_derivative_bessel(self, a, s, m, p, q):
        """Calculate derivative of generalized Laplace coefficient using Bessel functions.

        Args:
            a (float): radius-like coordinate, must be smaller than unity
            s (float): power to which denominator is raised
            m (float): numerator of integrant is :math:`\cos(m\phi)`
            p (float): factor multiplying :math:`a^2` in denominator
            q (float): constant term in denominator

        """
        K = sp.kn(1, m*np.sqrt((q*q + p*p*a*a - 2.0*a)/a))
        dxda = 0.5*m*(p*p*a*a  - q*q)/(a*a)/np.sqrt((q*q + p*p*a*a - 2.0*a)/a)
        ret = -0.5*self._laplace_bessel(a, s, m, p, q)/a - \
          2.0*K*dxda/(np.pi*np.sqrt(a))

        return ret

    def _laplace_second_derivative_bessel(self, a, s, m, p, q):
        """Calculate second derivative of generalized Laplace coefficient using Bessel functions.

        Args:
            a (float): radius-like coordinate, must be smaller than unity
            s (float): power to which denominator is raised
            m (float): numerator of integrant is :math:`\cos(m\phi)`
            p (float): factor multiplying :math:`a^2` in denominator
            q (float): constant term in denominator

        """

        arg = np.sqrt(p*p*a + q*q/a - 2)
        K0 = sp.kn(0, m*arg)
        K1 = sp.kn(1, m*arg)
        K2 = sp.kn(2, m*arg)

        ret = (m*m*arg*(q*q - p*p*a*a)**2*K2 + arg*(m*m*(q*q - p*p*a*a)**2 + 6*a*(a*(p*p*a - 2) + q*q))*K0 - 2*m*(6*q*q*a*(p*p*a - 2) + p*p*a*a*a*(4 - 3*p*p*a) + 5*q**4)*K1)/(8*a**(7/2)*arg*(a*(p*p*a - 2) + q*q))

        return 2*ret/np.pi

    def _laplace_hyper(self, a, s, m, p, q):
        """Calculate generalized Laplace coefficient using hypergeometric functions.

        Args:
            a (float): radius-like coordinate, must be smaller than unity
            s (float): power to which denominator is raised
            m (float): numerator of integrant is :math:`\cos(m\phi)`
            p (float): factor multiplying :math:`a^2` in denominator
            q (float): constant term in denominator

        """
        ia = 1.0/(a + 1.0e-30)
        b = self._beta(a, p ,q)
        F = sp.hyp2f1(s, m + s, m + 1, b*b)

        if np.isnan(F):
            return self._laplace_brute(a, s, m, p, q)

        return 2.0*np.power(-1,m)*sp.binom(-s, m)*np.power(b, m)*np.power(b*ia, s)*F

    def _laplace_derivative_hyper(self, a, s, m, p, q):
        """Calculate derivative of generalized Laplace coefficient using hypergeometric functions.

        Args:
            a (float): radius-like coordinate, must be smaller than unity
            s (float): power to which denominator is raised
            m (float): numerator of integrant is :math:`\cos(m\phi)`
            p (float): factor multiplying :math:`a^2` in denominator
            q (float): constant term in denominator

        """
        b = self._beta(a, p ,q)
        dbda = self._dbeta(a, p, q)
        ia = 1.0/(a + 1.0e-30)
        ib = 1.0/(b + 1.0e-30)

        dF = s*(m+s)*sp.hyp2f1(s + 1, m + s + 1, m + 2, b*b)/(m + 1.0)
        F = self._laplace_hyper(a, s, m, p, q)

        if (np.isnan(F) or np.isnan(dF)):
            return self._laplace_derivative_brute(a, s, m, p, q)

        #print(m,s,ib,dbda,s,ia,F)
        return (((m+s)*ib*dbda - s*ia)*F +
                np.power(-1,m)*4.0*np.power(b, m+1)*np.power(b*ia, s)*sp.binom(-s, m)*dF*dbda)

    def _laplace_second_derivative_hyper(self, a, s, m, p, q):
        """Calculate second derivative of generalized Laplace coefficient using hypergeometric functions.

        Args:
            a (float): radius-like coordinate, must be smaller than unity
            s (float): power to which denominator is raised
            m (float): numerator of integrant is :math:`\cos(m\phi)`
            p (float): factor multiplying :math:`a^2` in denominator
            q (float): constant term in denominator

        """
        # Laplace coefficient: 2.0*np.power(-1,m)*sp.binom(-s, m)*np.power(ia, s)*np.power(b, m+s)*F(b*b)
        # Write as f(a)*g(b)
        # First derivative: f'(a)*g(b) + f(a)*g'(b)*b'(a)
        # Second derivative: f''(a)*g(b) + 2*f'(a)*g'(b)*b'(a) + f(a)*(g''(b)*b'(a)^2 + g'(b)*b''(a))
        #
        # f(a) = C*a^(-s)
        # f'(a) = -s*C*a^(-s-1)
        # f''(a) = s*(s+1)*C*a^(-s-2)
        #
        # g(b) = b^(m+s)*F(b*b)
        # g'(b) = (m+s)*b^(m+s-1)*F(b*b) + 2*b^(m+s+1)*F'(b*b)
        # g''(b) = (m+s)*(m+s-1)*b^(m+s-2)*F(b*b) + 2*(2*m+2*s+1)*b^(m+s)*F'(b*b) + 4*b^(m+s+2)*F''(b*b)

        b = self._beta(a, p ,q)
        dbda = self._dbeta(a, p, q)
        d2bda2 = self._d2beta(a, p, q)

        ia = 1.0/(a + 1.0e-30)
        ib = 1.0/(b + 1.0e-30)

        f = 2.0*np.power(-1,m)*sp.binom(-s, m)*np.power(ia, s)
        df = -s*f*ia
        d2f = -(s+1)*df*ia

        F = sp.hyp2f1(s, m + s, m + 1, b*b)
        dF = (s*(m + s)*sp.hyp2f1(s + 1, m + s + 1, m + 2, b*b))/(m + 1)
        d2F = (s*(s + 1)*(m + s)*(m + s + 1)*sp.hyp2f1(s + 2, m + s + 2, m + 3, b*b))/((m + 1)*(m + 2))

        if (np.isnan(F) or np.isnan(dF) or np.isnan(d2F)):
            return self._laplace_second_derivative_brute(a, s, m, p, q)

        g = np.power(b, m+s)*F
        dg = (m+s)*g*ib + 2*np.power(b, m+s+1)*dF
        d2g = (m+s)*(m+s-1)*g*ib*ib + 2*(2*m+2*s+1)*np.power(b,m+s)*dF + 4*np.power(b,m+s+2)*d2F

        ret = d2f*g + 2*df*dg*dbda + f*(d2g*dbda*dbda + dg*d2bda2)

        return ret

    def _beta(self, a, p, q):
        """Helper function for hypergeometric method.

        Calculate beta, which is the square root of the required argument of the hypergeometric function.

        Args:
            a (float): radius-like coordinate, must be smaller than unity
            p (float): factor multiplying :math:`a^2` in denominator
            q (float): constant term in denominator

        """
        ia = 1.0/(a + 1.0e-30)
        return 0.5*ia*(q*q + p*p*a*a -
                    np.sqrt((q*q + 2.0*a + p*p*a*a)*(q*q - 2.0*a + p*p*a*a)))

    def _dbeta(self, a, p, q):
        """Helper function for hypergeometric method.

        Calculate the derivative of beta, where beta is the square root of the required argument of the hypergeometric function.

        Args:
            a (float): radius-like coordinate, must be smaller than unity
            p (float): factor multiplying :math:`a^2` in denominator
            q (float): constant term in denominator

        """
        ia = 1.0/(a + 1.0e-30)
        sq = np.sqrt((q*q + 2.0*a + p*p*a*a)*(q*q - 2.0*a + p*p*a*a) + 1.0e-30)
        b = 0.5*ia*(q*q + p*p*a*a - sq)

        return p*p - b*ia - ((q*q + p*p*a*a)*p*p - 2.0)/sq

    def _d2beta(self, a, p, q):
        """Helper function for hypergeometric method.

        Calculate the second derivative of beta, where beta is the square root of the required argument of the hypergeometric function.

        Args:
            a (float): radius-like coordinate, must be smaller than unity
            p (float): factor multiplying :math:`a^2` in denominator
            q (float): constant term in denominator

        """
        ia = 1.0/(a + 1.0e-30)
        sq = np.sqrt(a**4*p**4 + 2*a**2*(p**2*q**2 - 2) + q**4 + 1.0e-30)
        return ia**3*(a**6*(2*p**4 - p**6*q**2) + a**4*p**4*q**2*(sq - 3*q**2) + a**2*q**2*(p**2*q**2 - 2)*(2*sq - 3*q**2) + q**6*(sq - q**2))*(a**4*p**4 + 2*a**2*(p**2*q**2 - 2) + q**4)**(-3/2)

import matplotlib.pyplot as plt

