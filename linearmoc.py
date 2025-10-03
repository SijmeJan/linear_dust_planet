import numpy as np
import scipy as sp

class LinearMoc():
    '''Analyse problem of the form a(x,y)*du/dx + b(x,y)*du/dy = c(x,y)'''
    def __init__(self, a, b, c):
        self.abc = lambda x,y: (a(x,y), b(x,y), c(x,y))

    @classmethod
    def from_single(cls, abc):
        ret = cls(None, None, None)
        ret.abc = abc
        return ret

    def rhs(self, t, Q, N, reverse_flag=False):
        # ode dQ/dt = rhs(t, Q, N)

        # we are following N points, First N entries in Q are the current y coordinates,
        # followed by N y coordinates, followed by N values of u.

        x = Q[0:N]
        y = Q[N:2*N]

        a, b, c = self.abc(x, y)

        ret = 0.0*Q
        ret[0:N] = a
        ret[N:2*N] = b
        ret[2*N:3*N] = c

        if reverse_flag is True:
            return -1.0*ret

        return ret

    def calculate(self, start_x, start_y, start_u, dt=0.1, max_steps=10, method='RK45', reverse_flag=False):
        # dx/dt = a; dy/dt = b; du/dt = c

        # Number of points to follow
        N = len(start_x)

        # Initial conditions
        Q0 = np.zeros(3*N, dtype=np.float64)
        Q0[0:N] = start_x
        Q0[N:2*N] = start_y
        Q0[2*N:3*N] = start_u

        t_eval = dt*np.arange(max_steps)
        sol = sp.integrate.solve_ivp(self.rhs, [0, max_steps*dt], Q0, t_eval=t_eval, args=(N,reverse_flag), method=method, rtol=1.0e-5, atol=1.0e-8)

        x = sol.y[0:N]
        y = sol.y[N:2*N]
        u = sol.y[2*N:3*N]

        return x, y, u
