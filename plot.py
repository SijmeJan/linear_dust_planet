import numpy as np
import matplotlib.pyplot as plt

class surface_density_plot():
    def __init__(self, surf_dens):
        self.surf_dens = surf_dens

    def __call__(self, ax, color_bar=True):
        r = self.surf_dens.r
        phi = self.surf_dens.phi
        Sigma = np.real(self.surf_dens.Sigma)

        if np.shape(r) == np.shape(Sigma):
            # Unstructured
            temp=ax.tricontourf(phi.flatten(), r.flatten(), Sigma.flatten()*np.sqrt(r.flatten()), levels=100, cmap="RdBu_r")
        else:
            # Structured
            for i in range(0, len(phi)):
                Sigma[:,i] = Sigma[:,i]*np.sqrt(r)
            temp = plt.contourf(phi, r, Sigma, levels=100, cmap="RdBu_r")
        if color_bar is True:
            plt.colorbar(temp, ax=ax)

        return temp

class streamline_plot():
    def __init__(self, surf_dens):
        self.surf_dens = surf_dens

    def __call__(self, ax, stride=1, r = None, phi=None, colors=['turquoise', 'aquamarine']):
        if r is None:
            r = self.surf_dens.r
        if phi is None:
            phi = self.surf_dens.phi

        for n in range(0, len(r[:,0]), stride):
            # Accreting streamlines
                #if r[n,-1] > 1.4 and r[n,0] > 0.95:
            if r[n, -1] >= 0.95:
                abs_d_data = np.abs(np.diff(phi[n]))
                mask = np.hstack([ abs_d_data > np.pi, [False]])
                masked_data = np.ma.MaskedArray(r[n], mask)

                #ax.plot(phi[n], masked_data)
                ax.plot(masked_data, phi[n], color=colors[0])

        for n in range(0, len(r[:,0]), 1):
            # Accreting streamlines
                #if r[n,-1] > 1.4 and r[n,0] < 0.95:
            if r[n, -1] < 0.95:
                abs_d_data = np.abs(np.diff(phi[n]))
                mask = np.hstack([ abs_d_data > np.pi, [False]])
                masked_data = np.ma.MaskedArray(r[n], mask)
                ax.plot(masked_data, phi[n], color=colors[1])


