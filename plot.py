import numpy as np
import matplotlib.pyplot as plt

class surface_density_plot():
    def __init__(self, surf_dens):
        self.surf_dens = surf_dens

    def __call__(self, ax, color_bar=True):
        r = self.surf_dens.r
        phi = self.surf_dens.phi
        Sigma = self.surf_dens.Sigma

        if np.shape(r) == np.shape(Sigma):
            # Unstructured
            temp=ax.tricontourf(phi.flatten(), r.flatten(), Sigma.flatten(), levels=100, cmap="RdBu_r")
        else:
            # Structured
            temp = plt.contourf(phi, r, Sigma, levels=100, cmap="RdBu_r")
        if color_bar is True:
            plt.colorbar(temp, ax=ax)

        return temp

class streamline_plot():
    def __init__(self, surf_dens):
        self.surf_dens = surf_dens

    def __call__(self, ax, stride=1):
        r = self.surf_dens.r
        phi = self.surf_dens.phi

        for n in range(0, len(r[:,0]), stride):
            # Accreting streamlines
            if r[n,-1] > 1.4 and r[n,0] > 0.95:
                abs_d_data = np.abs(np.diff(phi[n]))
                mask = np.hstack([ abs_d_data > np.pi, [False]])
                masked_data = np.ma.MaskedArray(r[n], mask)

                #ax.plot(phi[n], masked_data)
                ax.plot(masked_data, phi[n], color='turquoise')

        for n in range(0, len(r[:,0]), 10):
            # Accreting streamlines
            if r[n,-1] > 1.4 and r[n,0] < 0.95:
                abs_d_data = np.abs(np.diff(phi[n]))
                mask = np.hstack([ abs_d_data > np.pi, [False]])
                masked_data = np.ma.MaskedArray(r[n], mask)

                #ax.plot(phi[n], masked_data)
                ax.plot(masked_data, phi[n], color='aquamarine')


