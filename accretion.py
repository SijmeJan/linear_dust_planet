import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from surface_density.surface_density import calc_surface_density_along_streamlines

def calculate_accretion_efficiency(moc, param_dict, from_streamlines=False, N=256, r_start=1.2):
    if from_streamlines is True:
        # Calculate fraction of streamlines that end up on the planet
        r, phi, Sigma = calc_surface_density_along_streamlines(moc, param_dict, N, r_start, phi_start=None, u_start=0*r_start, reverse_flag=False, dt=0.02*np.pi)
        r_final = r[:,-1]

        #for n in range(0, len(r[:,0])):
        #    abs_d_data = np.abs(np.diff(phi[n]))
        #    mask = np.hstack([ abs_d_data > np.pi, [False]])
        #    masked_data = np.ma.MaskedArray(r[n], mask)

        #    plt.plot(phi[n], masked_data)
        #plt.show()

        return len(np.asarray(np.abs(r_final-1)<0.02).nonzero()[0])/N, r, phi

    # Calculate mass flux inside planet orbit
    phi, w = sp.special.roots_legendre(N)
    phi_start = phi*np.pi
    w = w*np.pi

    r_start = 0.9
    #u_in, v_in = calc_2D_sum(r_start*np.ones(len(phi_start)), phi_start, max_m, param_dict, grid_output=False)
    u_in, v_in, src_in = moc.abc(r_start*np.ones(len(phi_start)), phi_start)

    r, phi, Sigma = calc_surface_density_along_streamlines(moc, param_dict, N, r_start, phi_start=phi_start, u_start=0*r_start, reverse_flag=True)

    # Unperturbed pebble flux
    pebble_flux = -4*np.pi*param_dict['taus']*param_dict['eta']/(1+param_dict['taus']**2)
    acc_eff = 1 - np.sum(w*Sigma[:,0]*np.real(u_in)*r_start)/pebble_flux

    return acc_eff
