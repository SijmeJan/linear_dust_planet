import numpy as np

def read_fargo(direc, number, n_fluid=1):
    ngh=3
    phi = np.loadtxt(direc + '/domain_x.dat')
    r = np.loadtxt(direc + '/domain_y.dat') [ngh:-ngh]

    # Cell centres
    phi = 0.5*(phi + np.roll(phi, 1))[1:]
    r = 0.5*(r + np.roll(r, 1))[1:]

    ngh=3
    nphi = np.max([1,len(np.loadtxt(direc+'/domain_x.dat'))-1])
    nr = np.max([1,len(np.loadtxt(direc+'/domain_y.dat')[ngh:-ngh])-1])

    print('Creating from FARGO snapshot with Nr = ', nr, ', Nphi = ', nphi)

    read = lambda f:np.transpose(np.fromfile(f).reshape(1,nr,nphi),[2,1,0])

    filename = direc + '/dust{}vx{}.dat'.format(n_fluid, number)
    if n_fluid == 0:
        filename = direc + '/gasvx{}.dat'.format(number)

    v = read(filename)[:,:,0]

    # Velocity perturbation over Keplerian
    for j in range(0, nphi):
        v[j,:] = v[j,:] + r  - 1/np.sqrt(r)

    filename= direc + '/dust{}vy{}.dat'.format(n_fluid, number)
    if n_fluid == 0:
        filename = direc + '/gasvy{}.dat'.format(number)
    u = read(filename)[:,:,0]

    #filename= direc + '/dust{}dens{}.dat'.format(n_fluid, number)
    #if n_fluid == 0:
    #    filename = direc + '/gasdens{}.dat'.format(number)
    #dens = read(filename)[:,:,0]

    #filename = direc + '/gasdens{}.dat'.format(number)
    #gasdens = read(filename)[:,:,0]
    #D = 0.001*0.05*0.05

    #if n_fluid == 0:
    #    import matplotlib.pyplot as plt
    #    #print(r[170])
    #    #plt.plot(phi, np.sqrt(v[:,170]**2 + u[:,170]**2))
    #    #plt.plot(phi, u[:,170])
    #    #plt.plot(phi, v[:,170])
    #    plt.plot(r, np.mean(v, axis=0))
    #    plt.show()

    return r, phi, u, v

