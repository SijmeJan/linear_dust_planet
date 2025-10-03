##################################
# PLOT PREVIOUS TORQUE DENSITIES #
##################################

#trq = np.load('tq_q3-5_tau1-1_b1-3_cut.npz')
#plt.plot(trq['r'], trq['torque_density'])
#print(trq['torque'])
#trq = np.load('tq_q3-5_tau2-1_b1-3_cut.npz')
#plt.plot(trq['r'], trq['torque_density'])
#print(trq['torque'])
#trq = np.load('tq_q3-5_tau4-1_b1-3_cut.npz')
#plt.plot(trq['r'], trq['torque_density'])
#print(trq['torque'])

#trq = np.load('tq_q15-6_tau1-2_b1-3_cut_1-1_eta18-3.npz')
#plt.plot((trq['r']-1)*(0.03/0.1)**(-1/3), trq['torque_density'])
#print(trq['torque'])

#trq = np.load('tq_q15-6_tau1-2_b1-3_cut_1-1_eta18-3.npz')
#plt.plot(trq['r']-1, trq['torque_density'])
#print(trq['torque'])

#trq = np.load('tq_q3-5_tau2-1_b1-3_cut_1-1_eta18-3.npz')
#plt.plot((trq['r']-1)*(0.1/0.1)**(-1/3), trq['torque_density'])
#print(trq['torque'])

#trq = np.load('tq_q3-6_tau1-1_b1-3_cut_1-1_eta18-3.npz')
#plt.plot((trq['r']-1)*(0.1)**(-1/3), trq['torque_density'])
#print(trq['torque'])

#r = np.linspace(0.9,1.1,10000)
#trq_approx = tq.calc_torque_density_approx(r, param_dict)
#plt.plot(r-1, trq_approx)

#print(sp.integrate.quad(lambda x: tq.calc_torque_density_approx(x, param_dict), 0.9, 1.1, limit=100))
#plt.show()
#exit()

#print(np.shape(trq['r']), np.shape(trq['phi']))
#temp = plt.contourf(trq['r'], trq['phi'], np.transpose(trq['torque_2D']), levels=100, cmap="RdBu_r")
#plt.colorbar(temp)
#plt.show()
#exit()

#################################
# PLOT CHRENKO AND GUILERA LAWS #
#################################

tau = np.logspace(-2,0,1000)
#q = np.logspace(-7,-4,100)
res=np.zeros(len(tau))
res_approx = np.zeros(len(tau))
res_square = np.zeros(len(tau))

mp = [3.0e-7, 3.0e-6, 1.5e-5, 3.0e-5]

for j in range(0, len(mp)):
    for i in range(0, len(tau)):
        res[i] = tq.chrenko_torque(tau[i], mp[j])

        #param_dict['q'] = mp[j]
        #param_dict['taus'] = tau[i]

        #res_int = sp.integrate.quad(lambda x: tq.calc_torque_density_approx(x, param_dict), 0.9, 1.1, limit=1000)
        #res_approx[i] = param_dict['q']*res_int[0]

        #res_square[i] = param_dict['q']*tq.calc_torque_approx(param_dict)

    #plt.plot(tau, res, color=colors[j])
    #plt.plot(tau, res_approx, color=colors[j], linestyle='dashed')
    #plt.plot(tau, res_square, color=colors[j], linestyle='dotted')

#plt.scatter([0.01,0.1,0.2,0.4], np.asarray([0.56,0.77,0.91,1.11])*3e-6, marker='o')
#plt.scatter([0.01,0.1,0.2,0.4], np.asarray([0.42,0.72,0.87,1.13])*1.5e-5, marker='o')
#plt.scatter([0.03,0.1,0.2,0.4], np.asarray([0.48,0.70,0.86,1.23])*3e-5, marker='o')

# New cutoff 0.1
#plt.scatter([0.1, 0.2, 0.4], np.asarray([0.77, 0.81, 0.86])*3e-7, marker='o')
#plt.scatter([0.01, 0.1, 0.2, 0.4], np.asarray([0.88, 0.97, 0.95, 1.12])*3e-6, marker='o')
#plt.scatter([0.1, 0.2, 0.4], np.asarray([0.88, 1.00, 1.14])*1.5e-5, marker='o')
#plt.scatter([0.03, 0.1, 0.2, 0.4], np.asarray([0.73, 0.85, 0.95, 1.28])*3e-5, marker='o')

#plt.plot(tau, 3.0e-6*tau**(1/3)*2, linestyle='dashed')

#res=np.zeros(len(tau))
#for i in range(0, len(tau)):
#    res[i] = tq.guilera_torque(tau[i], 3.0e-5)
#plt.plot(tau, res)

#plt.xscale('log')
#plt.yscale('log')
#plt.ylim([1.0e-7,1.0e-5])
#plt.xlabel(r'${\rm St}$')
#plt.ylabel(r'$\Gamma/\Gamma_0$')
#plt.show()

#exit()

############################
# CALCULATE TORQUE DENSITY #
############################

lin_vel = LinearVelocity(param_dict)
vel_field = VelocityField.from_linear(lin_vel, 2000, approx=True)
#vel_field = VelocityField.from_fargo('/Users/sjp/Downloads/snapshot_10M_e_St_01/', 131)
moc = LinearMoc.from_single(lambda x,y: moc_abc_interpolate(x, y, vel_field, min_dist=0.0))

sigma = lambda x,y: sd.calc_surface_density_moc(x, y, moc, param_dict)
torque_dens = lambda x: tq.calculate_torque_density(x, sigma, param_dict, N=320)

#torque_dens(np.asarray([0.9980035245805788]))

#res = sigma(0.9980035245805788, 0.01)
#print(res, np.shape(res))
#exit()

#torque, r, phi, tq_dens, res = tq.calculate_total_torque(torque_dens, param_dict, Nbase=40)
#np.savez('tq_q15-6_tau1-2_b1-3_cut_1-1_eta18-3.npz', r=r, phi=phi, torque_density=tq_dens, torque_2D=res, torque=np.asarray([torque]))
#print(torque)

#plt.plot(r, tq_dens)
#plt.show()

#exit()

#############################
# CALCULATE SURFACE DENSITY #
#############################

print('HALLO')
surf_dens = sd.SurfaceDensity.from_moc(param_dict, moc)
#surf_dens.save_to_file('sd_q3-7_tau1-2_cut_1-1_b1-3_eta18-3.npz')
#surf_dens = sd.SurfaceDensity.from_file('sd_q3-5_tau1-1_cut_1-1_b1-3_eta18-3.npz')

#phi = np.linspace(-np.pi, np.pi, 2000)
#r = 0.9980035245805788*np.ones(len(phi))
#Sigma = surf_dens(r, phi)

#plt.plot(phi, Sigma)

#plt.plot(surf_dens.phi[:,0], surf_dens.Sigma[:,0])
#plt.show()
#exit()

print('DOEI')

surface_density_plot(surf_dens)(plt.gca())

r = np.linspace(0.9,1.1,1000)

r_hill = (param_dict['q']/3)**(1/3)
racc = (param_dict['taus']*param_dict['q'])**(1/3)

u0 = -2*param_dict['eta']*param_dict['taus']

s_star = racc
s_cut = 0.5*r_hill

s = r - 1.0

phi_dagger = np.where(np.abs(s) < s_cut, np.acos(0.5*((1+s)**2+1-s_cut**2)/(1+s)), np.zeros(len(s)))

phi = -phi_dagger - (0.75*(s*s - s_star*s_star) + param_dict['eta']*(s-s_star))/u0

pot1 = -1/np.sqrt(r*r + 1 - 2*r*np.cos(phi) + param_dict['soft']**2)
pot2 = -1/np.sqrt(r*r + 1 - 2*r*np.cos(phi_dagger) + param_dict['soft']**2)

phi_arm = np.acos(1 - 0.5*s_cut**2) - (0.75*s*s + param_dict['eta']*s)/u0
pot3 = -1/np.sqrt(r*r + 1 - 2*r*np.cos(phi_arm) + param_dict['soft']**2)

plt.plot(phi*np.heaviside(s_star - s, 0), s+1)
plt.plot(phi_dagger, s+1)
plt.plot(phi_arm*np.heaviside(-s, 0), s+1)

#r = np.linspace(0.9,1.1,1000)
#phi = np.linspace(-np.pi,np.pi,1000)
#surf_dens = sd.SurfaceDensity.from_approx(param_dict, r, phi)

#temp = plt.contourf(surf_dens.phi, surf_dens.r, surf_dens.Sigma, levels=100, cmap="RdBu_r")
plt.ylim([0.9,1.1])
plt.xlim([-np.pi, np.pi])

#plt.colorbar(temp)

plt.show()

exit()





surf_dens = sd.SurfaceDensity.from_moc(param_dict, moc, N=100)




#print(calculate_accretion_efficiency(moc, param_dict, from_streamlines=True))
#print(epsilon(mode='2dset', times_eta=False, qp=param_dict['q'], tau=param_dict['taus'], eta=param_dict['eta']))

#r, phi, Sigma = surface_density(moc, param_dict, r_start=0.9, N=2048, dt=0.2*np.pi*0.1/param_dict['taus'], r_finish=1.5)

# Single streamline
#r_start = 0.9*np.ones(1000)
#p_start = np.linspace(-np.pi, np.pi, 1000)

#r, phi, Sigma = sd.surface_density(moc, param_dict, N=512, r_start=0.9, dt=0.2*np.pi, r_finish=None)
#r, phi, Sigma = sd.calc_surface_density_field(moc, param_dict, len(r_start), np.atleast_1d(r_start), phi_start=np.atleast_1d(p_start), reverse_flag=True, r_finish=1.5, dt=0.2*np.pi)

#r = surf_dens.r
#phi = surf_dens.phi

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.set_figwidth(15)

streamline_plot(surf_dens)(ax1)
ax1.set_ylim([0.9,1.1])

plt.show()

exit()
