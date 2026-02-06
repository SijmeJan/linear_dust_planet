import numpy as np
import matplotlib.pyplot as plt

def surface_density_perturbation():
    stokes = [0.01,       0.01274275, 0.01623777, 0.02069138, 0.02636651, 0.03359818,
              0.04281332, 0.05455595, 0.06951928, 0.08858668, 0.11288379, 0.14384499,
              0.18329807, 0.23357215, 0.29763514, 0.37926902, 0.48329302, 0.61584821,
              0.78475997, 1.00000000, 1.29154967, 1.66810054, 2.15443469, 2.7825594,
              3.59381366, 4.64158883, 5.9948425,  7.74263683, 10. ]
    sigma_c = [0.03265152, 0.03478893, 0.03873597, 0.04450148, 0.05020577, 0.05671094,
               0.06256518, 0.07190966, 0.08012256, 0.09187682, 0.10380042, 0.11553373,
               0.13078683, 0.14987552, 0.16937304, 0.19538407, 0.22662343, 0.2652577,
               0.31089016, 0.36342636, 0.42062674, 0.47548269, 0.52232583, 0.55635917,
               0.57557507, 0.58277646, 0.57827958, 0.56715088, 0.55451694]
    sigma_approx = [0.03095302, 0.03490798, 0.0393792,  0.04443527, 0.05015475, 0.05662825,
                    0.06396173, 0.0722815,  0.08174254, 0.09254193, 0.10494042, 0.11929423,
                    0.13609551, 0.15600733, 0.17985769, 0.20854694, 0.24288535, 0.28354647,
                    0.33136386, 0.3874629,  0.45449296, 0.52021484, 0.57075786, 0.59971554,
                    0.60956293, 0.60567846, 0.59296272, 0.57524612, 0.55536498]

    plt.plot(stokes, sigma_c, label='numerical')
    plt.plot(stokes, sigma_approx, label='approx')
    plt.plot(stokes, 0.03*np.sqrt(np.asarray(stokes)/stokes[0]), label=r'$\propto {\rm St}^{1/2}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('St')
    plt.ylabel(r'$|\Sigma\'(r_c)|$')
    plt.title('Planet mass $q=3\cdot 10^{-6}$, Softening $b/r_0=0.04$')
    plt.legend()
    plt.show()

    # one of Pablo's cases, using Roche smoothing
    stokes = [0.01,        0.01268961,  0.01610262,  0.0204336,   0.02592944,  0.03290345,
              0.04175319,  0.05298317,  0.06723358,  0.08531679,  0.10826367,  0.13738238,
              0.17433288,  0.22122163,  0.28072162,  0.35622479,  0.45203537,  0.57361525,
              0.72789538,  0.92367086,  1.1721023,   1.48735211,  1.88739182,  2.39502662,
              3.03919538,  3.85662042,  4.89390092,  6.21016942,  7.88046282,  10.        ]
    sigma_c = [0.69744946, 0.7844379,  0.88105649, 0.98800608, 1.10597442, 1.23556445, 1.37724186,
               1.53126742, 1.69759777, 1.87576948, 2.06480221, 2.26333588, 2.4701194,
               2.6839455 , 2.90631312, 3.14221441, 3.40244062, 3.70525977, 4.07281939,
               4.52354061, 5.0599107,  5.66128759, 6.28574164, 6.88180271, 7.399564,
               7.80468714, 8.07988189, 8.23240178, 8.26748891, 8.22908877]
    sigma_approx = [0.70546041,  0.79888825,  0.9054432,   1.02732515,  1.16733709,  1.32920394,
                    1.51811661,  1.74165326,  2.01129967,  2.34483464,  2.76975762,  3.32750144,
                    4.07711989,  5.09530768,  6.4670091,   8.25745042,  10.45338567, 12.8688647,
                    15.06062273, 16.40743847, 16.4991391,  15.55284042, 14.25247763, 13.13237151,
                    12.30097614, 11.64910759, 11.07401312, 10.53283253, 10.01963891,  9.54084115]
    plt.plot(stokes, sigma_c, label='numerical')
    plt.plot(stokes, sigma_approx, label='approx')
    plt.plot(stokes, 0.65*np.sqrt(np.asarray(stokes)/stokes[0]), label=r'$\propto {\rm St}^{1/2}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('St')
    plt.ylabel(r'$|\Sigma\'(r_c)|$')
    plt.title('Planet mass $q=4.53\cdot 10^{-6}$, Softening $b/r_0=r_h=0.011$')
    plt.legend()
    plt.show()

def evolving_static():
    from velocity_field import VelocityField
    from linearmoc import LinearMoc
    from main import moc_abc_interpolate
    from accretion import calculate_accretion_efficiency
    from plot import streamline_plot

    # Create velocity field from linear equations
    vel_field = VelocityField.from_fargo('/Users/sjp/Downloads/Archive/static/fargo3d/outputs/lui_ormel/', 20)
    # Method of Characteristics object
    moc = LinearMoc.from_single(lambda x,y: moc_abc_interpolate(x, y, vel_field, min_dist=0.0))

    # Calculate accretion efficiency from streamlines
    eff, r, phi = calculate_accretion_efficiency(moc, None, from_streamlines=True, N=512, r_start=1.4)
    print('Efficiency: ', eff)

    fig, (ax1, ax2) = plt.subplots(1, 2)# , sharey=True, sharex=True)
    centre = [1.0,0.0]
    size = 0.05
    streamline_plot(None)(ax1, stride=1, r=r, phi=phi)
    ax1.set_xlim([centre[0] - size, centre[0] + size])
    ax1.set_ylim([centre[1] - size, centre[1] + size])
    ax1.set_xlabel(r'$r$')
    ax1.set_ylabel(r'$\varphi$')
    ax1.set_title('Static gas')
    ax1.set_aspect('equal', 'box')

    vel_field = VelocityField.from_fargo('/Users/sjp/Downloads/Archive/evolving/fargo3d/outputs/lui_ormel/', 20)
    # Method of Characteristics object
    moc = LinearMoc.from_single(lambda x,y: moc_abc_interpolate(x, y, vel_field, min_dist=0.0))
    # Calculate accretion efficiency from streamlines
    eff, r, phi = calculate_accretion_efficiency(moc, None, from_streamlines=True, N=1024, r_start=1.4)
    print('Efficiency: ', eff)

    streamline_plot(None)(ax2, stride=1, r=r, phi=phi)
    ax2.set_xlim([centre[0] - size, centre[0] + size])
    ax2.set_ylim([centre[1] - size, centre[1] + size])
    ax2.set_xlabel(r'$r$')
    ax2.set_ylabel(r'$\varphi$')
    ax2.set_title('Evolving gas')
    ax2.set_aspect('equal', 'box')

    #vel_field = VelocityField.from_fargo('/Users/sjp/Downloads/Archive/evolving/fargo3d/outputs/lui_ormel/', 20, n_fluid=0)
    # Method of Characteristics object
    #moc = LinearMoc.from_single(lambda x,y: moc_abc_interpolate(x, y, vel_field, min_dist=0.0))

    #sigma = SurfaceDensity.from_moc(param_dict, moc, r_start=0.8, N=256, r_circle=0.05, dt=0.02*np.pi)

    #streamline_plot(None)(ax2, r=sigma.r[238:239,1000:5000], phi=sigma.phi[238:239,1000:5000]+0.02, colors=['black','black'])
    #ax2.annotate("", xytext=(0.99567, 0.0395), xy=(0.99485, 0.045), arrowprops=dict(arrowstyle="->"))
    #ax2.annotate("", xytext=(1.00183, 0.03734), xy=(1.00157, 0.03442), arrowprops=dict(arrowstyle="->"))

    import matplotlib.patches as mpatches

    style = "Simple, tail_width=0.5, head_width=4, head_length=8"
    kw = dict(arrowstyle=style, color="k", zorder=2)

    rc = 0.999
    w = 0.015
    a3 = mpatches.FancyArrowPatch((rc, 0.022), (rc-w, 0.045), connectionstyle="angle3,angleA=0, angleB=90", **kw)
    ax2.add_patch(a3)
    a4 = mpatches.FancyArrowPatch((rc+w, 0.045), (rc, 0.022), connectionstyle="angle3,angleA=90, angleB=0", **kw)
    ax2.add_patch(a4)
    a3 = mpatches.FancyArrowPatch((rc, -0.022), (rc+w, -0.045), connectionstyle="angle3,angleA=0, angleB=90", **kw)
    ax2.add_patch(a3)
    a4 = mpatches.FancyArrowPatch((rc-w, -0.045), (rc, -0.022), connectionstyle="angle3,angleA=90, angleB=0", **kw)
    ax2.add_patch(a4)

    phic = np.linspace(0,2*np.pi,200)
    rc = 0.02*np.ones_like(phic)
    xc = rc*np.cos(phic) + 1.0
    yc = rc*np.sin(phic)
    ax1.plot(xc, yc, color='black', linestyle='dashdot')
    ax2.plot(xc, yc, color='black', linestyle='dashdot')
    #ax3.plot(xc, yc, color='black', linestyle='dashdot')

    #from fargo import read_fargo
    #r, phi, u, v = read_fargo('/Users/sjp/Downloads/Archive/evolving/fargo3d/outputs/lui_ormel/', 20, n_fluid=0)
    #r0, phi0, u0, v0 = read_fargo('/Users/sjp/Downloads/Archive/static/fargo3d/outputs/lui_ormel/', 20, n_fluid=0)
    #rr,phiphi = np.meshgrid(r, phi)
    #v0 = v0 - rr + 1/np.sqrt(rr)
    #Q = ax2.quiver(r, phi, u-u0, (v-v0)/rr, scale=1.0e-1, units='width')


#surface_density_perturbation()
evolving_static()