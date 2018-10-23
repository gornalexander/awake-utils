from LCODEplot import *
from shutil import copyfile
from IPython.display import Markdown, display
def printmd(string):
    display(Markdown(string))

config = """
# Simulation area:
geometry = c
window-width = 10;		r-step = 0.01
window-length = 1500;		xi-step = 0.01
time-limit = 50000.1;		time-step = 200

# Particle beams:
beam-current = 0.00281
rigid-beam = n
beam-substepping-energy = 810000
beam-particles-in-layer = 160
beam-profile = \"\"\"
xishape=b, ampl=1.0, length=1500, rshape=g, radius=1.0, angshape=l, angspread=4.5e-5, energy=7.83e5, m/q=1836, eshape=g, espread=2740.0
\"\"\"

# Plasma:
plasma-model = P # Plasma model (fluid/particles/newparticles, f/p/P)
plasma-particles-number = 5000
plasma-profile = 2 # Initial profile (uniform/stepwise/gaussian/arbitrary/channel)
plasma-width = 2
plasma-temperature = 0
ion-model = Y # Model of plasma ions (mobile/background/absent/equilibrium, Y/y/n/N)
ion-mass = 157000
substepping-depth = 3
substepping-sensivity = 0.2

# Every-time-step diagnostics:
indication-line-format = 1 # On-screen indication line format (eacht/eachxi)
output-Ez-minmax = y;		output-Ez-local = y
output-Phi-minmax = y;		output-Phi-local = y
output-lost-particles = y

# Periodical diagnostics:
output-time-period = 5000

#  Colored maps: (Er,Ef,Ez,Phi,Bf,Bz,pr,pf,pz,pri,pfi,pzi
#                 nb,ne,ni,Wf,dW,SEB,Sf,Sf2,Sr,Sr2,dS,dS2):
colormaps-full = ""
colormaps-subwindow = ""
colormaps-type = F
drawn-portion = 1 # Drawn portion of the simulation window
subwindow-xi-from = -0;		subwindow-xi-to = -100
subwindow-r-from = 0;		subwindow-r-to = 10
output-reference-energy = 1000
output-merging-r = 2;		output-merging-z = 50
output-skipping-r = 20;          output-skipping-xi = 20
palette = d # Colormaps palette (default/greyscale/hue/bluewhitered, d/g/h/b)
                E-step = 0.059;	               nb-step = 0.00056
              Phi-step = 0.059;	               ne-step = 0.1
               Bf-step = 0.059;	               ni-step = 0.01
               Bz-step = 0.059;	             flux-step = 0.02
 electron-momenta-step = 0.1;	 r-corrected-flux-step = 0.02
      ion-momenta-step = 0.1;	           energy-step = 10

#  Output of various quantities as functions of xi:
#   (ne,nb,Ez,<Ez>,Bz,Phi,pz,emitt,dW,Wf,ni,pzi)
#   (nb2,Er,Ez2,Bf,Bz2,Fr,pr,pf,<rb>,dS,Sf,SEB,pri,pfi,Ef)
f(xi) = Ez,Phi,emitt,nb2,Ez2,<rb>,dS,Sf,SEB
f(xi)-type = n
axis-radius = 0;		auxillary-radius = 1
               E-scale = 0.59;	              nb-scale = 0.02
             Phi-scale = 0.59;	              ne-scale = 2
              Bz-scale = 0.59;	              ni-scale = 0.1
electron-momenta-scale = 0.5;	            flux-scale = 0.5
     ion-momenta-scale = 0.5;	          energy-scale = 1
     beam-radius-scale = 5;	       emittance-scale = 300

#  Beam particle information as pictures (r,pr,pz,M):
output-beam-particles = ""
draw-each = 1
beam-picture-height = 900
beam-pr-scale = 1000
beam-a-m-scale = 1000;		beam-pz-scale = 15000

# Output of beam characteristics in histogram form (r,z,M,a):
histogram-output = ""
histogram-output-accel = ""
histogram-type = y
histogram-bins = 300;		beam-angle-scale = 0.02

#  Trajectories of plasma particles:
trajectories-draw = n
trajectories-each = 50;	trajectories-spacing = 100
trajectories-min-energy = 1;	trajectories-energy-step = 0.5

# Saving run state periodically:
saving-period = 50000
save-beam = n
save-plasma = n
"""

### Making directory module

def makedir(path, name, cfg=config): 
    path = os.path.join(path, name)
    print('Making directory... ', end='')
    try:
        os.makedirs(path)
        print('done.')
    except:
        print('seems like the directory already exists.')
    
    with open(os.path.join(path, 'lcode.cfg'), "w") as text_file:
        print('Writing lcode.cfg... ', end='')
        text_file.write(cfg)
    print('done')
    return path

### Simulation setting up module

def setup_simulation(lcode, time_step, window_width, window_length):
    lcode.set_parameter('time-step', time_step)
    print(type(window_width))
    window_width = round(window_width, 2)
    lcode.set_parameter('window-width', window_width)
    window_length = float(window_length)
    lcode.set_parameter('window-length', window_length)
    print('')

### Making plasma module

def plasma_zshape(x, n, form='L'):
    zshape = ''
    dxs = x[1:] - x[:-1]
    arr = np.array([dxs, n[:-1], n[1:]]).T
    zshape += 'plasma-zshape = """\n'
    for dx, n_start, n_end in arr:
        zshape += '%f %f %s %f\n' % (dx, n_start, form, n_end)
    zshape +='"""'
    return zshape

def make_plasma(lcode, n_plasma, plasma_length, plasma_radius, n_profile):
    print("# PLASMA")
    plasma_length = round(plasma_length)
    lcode.set_parameter('time-limit', plasma_length + 0.1)

    plasma_radius = round(plasma_radius, 2)
    lcode.set_parameter('plasma-width', plasma_radius)

    # z-shape
    n0 = n_plasma
    z = np.arange(0, plasma_length, lcode.get_parameter('time-step'))
    nz = np.array([n0*n_profile(x) for x in z])
    plt.title('density profile')
    plt.plot(z*lcode.cwp/100, lcode.n*nz)
    plt.xlabel(r'$z$ [m]')
    plt.ylabel(r'$n$ [cm-3]')
    print('Writing pzshape.txt... ', end='')
    with open(os.path.join(lcode.path, 'pzshape.txt'), 'w') as f:
        f.write(plasma_zshape(z, nz))
    print('done\n')

### Beamfile construction module

def Q2I(Q, sigma_z): # SI (return Amps)
    return c/100. * Q / np.sqrt(2*np.pi) / sigma_z
def make_beam(lcode, sigma_z, pos_xi, sigma_r, gamma, enspread, emittance, N_particles):
    print("# BEAM")
    q_m = m_MeV/M_MeV
    # xi
    xi_shape = Gauss(pos_xi, sigma_z)
    # r
    r_shape = rGauss(sigma_r)
    # pz
    pz_shape = Gauss(gamma, enspread)
    # angle
    angspread = emittance / (gamma / (M_MeV/m_MeV) * sigma_r)
    #print('%e' % angspread)
    ang_shape = Gauss(0, angspread)
    # current
    Ipeak_A = Q2I(N_particles*1.6e-19, sigma_z*lcode.cwp/100)
    Ipeak_kA = Ipeak_A/1000
    lcode.beam = lcode.make_beam(xi_shape, r_shape, pz_shape, ang_shape, Ipeak_kA, q_m, saveto=lcode.path)
    print('Writing beamfile.bit...', end='')
    with open(os.path.join(lcode.path, 'beamfile.bit'), 'w') as f:
        f.write('0.0')
    print('done.\n')

### Making task module

def make_task(lcode, name, N_nodes, cluster='matrosov'):
    if cluster == 'matrosov':
        task = """#!/bin/bash
#
#PBS -N {}
#PBS -l nodes={}:intel:ppn=36,pvmem=30000mb,walltime=100:00:00
cd $PBS_O_WORKDIR
/share/apps/bin/mpiexec -perhost 36 lcode lcode.cfg pzshape.txt""".format(name, N_nodes)
        print('Making task.sh... ', end='')
        with open(os.path.join(lcode.path, 'task.sh'), 'w') as f:
            f.write(task)
        print('done.\n')
    elif cluster == 'matrosov2':
        task = """#!/bin/bash
#
#PBS -N {}
#PBS -l nodes={}:amd:ppn=32,pvmem=30000mb,walltime=100:00:00
cd $PBS_O_WORKDIR
/share/apps/bin/mpiexec -perhost 32 lcode2 lcode.cfg pzshape.txt""".format(name, N_nodes)
        print('Making task.sh... ', end='')
        with open(os.path.join(lcode.path, 'task.sh'), 'w') as f:
            f.write(task)
        print('done.\n')
    elif cluster == 'condor':
        print('Making condor directory... ', end='')
        try:
            os.makedirs(os.path.join(lcode.path, 'condor'))
            print('done.')
        except:
            print('seems like the directory already exists.')
        print('Copying condor files... ', end='')
        copyfile('../condor/job.py', os.path.join(lcode.path, 'condor/job.py'))
        copyfile('../condor/job_start.sh', os.path.join(lcode.path, 'condor/job_%s.sh' % name))
        copyfile('../condor/job.submit', os.path.join(lcode.path, 'condor/job.submit'))
        with open('../condor/job.sub') as f:
            lines = f.readlines()
        lines[0] = 'executable            = job_%s.sh\n' % name
        with open(os.path.join(lcode.path, 'condor/job.sub'), "w") as f:
            f.writelines(lines)
        print('done.')