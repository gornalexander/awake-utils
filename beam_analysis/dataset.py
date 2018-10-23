import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from .processing import load_radii, update_meta, Reader
from .geometry import fit_ellipse, fit_circle, shift_radius
from .geometry import ellipse, polar_to_cart


NAMES = {
    # radii data
    'mean':        (r'mean diameter $\mathrm{[mm]}$',
                    lambda dt, i: dt['mean%s'%i]),
    'var':         (r'diameter variance $\mathrm{[mm^2]}$',
                    lambda dt, i: dt['var%s'%i]),
    'max':         (r'maximal diameter $\mathrm{[mm]}$',
                    lambda dt, i: dt['max%s'%i]),
    'min':         (r'minimal diameter $\mathrm{[mm]}$',
                    lambda dt, i: dt['min%s'%i]),
    'rel_range':   (r'relative diameter range',
                    lambda dt, i: (dt['max%s'%i] - dt['min%s'%i]) /
                                   dt['mean%s'%i]),
    'rel_std':     (r'relative diameter standard deviation $\sigma / \mu$',
                    lambda dt, i: np.sqrt(dt['var%s' % i]) / dt['mean%s'%i]),

    'r_mean':      (r'mean radius $\mathrm{[mm]}$',
                    lambda dt, i: dt['r_mean%s'%i]),
    'r_var':       (r'radius variance $\mathrm{[mm^2]}$',
                    lambda dt, i: dt['r_var%s'%i]),
    'r_max':       (r'maximal radius $\mathrm{[mm]}$',
                    lambda dt, i: dt['r_max%s' % i]),
    'r_min':       (r'minimal radius $\mathrm{[mm]}$',
                    lambda dt, i: dt['r_min%s' % i]),
    'r_rel_range': ('relative radius range',
                    lambda dt, i: (dt['r_max%s' % i] - dt['r_min%s' % i]) /
                                   dt['r_mean%s' % i]),
    'r_rel_std':   (r'relative diameter standard deviation $\sigma / \mu$',
                    lambda dt, i: np.sqrt(dt['var%s' % i]) / dt['mean%s' % i]),

    # fixed meta data
    'laser_energy_1': (r'laser energy (1) $\mathrm{[mJ]}$', None),
    'laser_energy_2': (r'laser energy (2) $\mathrm{[mJ]}$', None),
    'proton_count': (r'proton count $\mathrm{[10^{10}]}$', None),
    'bunch_len': (r'1-$\sigma$ bunch length $\mathrm{[ns]}$', None),
    'laser_bunch_offset': (r'bunch to laser offset $\mathrm{[ps]}$', None),
    'density_ups': (r'upstream vapor density', None),
    'density_downs': (r'downstream vapor density', None),
    'density_grad': (r'vapor density gradient', None),
    'timing': (r'timing $\mathrm{[ns]}$',
               lambda dt, i: (dt['laser_bunch_offset'] -
                              dt['laser_bunch_offset_fine'])),
    'density_mean': (r'mean vapor density',
                     lambda dt, i: (dt['density_ups'] + dt['density_downs'])/2),

    # laser alignment
    'diff_in': (r'bunch - laser offset (upstream) $\mathrm{[mm]}$', None),
    'diff_out': (r'bunch - laser offset(downstream) $\mathrm{[mm]}$', None),
    'laser_tilt': (r'laser $|x_{out} - x_{in}|$ $\mathrm{[mm]}$', None),
    'misa_shift': (r'$|\Delta x_{out} + \Delta x_{in}|$ $\mathrm{[mm]}$', None),
    'misa_tilt': (r'$|\Delta x_{out} - \Delta x_{in}|$ $\mathrm{[mm]}$', None),
}

IMG_HTML = """
<div style="width: 100%">
    <h4>{0}</h4>
    <div style="width: 50%; display:inline-block; float:left;">
        <h4 style="display: inline;">1</h4>
        <img style="display: inline; max-width: 98%;" src="{1}"></img>
    </div>
    <div style="width: 50%; display:inline-block; float:right;">
        <h4 style="display: inline;">2</h4>
        <img style="display: inline; max-width: 98%;" src="{2}"></img>
    </div>
</div>"""


def _get_ax(fig=None, ax=None, *figa, **figkwa):
    if ax is None:
        if fig is None:
            plt.figure(*figa, **figkwa)
            return plt.gca()
        return fig.gca()
    return ax


class MeasureImgs(object):
    def __init__(self, names, titles=None, out='./out_bigstat/'):
        self.out = out
        if isinstance(names, str):
            self.names = [names]
        else:
            self.names = names
        if titles is None:
            self.titles = [n.replace('.h5', '') for n in self.names]
        elif isinstance(titles, str):
            self.titles = [titles]
        else:
            self.titles = titles

    def _repr_html_(self):
        return '\n'.join(IMG_HTML.format(
            title,
            os.path.join(self.out, 'plots/1/' + name.replace('.h5', '.png')),
            os.path.join(self.out, 'plots/2/' + name.replace('.h5', '.png')))
                         for name, title in zip(self.names, self.titles))


def adjust_elliptic(radii, un_ellipse=False, plot=True):
    angles = np.linspace(0, 1, len(radii[0]))
    mean_rad = np.mean(radii, axis=0)
    r0: float = np.mean(mean_rad)

    (r, x0, y0), _ = fit_circle(angles, mean_rad, r0)
    angles, mean_rad_s = shift_radius(angles, mean_rad, -x0, -y0)
    (a, b, c), _ = fit_ellipse(angles, mean_rad_s)

    if plot:
        plt.figure(figsize=(15, 4))
        plt.subplot2grid((1, 3), (0, 0), colspan=2)
        plt.plot(angles * 360, mean_rad, ':', label='raw')
        plt.plot(angles * 360, mean_rad_s, '-', label='shifted')
        plt.plot(angles * 360, ellipse(angles, a, b, c), '--',
                 label='ellipse fit')
        plt.xlabel('angle')
        plt.ylabel('mean radius')
        plt.legend()
        plt.grid()

        plt.subplot2grid((1, 3), (0, 2))
        x, y = polar_to_cart(np.zeros(2), angles, mean_rad, False)
        plt.plot(x, y, ':', label='raw')
        x, y = polar_to_cart(np.zeros(2), angles, mean_rad_s, False)
        plt.plot(x, y, label='shifted')
        x, y = polar_to_cart(np.zeros(2), angles, r0, False)
        plt.plot(x, y, '--', label='circle')
        plt.gca().set_aspect('equal')
        plt.legend()
        plt.grid()
        plt.show()

    adjusted = shift_radius(angles, radii, -x0, -y0)
    if un_ellipse:
        adjusted *= r0 / ellipse(angles, a, b, c)
    return adjusted, (a, b, c, x0, y0)


class DataSet(object):
    def __init__(self, data_folders, setups,
                 out_folder='./out',
                 laser_data='./LaserProtonPointingAugustRun.txt',
                 min_time=None, max_time=None,
                 plot=True, process=True):
        self.out_folder = out_folder
        self.data_folders = data_folders
        self.plot = plot

        if process:
            if os.path.isfile(data_folders[0]):
                if isinstance(setups, dict):
                    self.setups = [setups]
                elif len(setups) == 1:
                    self.setups = setups
                else:
                    raise ValueError('If first argument is a file list, '
                                     'can only have one setup (must be dict).')
                self.process_list(data_folders)
            else:
                self.setups = setups
                self.process_data(min_time, max_time)

        self.names, self.data, radii1, radii2 = load_radii(
            out_folder, min_time, max_time,
            data_folders if os.path.isfile(data_folders[0]) else None)

        self.parse_radii(radii1, radii2)
        self.laser = dict()
        self.parse_laser(laser_data)

    def parse_radii(self, radii1, radii2):
        # check time repetitions and rm duplicates
        unique = self['cam_time'] != np.roll(self['cam_time'], 1)
        print('Deleting %s duplicates.' % sum(~unique))
        radii1 = radii1[unique]
        radii2 = radii2[unique]
        for key in self.data:
            self.data[key] = self[key][unique]
        self.names = self.names[unique]

        # shift center and fit ellipses
        radii1, self.popt1 = adjust_elliptic(radii1)
        print('fit 2: a=%s b=%s c=%s x0=%s y0=%s' % self.popt1,
              'eccentricity: ', self.eccentric1)
        radii2, self.popt2 = adjust_elliptic(radii2)
        print('fit 2: a=%s b=%s c=%s x0=%s y0=%s' % self.popt2,
              'eccentricity: ', self.eccentric2)

        # smooth
        self['radii1'] = radii1
        self['radii2'] = radii2

        # compute diameters; note one data point may be lost for odd length
        self['nrad'] = (radii1.shape[1], radii2.shape[1])
        self['ndiam'] = (radii1.shape[1] // 2, radii2.shape[1] // 2)
        self['diam1'] = (radii1[:, :self.ndiam[0]] +
                         radii1[:, self.ndiam[0]:2 * self.ndiam[0]])
        self['diam2'] = (radii2[:, :self.ndiam[1]] +
                         radii2[:, self.ndiam[1]:2 * self.ndiam[1]])

        # compute some things
        self['mean1'] = np.mean(self.diam1[:, ::10], axis=1)
        self['mean2'] = np.mean(self.diam2, axis=1)

        self['var1'] = np.var(self.diam1[:, ::10], axis=1)
        self['var2'] = np.var(self.diam2, axis=1)

        self['max1'] = np.max(self.diam1[:, ::10], axis=1)
        self['max2'] = np.max(self.diam2, axis=1)

        self['min1'] = np.min(self.diam1[:, ::10], axis=1)
        self['min2'] = np.min(self.diam2, axis=1)

        self['r_mean1'] = np.mean(self.diam1[:, ::10], axis=1)
        self['r_mean2'] = np.mean(self.diam2, axis=1)

        self['r_var1'] = np.var(self.diam1[:, ::10], axis=1)
        self['r_var2'] = np.var(self.diam2, axis=1)

        self['r_max1'] = np.max(self.diam1[:, ::10], axis=1)
        self['r_max2'] = np.max(self.diam2, axis=1)

        self['r_min1'] = np.min(self.diam1[:, ::10], axis=1)
        self['r_min2'] = np.min(self.diam2, axis=1)

    def parse_laser(self, data_path):
        laser = dict()
        with open(data_path, 'r') as laser_data:
            s = laser_data.readline()
            for line in laser_data:
                values = line.strip().split(';')
                if values[0] == 'NaN':
                    continue
                laser[int(values[0])] = [float(s.strip()) for s in values[1:-1]]

            laser_in = []
            laser_out = []
            beam_in = []
            beam_out = []

            keys = list(laser.keys())
            for i in range(len(self.names)):
                try:
                    k = np.where(np.isclose(keys, self['bct_time'][i],
                                            rtol=1e-9))[0][0]
                    line = laser[keys[k]]
                    laser_in.append(line[0:2])
                    laser_out.append(line[2:4])
                    beam_in.append(line[4:6])
                    beam_out.append(line[6:8])
                except IndexError:
                    laser_in.append([np.nan] * 2)
                    laser_out.append([np.nan] * 2)
                    beam_in.append([np.nan] * 2)
                    beam_out.append([np.nan] * 2)

            self.laser['laser_in'] = laser_in = np.array(laser_in)
            self.laser['laser_out'] = laser_out = np.array(laser_out)
            self.laser['beam_in'] = beam_in = np.array(beam_in)
            self.laser['beam_out'] = beam_out = np.array(beam_out)

            self.laser['diff_in'] = np.linalg.norm(
                laser_in - beam_in, axis=1)
            self.laser['diff_out'] = np.linalg.norm(
                laser_out - beam_out, axis=1)
            self.laser['laser_tilt'] = np.linalg.norm(
                laser_out - laser_in, axis=1)
            self.laser['misa_tilt'] = np.linalg.norm(
                (laser_in - beam_in) - (laser_out - beam_out), axis=1)
            self.laser['misa_shift'] = np.linalg.norm(
                (laser_in - beam_in) + (laser_out - beam_out), axis=1)

    def process_data(self, min_time=None, max_time=None, plot=None):
        for folder, setup in zip(self.data_folders, self.setups):
            reader = Reader(setup, folder, self.out_folder)
            reader.process_all(min_time, max_time, plot or self.plot)

    def process_list(self, file_names):
        reader = Reader(self.setups[0], './', self.out_folder)
        reader.process_list(file_names)

    def update_meta(self, setups, logging=True):
        self.setups = setups
        for folder, setup in zip(self.data_folders, self.setups):
            update_meta(setup, folder, self.out_folder, logging)

    def __setitem__(self, key, value):
        self.data[key] = value

    def __getitem__(self, item,):
        if isinstance(item, tuple):
            item, i = item
            getter = NAMES[item]
            return getter[1](self, str(i))
        try:
            return getattr(self, item)
        except AttributeError:
            raise KeyError('Data set does not have the item ' + str(item))

    def __getattr__(self, item):
        try:
            return self.data[item]
        except KeyError:
            pass
        try:
            return self.laser[item]
        except KeyError:
            pass
        try:
            getter = NAMES[item]
            return getter[1](self, None)
        except KeyError:
            raise AttributeError('Data set does not have the item ' + str(item))

    def __len__(self):
        return len(self.names)

    def update(self, other):
        if not isinstance(other, DataSet):
            raise ValueError("Can only join two data sets.")
        self.data_folders = list(set(self.data_folders + other.data_folders))
        new_names = list(set(other.names) ^ set(self.names))
        new_filter = [n in new_names for n in other.names]
        self.names = np.append(self.names, other.new_names)
        for key in self.data:
            self.data[key].extend(other.data[new_filter])
        for key in self.laser:
            self.laser[key].extend(other.laser[new_filter])

    @property
    def eccentric1(self):
        return np.sqrt(1 - min(self.popt1[0] / self.popt1[1],
                               self.popt1[1] / self.popt1[0]) ** 2)

    @property
    def eccentric2(self):
        return np.sqrt(1 - min(self.popt2[0] / self.popt2[1],
                               self.popt2[1] / self.popt2[0]) ** 2)

    def show_img(self, names):
        try:
            names = self.names[names]
        except IndexError:
            if isinstance(names, str):
                names = [names]
        if not isinstance(names[0], str):
            raise ValueError('Names must be of type int or str.')

        where = [np.argwhere(self.names == n)[0][0] for n in names]
        titles = [
            n + ' || σ1/μ1=%.4f, σ2/μ2=%.4f, laser_energy=%.2f, vapor=%.2g' % (
                NAMES['rel_std'][1](self, 1)[w],
                NAMES['rel_std'][1](self, 2)[w],
                self['laser_energy_1'][w],
                NAMES['density_mean'][1](self, 1)[w])
            for w, n in zip(where, names)]
        return MeasureImgs(names, titles, self.out_folder)

    # PLOTTING
    def _get_name(self, name, i=None):
        if isinstance(name, tuple):
            try:
                return name[0], name[1](i)
            except TypeError:
                return name
        xname, xget = NAMES[name]
        if xget is None:
            return xname, self[name]
        try:
            return xname, xget(self, i)
        except KeyError:
            pass
        return xname, self[name]

    def plot_meta(self, fig=None, ax=None):
        ax = _get_ax(fig, ax, figsize=(10, 4))
        ax.set_title('Qualitative Parameter Range')
        ax.plot(self['proton_count'] / self['proton_count'][0],
                label='proton count')
        ax.plot(self['bunch_len'] / self['bunch_len'][0],
                label='bunch len')
        timing = NAMES['timing'][1](self, None)
        ax.plot(timing / timing[0],
                label='laser-bunch offset')
        ax.legend()
        ax.grid()
        return ax

    def plot_correlations(self, fig=None):
        if fig is None:
            fig = plt.figure(figsize=(18, 10))

        ax1 = fig.add_subplot(231)
        plt.plot(self.mean1, self.mean2, 'o', alpha=.5)
        plt.title('mean diameter')
        plt.xlabel('measurement 1')
        plt.ylabel('measurement 2')
        plt.grid()

        fig.add_subplot(232, sharex=ax1, sharey=ax1)
        plt.plot(self.max1, self.max2, 'o', alpha=.5)
        plt.title('max diameter')
        plt.xlabel('measurement 1')
        plt.ylabel('measurement 2')
        plt.grid()

        fig.add_subplot(233, sharex=ax1, sharey=ax1)
        plt.plot(self.min1, self.min2, 'o', alpha=.5)
        plt.title('min diameter')
        plt.xlabel('measurement 1')
        plt.ylabel('measurement 2')
        plt.grid()

        fig.add_subplot(234)
        plt.plot(np.sqrt(self.var1), np.sqrt(self.var2), 'o', alpha=.5)
        plt.title('diameter rms deviation')
        plt.xlabel('measurement 1')
        plt.ylabel('measurement 2')
        plt.grid()

        fig.add_subplot(235)
        plt.plot(np.sqrt(self.var1) / self.mean1,
                 np.sqrt(self.var2) / self.mean2, 'o', alpha=.5)
        plt.title('relative diameter rms deviation')
        plt.xlabel('measurement 1')
        plt.ylabel('measurement 2')
        plt.grid()

        fig.add_subplot(236)
        plt.plot(self['laser_energy_1'], self['laser_energy_2'], 'o', alpha=.5)
        plt.title('laser energy')
        plt.xlabel('measurement 1')
        plt.ylabel('measurement 2')
        plt.grid()

        return fig

    def plot_single(self, x='laser_energy1', y='mean', i=1, col=None, cut=None,
                    fig=None, ax=None, colorbar=True, **plot_args):
        ax = _get_ax(fig, ax, figsize=(14, 7))

        if col is not None:
            col_name, col_data = self._get_name(col)
        else:
            col_name = col_data = None

        xname, xdata = self._get_name(x, i)
        yname, ydata = self._get_name(y, i)

        plt.sca(ax)
        if cut is None:
            scat = plt.scatter(xdata, ydata, c=col_data, **plot_args)
        else:
            cdata = None if col_data is None else col_data[cut]
            scat = plt.scatter(xdata[cut], ydata[cut],
                               c=cdata, **plot_args)
        plt.xlabel(xname)
        plt.ylabel(yname)
        plt.grid(True)
        plt.legend(loc='best')
        if colorbar and col is not None:
            plt.colorbar(label=col_name)

        return scat

    def plot_double(self, x, y, col=None, fig=None, share_axes=True,
                    cuts=(None, None), **plot_args):
        if fig is None:
            fig = plt.figure(figsize=(20, 7))

        if col is not None:
            col_name, col_data = self._get_name(col)
            clim = (min(col_data), max(col_data))
            if share_axes:
                plot_args['vmin'], plot_args['vmax'] = clim
        else:
            col_name = None

        ax1 = fig.add_subplot(121)
        plt.title('Measurement 1')
        self.plot_single(
            x, y, 1, col, cuts[0], None, ax1, not share_axes, **plot_args)

        if share_axes:
            ax2 = fig.add_subplot(122, sharex=ax1, sharey=ax1)
        else:
            ax2 = fig.add_subplot(122)
        plt.title('Measurement 2')
        scat = self.plot_single(
            x, y, 2, col, cuts[1], None, ax2, not share_axes, **plot_args)

        if share_axes:
            plt.setp(ax2.get_yticklabels(), visible=False)
            plt.ylabel('')

        if col and share_axes:
            cb_axis = fig.add_axes([0.9, 0.1, 0.01, 0.8])
            plt.colorbar(scat, cax=cb_axis, label=col_name)
            plt.subplots_adjust(right=.88, wspace=.04)

        return fig

    def plot_2in1(self, x, y, fig=None, ax=None, cuts=(None, None),
                  labels=('IS1', 'IS2'), **plot_args):
        ax = _get_ax(fig, ax, figsize=(14, 7))

        plot_args['label'] = labels[0]
        args_1 = dict()
        args_1.update(plot_args)
        args_1['colorbar'] = False
        self.plot_single(x, y, 1, cut=cuts[0], ax=ax, **args_1)

        plot_args['label'] = labels[1]
        self.plot_single(x, y, 2, cut=cuts[1], ax=ax, **plot_args)

        return fig
