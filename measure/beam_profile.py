import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.signal import medfilt
import datetime
import h5py
from threading import Thread

from .default_setups import *
from .geometry import *


class EmptyImageError(Exception):
    pass


def get_meta(ds, setup):
    meta = dict()
    for key in setup['meta']:
        if setup['meta'][key] is None:
            continue
        meta[key] = setup['meta'][key](ds)

    cam_meta = get_cam_meta(ds, setup)
    assert (np.isclose(cam_meta['setup_1']['core']['time'],
                       cam_meta['setup_1']['halo']['time']) and
            np.isclose(cam_meta['setup_1']['halo']['time'],
                       cam_meta['setup_2']['core']['time']) and
            np.isclose(cam_meta['setup_2']['core']['time'],
                       cam_meta['setup_2']['halo']['time'])), 'Times unequal.'
    meta['cam_time'] = cam_meta['setup_1']['core']['time']

    meta['core_1_exposure'] = cam_meta['setup_1']['core']['exposure']
    meta['core_2_exposure'] = cam_meta['setup_2']['core']['exposure']
    meta['halo_1_exposure'] = cam_meta['setup_1']['halo']['exposure']
    meta['halo_2_exposure'] = cam_meta['setup_2']['halo']['exposure']
    return meta


def update_meta(setup, base_dir, out_dir='./out', log_process=True):
    meta_file = os.path.join(out_dir, 'meta.dat')
    with open(meta_file, 'w') as meta_out:
        meta_out.write('\t'.join(setup['meta'].keys()) + '\n')
        with open(os.path.join(out_dir, 'measurements.txt'), 'r') as done:
            lines = done.readlines()
            count = len(lines)
            for i, line in enumerate(lines):
                path = os.path.join(base_dir, line.strip())
                if log_process:
                    print('%d / %d: %s' % (i, count, line.strip()))
                ds = h5py.File(path, 'r')
                meta = get_meta(ds, setup)
                ds.close()
                meta_out.write('\t'.join(str(meta[k]) for k in meta) + '\n')


def get_cam_meta(ds, setup):
    meta = dict()

    for pos in ['setup_1', 'setup_2']:
        meta[pos] = dict()
        for focus in ['core', 'halo']:
            meta[pos][focus] = dict()
            name = setup[pos][focus][0]
            cam = ds['AwakeEventData'][name]
            if name.startswith('BOVWA.'):
                if cam['ExtractionImage'].attrs.get('exception'):
                    raise EmptyImageError
                time = cam['ExtractionImage']['imageTimeStamp'][0]
                exposure = cam['PublishedSettings']['exposureTime'][0]
            elif name.startswith('TT41.BTV'):
                if cam['Image'].attrs.get('exception'):
                    raise EmptyImageError
                acq_time = cam['Image']['acqTime'][0]
                time = datetime.datetime.strptime(
                    acq_time.decode(), '%Y/%m/%d %H:%M:%S.%f').timestamp()
                time = int(time * 1e9) + 2 * 60 * 60 * 1e9  # utc vs local
                exposure = None
            else:
                raise RuntimeError('Unknown camera type.')
            meta[pos][focus]['time'] = time
            meta[pos][focus]['exposure'] = exposure
    return meta


def image_hicontrast(image, sigma0=3, sigma=8, center=None):
    # smooth log of image
    log_data = np.ma.log(image)
    log_smoothed = gaussian_filter(log_data.astype(np.float64), sigma0)

    # gradient of log
    if center is None:
        center = np.mean(np.where(log_smoothed == np.max(log_smoothed)),
                         axis=1).astype(int)
    grad_y, grad_x = np.gradient(log_smoothed)

    # project gradient
    x = np.arange(image.shape[1])
    y = np.arange(image.shape[0])
    mgrid = np.meshgrid(x, y)
    grad = project(center, *mgrid, grad_x, grad_y)

    return gaussian_filter(grad, sigma)


def prompt_init_selection(data: np.ndarray, choices, angle, center, rad_max):
    plt.figure(figsize=(15, 4))

    ax1 = plt.subplot(121)
    y, x = polar_to_x(center, angle, np.arange(-rad_max, rad_max))
    strip = data[y, x]
    ax1.semilogy(strip, label='data')

    ax2 = plt.subplot(122)
    ax2.pcolormesh(data, norm=LogNorm())
    ax2.set_aspect(aspect='equal')

    for i, choice in enumerate(choices):
        ax1.axvline(center[0] + choice, label='%d' % i, color='C%d' % i)
        y, x = polar_to_x(center, angle, choice)
        ax2.plot(x, y, 'o', label='%d' % i, color='C%d' % i)

    ax1.legend()
    ax2.legend()
    ax1.grid()
    plt.show()

    choice = int(input('Choose the index: '))
    return choice


def measure_center_plate(cam):
    pol = image_in_polar(cam.center, cam.data, 400, 400)
    small = gaussian_filter(pol, 3) < 300
    shift = small[:-1] & ~small[1:]
    plate_rad = []
    for rad in shift.transpose():
        plate_rad.append(np.where(rad)[0][0])
    plate_rad = medfilt(plate_rad, 3)

    plt.imshow(small)
    plt.imshow(small, origin='lower')
    plt.plot(plate_rad)
    plt.colorbar()
    plt.show()

    angles = np.linspace(0, 1, 400)
    r0 = np.mean(plate_rad)
    (r, x0, y0) = fit_circle(angles, plate_rad, r0)
    plate_rad = shift_radius(angles, plate_rad, x0, y0)
    popt = fit_ellipse(angles, plate_rad)
    print('eccentricity: ',
          np.sqrt(1 - min(popt[0] / popt[1], popt[1] / popt[0]) ** 2))
    return popt


def plot_analysis(radii, data, grad, step, center, max_rad, short=True):
    angles = np.linspace(0, 1, len(radii))
    angle_count = len(angles)
    if short:
        plt.figure(figsize=(12, 4))
        plt.subplot(121)
        plt.pcolormesh(image_in_polar(center, data, angle_count, max_rad),
                       norm=LogNorm())
        plt.plot(angles[::20] * angle_count, radii[::20], ',', color='white')

        plt.subplot(122)
        plt.pcolormesh(np.array(data), norm=LogNorm())
        y, x = polar_to_x(center, angles, radii)
        plt.plot(x, y, ',', color='orange')
        plt.plot(center[1], center[0], '+', color='red')
    else:
        widths = np.ones_like(radii)

        plt.figure(figsize=(17, 8))
        plt.subplot(231)
        plt.pcolormesh(data, norm=LogNorm())
        #         plt.plot(contour[:, 1], contour[:, 0], color='black')
        plt.subplot(232)
        plt.imshow(grad, origin='lower', vmin=-.1, vmax=.1)
        plt.subplot(233)
        plt.imshow(step, origin='lower')

        plt.subplot(234)
        plt.pcolormesh(image_in_polar(center, data, angle_count, max_rad),
                       norm=LogNorm())
        plt.errorbar(angles[::20] * angle_count,
                     radii[::20], color='white', yerr=widths[::20], fmt=',')
        plt.subplot(235)
        plt.pcolormesh(np.array(data), norm=LogNorm())
        y, x = polar_to_x(center, angles, radii)
        plt.plot(x, y, ',', color='orange')
        plt.plot(center[1], center[0], '+', color='red')

        plt.subplot(236)
        plt.semilogy(data[center[0], :])
        plt.axvline(x=center[1] + radii[0], color='C1')
        plt.axvline(x=center[1] - radii[len(radii) // 2], color='C2')


def get_radii(data, config, center, plot=True, min_rad=2):
    # smooth log of image
    log_data = np.ma.log(np.minimum(data, config['above_noise']))
    log_mask = log_data.mask
    log_smoothed = config['log_filter'](log_data.astype(np.float64))

    # gradient of log
    grad_y, grad_x = np.gradient(log_smoothed)

    # project gradient
    x = np.arange(data.shape[1])
    y = np.arange(data.shape[0])
    mgrid = np.meshgrid(x, y)
    grad = project(center, *mgrid, grad_x, grad_y)
    grad[log_mask] = 1

    # regions with negative gradient
    negativity = config['neg_filter']((grad < 0) + .5 * (grad == 0))

    step = negativity > config['neg_threshold']

    # polar transformation
    max_rad = min(data.shape[0] - center[0], data.shape[1] - center[1]) * .9

    angle_count = config['angle_count']
    polar_angle = np.linspace(0, 1, angle_count)

    step_polar = image_in_polar(center, step, angle_count, max_rad, min_rad)

    # find transition
    diffs = step_polar[:-1] & ~step_polar[1:]

    # find radii
    radii = []
    for i in range(angle_count):
        radii.append(np.where(diffs[1:, i])[0] + min_rad)

    # filter radii

    # starting position should have unambiguous radius
    init = int(np.argmin([len(rads) for rads in radii]))
    if len(radii[init]) == 0:
        raise EmptyImageError
    elif len(radii[init]) == 1:
        radii[init] = radii[init][0]
    else:
        init = int(np.argmin([len(rads) for rads in radii]))
        prompt_angle = polar_angle[init]
        select = prompt_init_selection(data, radii[init],
                                       prompt_angle, center, max_rad)
        radii[init] = radii[init][select]

    for dist in range(1, len(radii)):
        keep = np.argmin(
            np.abs(radii[init - dist + 1] - radii[init - dist]))
        radii[init - dist] = radii[init - dist][keep]

    radii = config['rad_filter'](np.array(radii))

    if plot:
        plot_analysis(radii, data, grad, step, center, max_rad)

    return radii


class Camera(object):
    def __init__(self, ds, cam, px_len, trafo=None):
        event_data = ds['AwakeEventData']
        self.px_len = px_len  # in mm

        if cam.startswith('BOVWA.'):
            if event_data[cam]['ExtractionImage'].attrs.get('exception'):
                raise EmptyImageError
            self.data = np.array(
                event_data[cam]['ExtractionImage']['imageRawData'])

        elif cam.startswith('TT41.BTV.'):
            if event_data[cam]['Image'].attrs.get('exception'):
                raise EmptyImageError
            img = event_data[cam]['Image']
            self.data = np.array(img['imageSet']).reshape(img['nbPtsInSet2'][0],
                                                          img['nbPtsInSet1'][0])

        if trafo is not None:
            self.data = trafo(self.data)
        self.dimensions = (
            self.data.shape[0] * px_len, self.data.shape[1] * px_len)

    def plot(self, shift=(0, 0)):
        plt.imshow(np.ma.log(self.data), origin='lower',
                   extent=(shift[1] * self.px_len,
                           self.dimensions[1] + shift[1] * self.px_len,
                           shift[0] * self.px_len,
                           self.dimensions[0] + shift[0] * self.px_len))
        plt.gca().set_aspect(aspect='equal')

    def _repr_png_(self):
        plt.figure()
        self.plot()


class DualCameras(Camera):
    def __init__(self, ds, setup):
        self.core = Camera(ds, *setup['core'], setup['core_trafo'])
        super().__init__(ds, *setup['halo'], setup['halo_trafo'])

        smoothed = gaussian_filter(self.core.data, 10)
        max_where = np.unravel_index(np.argmax(smoothed, axis=None),
                                     smoothed.shape)
        self.center = ((np.array(max_where) + setup['shift']) *
                       self.core.px_len / self.px_len).astype(int)
        self.shift = setup['shift']
        self.raw_radii = None
        self.radii = None

    def process(self, config, plot=True, min_rad=2):
        radii = get_radii(self.data, config, self.center, plot, min_rad)

        self.raw_radii = radii
        self.radii = radii * self.px_len

    def plot(self, shift=(0, 0)):
        super().plot(shift)
        if self.radii is not None:
            angles = np.linspace(0, 1, len(self.radii))
            y, x = polar_to_x(self.center * self.px_len, angles,
                              self.radii, False)
            plt.plot(x, y, '-', color='orange')
            plt.plot(self.center[1] * self.px_len,
                     self.center[0] * self.px_len, 'o', color='red')

    def _repr_png_(self):
        plt.figure(figsize=(10, 4))
        plt.subplot(121)
        self.plot()
        plt.subplot(122)
        self.core.plot(self.shift)
        if self.radii is not None:
            angles = np.linspace(0, 1, len(self.radii))
            y, x = polar_to_x(self.center * self.px_len,
                              angles, self.radii, False)
            plt.plot(x, y, '-', color='orange')


def _save_join(base, relative):
    path = os.path.join(base, relative)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


class Reader(object):
    def __init__(self, full_setup, base_dir, out_dir='./out/'):
        self.full_setup = full_setup
        self.setup_1 = full_setup['setup_1']
        self.setup_2 = full_setup['setup_2']
        self.config = full_setup['radius_processing']
        self.meta = full_setup['meta']

        self.base_dir = base_dir
        self.meta_file = os.path.join(out_dir, 'meta.dat')
        self.done_file = os.path.join(out_dir, 'measurements.txt')
        self.empty_file = os.path.join(out_dir, 'empty.txt')
        self.error_file = os.path.join(out_dir, 'errors.txt')
        plot_dir = _save_join(out_dir, 'plots/')
        self.plot_1_dir = _save_join(plot_dir, '1/')
        self.plot_2_dir = _save_join(plot_dir, '2/')
        radii_dir = _save_join(out_dir, 'radii/')
        self.center_1_file = os.path.join(radii_dir, 'centers_1.dat')
        self.center_2_file = os.path.join(radii_dir, 'centers_2.dat')
        self.radii_1_dir = _save_join(radii_dir, '1/')
        self.radii_2_dir = _save_join(radii_dir, '2/')

        if not os.path.exists(self.plot_1_dir):
            os.makedirs(self.plot_1_dir)
        if not os.path.exists(self.plot_2_dir):
            os.makedirs(self.plot_2_dir)
        if not os.path.exists(self.radii_1_dir):
            os.makedirs(self.radii_1_dir)
        if not os.path.exists(self.radii_2_dir):
            os.makedirs(self.radii_2_dir)

        if not os.path.isfile(self.meta_file):
            with open(self.meta_file, 'w') as out:
                out.write('\t'.join(self.meta.keys()) + '\n')
        else:
            with open(self.meta_file, 'r') as in_file:
                assert in_file.readline() == '\t'.join(self.meta.keys()) + '\n'

    def log_empty(self, name):
        with open(self.empty_file, 'a') as out:
            out.write(name + '\n')

    def log_error(self, name, error):
        with open(self.error_file, 'a') as out:
            out.write('-' * 10 + name + '\n')
            out.write(repr(error) + '\n')

    def write_done(self, name, radii_1, radii_2, center_1, center_2, meta):

        np.save(os.path.join(self.radii_1_dir, name.replace('.h5', '.npy')),
                radii_1)
        np.save(os.path.join(self.radii_2_dir, name.replace('.h5', '.npy')),
                radii_2)

        with open(self.meta_file, 'a') as meta_out:
            meta_out.write('\t'.join(str(meta[k]) for k in meta) + '\n')

        with open(self.center_1_file, 'a') as center_out:
            center_out.write('\t'.join(str(i) for i in center_1.tolist()))
            center_out.write('\n')
        with open(self.center_2_file, 'a') as center_out:
            center_out.write('\t'.join(str(i) for i in center_2.tolist()))
            center_out.write('\n')

        with open(self.done_file, 'a') as done:
            done.write(name + '\n')

    def _process_cams(self, ds, name, plot=True):
        set_1 = DualCameras(ds, self.setup_1)
        set_1.process(self.config, plot)
        plt.savefig(
            os.path.join(self.plot_1_dir, name.replace('.h5', '.png')))
        plt.close()

        set_2 = DualCameras(ds, self.setup_2)
        set_2.process(self.config, plot)
        plt.savefig(
            os.path.join(self.plot_2_dir, name.replace('.h5', '.png')))
        plt.close()

        # threading prevents the operation from terminating unfinished
        thread = Thread(target=self.write_done, args=(
            name, set_1.radii, set_2.radii,
            set_1.center * set_1.px_len, set_2.center * set_2.px_len,
            get_meta(ds, self.full_setup)))
        thread.start()
        thread.join()

    def process(self, path, plot=True):
        name = os.path.basename(path)
        path = os.path.join(self.base_dir, path)
        try:
            with h5py.File(path, 'r') as ds:
                self._process_cams(ds, name, plot)
        except EmptyImageError:
            self.log_empty(name)
        except (ValueError, AssertionError) as e:
            self.log_error(name, e)

    def process_all(self, min_time=None, max_time=None, plot=True):
        todo = [f for f in os.listdir(self.base_dir) if f.endswith('.h5')]
        if min_time is not None:
            todo = [f for f in todo if int(f.split('_')[0]) >= min_time]
        if max_time is not None:
            todo = [f for f in todo if int(f.split('_')[0]) <= max_time]

        self.process_list(todo, plot)

    def process_list(self, files, plot=True):
        if os.path.isfile(self.done_file):
            with open(self.done_file, 'r') as done_file:
                done = done_file.read().strip().split('\n')

            files = [f for f in files if os.path.basename(f) not in done]

        if os.path.isfile(self.empty_file):
            with open(self.empty_file, 'r') as empty_file:
                empty = empty_file.read().strip().split('\n')

            files = [f for f in files if os.path.basename(f) not in empty]

        for path in files:
            self.process(path, plot)
            print('done: ' + path)


def _load_radii_single(out_dir, setup, names):
    first = np.load(
        os.path.join(out_dir, 'radii', setup, names[0][:-2] + 'npy'))
    radii = np.empty((len(names), len(first)))
    for i, name in enumerate(names):
        radii[i] = np.load(
            os.path.join(out_dir, 'radii', setup, name[:-2] + 'npy'))

    return radii


def load_radii(out_dir='./out/', min_time=None, max_time=None, file_list=None):
    with open(os.path.join(out_dir, 'measurements.txt'), 'r') as done:
        names = map(lambda name: name.strip(), done.readlines())
    names = list(filter(lambda name: name != '', names))

    where = np.full(len(names), True, dtype=bool)
    if file_list is not None:
        wanted = [os.path.basename(n).strip() for n in file_list]
        for i, n in enumerate(names):
            where[i] = names[i] in wanted
    if min_time is not None:
        for i in range(len(where)):
            where[i] *= int(names[i][:19]) >= min_time
    if max_time is not None:
        for i in range(len(where)):
            where[i] *= int(names[i][:19]) <= max_time

    final_names = np.array(names)[where]
    if len(final_names) == 0:
        raise RuntimeError("No files in " + out_dir + " matched the criteria.")

    meta = np.loadtxt(os.path.join(out_dir, 'meta.dat'), skiprows=1).transpose()
    assert len(meta[0]) == len(names), 'File list and meta.dat mismatch.'
    meta_dict = dict()
    with open(os.path.join(out_dir, 'meta.dat'), 'r') as meta_dat:
        keys = meta_dat.readline().strip().split('\t')
    for i, key in enumerate(keys):
        meta_dict[key] = meta[i][where]

    radii_1 = _load_radii_single(out_dir, '1', final_names)
    radii_2 = _load_radii_single(out_dir, '2', final_names)

    return final_names, meta_dict, radii_1, radii_2
