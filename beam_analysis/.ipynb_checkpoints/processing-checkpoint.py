import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
from threading import Thread
from .acquisition import get_meta, DualCamera
from .util import safe_join, EventCorruptError
from .geometry import polar_to_cart


def update_meta(setup, base_dir, out_dir='./out', log_process=True):
    """ Update the meta data saved in file (e.g. when setup was updated). """
    # TODO: allow base_dir to be a list of directories
    meta_file = os.path.join(out_dir, 'meta.dat')
    with open(meta_file, 'w') as meta_out:
        meta_out.write('\t'.join(setup.keys()) + '\n')
        with open(os.path.join(out_dir, 'measurements.txt'), 'r') as done:
            lines = done.readlines()
            count = len(lines)
            for i, line in enumerate(lines):
                path = os.path.join(base_dir, line.strip())
                if log_process:
                    print('%d/%d: %s' % (i+1, count, line.strip()))
                ds = h5py.File(path, 'r')
                meta = get_meta(ds, setup)
                ds.close()
                meta_out.write('\t'.join(str(meta[k]) for k in meta) + '\n')


class Reader(object):
    def __init__(self, full_setup, base_dir, out_dir='./out/'):
        """ Process files in base_dir and save to out_dir.

        The output structure is as follows:

        out_dir/
            meta.dat                # Save metadata, tab separated
            blocked.dat             # Indicate if radii were blocked by mask
            measurements.txt        # List of processed file names
            errors.txt              # List of files where errors occurred
            corrupt.txt             # List files where EventCorrupt was thrown
            plots/
                1/
                    filename.png    # Plots showing the radii
                2/
                    filename.png
            radii/
                centers_1.dat       # Center position in mm
                centers_2.dat
                1/
                    data.npy        # Load using np.load, radii in mm
                2/
                    data.npy

        """
        self.full_setup = full_setup
        self.setup_1 = full_setup.IS1
        self.setup_2 = full_setup.IS2
        self.rad_processing = full_setup.get_radii
        self.meta = full_setup.META

        self.base_dir = base_dir
        self.meta_file = os.path.join(out_dir, 'meta.dat')
        self.done_file = os.path.join(out_dir, 'measurements.txt')
        self.corrupt_file = os.path.join(out_dir, 'corrupt.txt')
        self.error_file = os.path.join(out_dir, 'errors.txt')
        self.blocked_file = os.path.join(out_dir, 'blocked.dat')
        plot_dir = safe_join(out_dir, 'plots/')
        self.plot_1_dir = safe_join(plot_dir, '1/')
        self.plot_2_dir = safe_join(plot_dir, '2/')
        radii_dir = safe_join(out_dir, 'radii/')
        self.center_1_file = os.path.join(radii_dir, 'centers_1.dat')
        self.center_2_file = os.path.join(radii_dir, 'centers_2.dat')
        self.radii_1_dir = safe_join(radii_dir, '1/')
        self.radii_2_dir = safe_join(radii_dir, '2/')

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
                if in_file.readline() != '\t'.join(self.meta.keys()) + '\n':
                    raise RuntimeError('Meta file exists but keys do not'
                                       'match the setup. Run update_meta'
                                       'before trying to process new files.')

    def log_corrupt(self, name):
        with open(self.corrupt_file, 'a') as out:
            out.write(name + '\n')

    def log_error(self, name, error):
        with open(self.error_file, 'a') as out:
            out.write('-' * 10 + name + '\n')
            out.write(repr(error) + '\n')

    def write_done(self, name, radii_1, radii_2, center_1, center_2,
                   blocked_1, blocked_2, meta):
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

        with open(self.blocked_file, 'a') as blocked_out:
            blocked_out.write('%s\t%s\n' % (blocked_1, blocked_2))

        with open(self.done_file, 'a') as done:
            done.write(name + '\n')

    def _process_cam(self, cam, plot_dir, name, plot):
        blocked, angles, radii = self.rad_processing(cam)
        radii = radii * cam.halo.px_width
        center = cam.halo.center * cam.halo.px_width

        if plot:
            plt.figure(figsize=(15, 7))
            ax1 = plt.subplot(121)
            ax2 = plt.subplot(122)
            cam.plot(axes=(ax1, ax2))
            x, y = polar_to_cart([0, 0], angles, radii)
            ax1.plot(x, y, color='C3')
            ax2.plot(x, y, color='C3')
            plt.savefig(
                os.path.join(plot_dir, name.replace('.h5', '.png')))
            plt.close()

        return blocked, radii, center

    def _process_cams(self, ds, name, plot=True):
        cam_1 = DualCamera(ds, self.setup_1)
        blocked_1, radii_1, center_1 = self._process_cam(
            cam_1, self.plot_1_dir, name, plot)
        cam_2 = DualCamera(ds, self.setup_2)
        blocked_2, radii_2, center_2 = self._process_cam(
            cam_2, self.plot_2_dir, name, plot)

        if not np.isclose(cam_1.time, cam_2.time):
            raise EventCorruptError

        meta = get_meta(ds, self.meta)
        meta['cam_time'] = cam_1.time

        # threading prevents the operation from terminating unfinished
        thread = Thread(target=self.write_done, args=(
            name, radii_1, radii_2, center_1, center_2, blocked_1, blocked_2,
            meta))
        thread.start()
        thread.join()

    def process(self, path, plot=True):
        """ Process a single file at path. """
        name = os.path.basename(path)
        path = os.path.join(self.base_dir, path)
        try:
            with h5py.File(path, 'r') as ds:
                self._process_cams(ds, name, plot)
        except EventCorruptError:
            self.log_corrupt(name)
        except (ValueError, AssertionError) as e:
            self.log_error(name, e)

    def process_all(self, min_time=None, max_time=None, plot=True):
        """ Process all files in self.base_dir matching the constraints. """
        todo = [f for f in os.listdir(self.base_dir) if f.endswith('.h5')]
        if min_time is not None:
            todo = [f for f in todo if int(f.split('_')[0]) >= min_time]
        if max_time is not None:
            todo = [f for f in todo if int(f.split('_')[0]) <= max_time]

        self.process_list(todo, plot)

    def process_list(self, files, plot=True):
        """ Process all files in a list of paths. """
        if os.path.isfile(self.done_file):
            with open(self.done_file, 'r') as done_file:
                done = done_file.read().strip().split('\n')

            files = [f for f in files if os.path.basename(f) not in done]

        if os.path.isfile(self.corrupt_file):
            with open(self.corrupt_file, 'r') as empty_file:
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

    meta = np.loadtxt(os.path.join(out_dir, 'meta.dat'), skiprows=1, ndmin=2)
    meta = meta.transpose()
    assert len(meta[0]) == len(names), 'File list and meta.dat mismatch.'
    meta_dict = dict()
    with open(os.path.join(out_dir, 'meta.dat'), 'r') as meta_dat:
        keys = meta_dat.readline().strip().split('\t')
    for i, key in enumerate(keys):
        meta_dict[key] = meta[i][where]

    radii_1 = _load_radii_single(out_dir, '1', final_names)
    radii_2 = _load_radii_single(out_dir, '2', final_names)

    return final_names, meta_dict, radii_1, radii_2
