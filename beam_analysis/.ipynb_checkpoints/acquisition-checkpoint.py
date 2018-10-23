import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed only for Camera.plot_3d


from .geometry import image_in_polar, polar_to_cart
from .util import mm_mgrid, EventCorruptError
from .mask import apply_mask
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter


# sigma for gaussian filter applied to core image to find the center
CORE_CENTER_SIGMA = 10
ANGLE_TICKS = [0, 45, 90, 135, 180, 225, 270, 315, 360]
ANGLE_LABEL = ['%s°' % a for a in ANGLE_TICKS]


def _try_get(df, keys, default=np.nan, trafo=None):
    try:
        res = df
        for key in keys:
            res = res[key]
        return res if trafo is None else trafo(res)
    except (ValueError, KeyError):
        return default


def get_meta(df, setup):
    """ Load the meta information from data file using a META setup. """
    meta = dict()
    for key in setup:
        meta[key] = _try_get(df, *setup[key])
    return meta


def transfer_shift(point, shift, from_px_width, to_px_width):
    """ Translate pixel coordinates from one image to another. """
    return (point + shift) * from_px_width / to_px_width


def radius_limit(data, center):
    """ Find a reasonable largest radius given an image and the center. """
    return int(min(center[0], data.shape[0] - center[0],
                   center[1], data.shape[1] - center[1]) * .99)


class Camera(object):
    def __init__(self, data, px_width=1, center=None):
        """ Extract camera data from df using specified setup. """
        self.data = data
        self.center = center
        self.px_width = px_width

    @property
    def shape(self):
        return self.data.shape

    @property
    def max_radius(self):
        return radius_limit(self.data, self.center)

    def get_polar(self, angles, min_rad=1, max_rad=None):
        """ Get image data in polar coordinates, angles along dim 2. """
        if max_rad is None:
            max_rad = self.max_radius
        return image_in_polar(self.data, self.center, max_rad, angles, min_rad)

    def get_radial(self, angle, min_rad=1, max_rad=None):
        """ Get radial profile along given angle. """
        if max_rad is None:
            max_rad = self.max_radius
        x, y = polar_to_cart(self.center, angle, np.arange(min_rad, max_rad))
        return self.data[y.astype(int), x.astype(int)]

    def get_radial_mean(self, angle_count, min_rad=1, max_rad=None):
        """ Get mean radial profile. """
        if max_rad is None:
            max_rad = self.max_radius
        _, polar = self.get_polar(angle_count, min_rad, max_rad)
        return np.average(polar, axis=1)

    def plot(self, log=True, aspect='equal', **kwargs):
        """ Plot the camera image (in log scale). """
        plt.pcolormesh(*mm_mgrid(self.px_width, self.shape, self.center),
                       self.data, norm=LogNorm() if log else None, **kwargs)
        plt.gca().set_aspect(aspect)
        plt.xlabel(r'adjusted position $\mathrm{[mm]}$')
        plt.ylabel(r'adjusted position $\mathrm{[mm]}$')

    def plot_polar(self, log=True, angles=360, min_rad=1, **kwargs):
        """ Plot image in polar projection, with angles along the x-axis."""
        max_rad = self.max_radius
        _, polar = image_in_polar(self.data, self.center,
                                  max_rad, angles, min_rad)
        mgrid = np.meshgrid(np.linspace(0, 360, angles),
                            np.linspace(min_rad * self.px_width,
                                        max_rad * self.px_width,
                                        max_rad - min_rad))
        plt.pcolormesh(*mgrid, polar,
                       norm=LogNorm() if log else None, **kwargs)
        plt.xlabel('angle')
        plt.ylabel(r'radius $\mathrm{[mm]}$')
        plt.xticks(ANGLE_TICKS, ANGLE_LABEL)

    def plot_radial(self, angle=None, log=False, **kwargs):
        """ Plot the intensity depending on the radius.

        Angle is a value between 0 and 1. If angle is None, the average is
        shown.
        """
        if angle is None:
            polar = self.get_radial_mean(2 * self.max_radius)
            if log:
                polar = np.ma.log(polar)
            plt.plot(polar, **kwargs)
        else:
            polar = self.get_radial(angle)
            if log:
                polar = np.ma.log(polar)
            plt.plot(polar, **kwargs)

    def plot_radial_var(self, log=False, angles=360, min_rad=1, max_rad=None,
                        mode='-', cmap=plt.cm.get_cmap('RdYlGn'), **kwargs):
        """ Plot the deviations from the mean radial profile.

        :param mode: may be '-' or '/'. For '-' the difference between actual
            and mean value is computed, for '/' it is the ratio.
        """
        (x, y), polar = self.get_polar(angles, min_rad, max_rad)
        mean = self.get_radial_mean(2 * self.max_radius, min_rad, max_rad)

        if mode == '/':
            data = polar / mean[:, np.newaxis]
        elif mode == '-':
            data = polar - mean[:, np.newaxis]
        else:
            raise ValueError("Only modes are - and /.")
        plt.pcolormesh(x, y, data, cmap=cmap,
                       norm=LogNorm() if log else None, **kwargs)
        plt.colorbar(label='data / mean' if mode == '/' else 'data - mean')
        plt.gca().set_aspect('equal')

    def plot_3d(self, log=False, cmap=plt.get_cmap('jet'), **kwargs):
        """ 3D surface plot. Set cmap to None for no color. """
        ax = plt.subplot(111, projection='3d')
        data = np.ma.log(self.data) if log else self.data
        ax.plot_surface(*mm_mgrid(self.px_width, self.shape, self.center),
                        data, cmap=cmap, **kwargs)

    def save_image(self, path, log=True, dpi=80, **kwargs):
        """ Save the data as image at given path.

        Matplotlib methods such as `imshow` and `pcolormesh` are built to
        automatically scale the image data. This function can be used to
        obtain an image showing exactly each pixel of the data.

        If the argument log is True, the pixel intensity is logarithmic.
        Additional parameters to figimage (e.g. cmap) may be passed as keyword
        arguments.
        """
        xpixels, ypixels = self.data.shape
        fig = plt.figure(figsize=(ypixels / dpi, xpixels / dpi), dpi=dpi)
        fig.figimage(np.ma.log(self.data) if log else self.data, **kwargs)
        plt.savefig(path)
        plt.close()


class DualCamera(object):
    def __init__(self, df, setup):
        self.shift = setup['shift']

        core_data, (c_t, c_exp) = setup['core'][0](df, setup['core'][1])
        if setup['core_trafo'] is not None:
            core_data = setup['core_trafo'](core_data)
        halo_data, (h_t, h_exp) = setup['halo'][0](df, setup['halo'][1])
        if setup['halo_trafo'] is not None:
            halo_data = setup['halo_trafo'](halo_data)

        if not np.isclose(c_t, h_t):
            raise EventCorruptError
        self.exposure = (c_exp, h_exp)
        self.time = h_t

        core_center = np.unravel_index(
            np.argmax(gaussian_filter(core_data, CORE_CENTER_SIGMA), axis=None),
            core_data.shape)
        core_center = np.array(core_center)
        halo_center = transfer_shift(core_center, self.shift,
                                     setup['core_px_mm'], setup['halo_px_mm'])

        self.core = Camera(core_data, setup['core_px_mm'], core_center)
        self.halo = Camera(halo_data, setup['halo_px_mm'], halo_center)

        if setup['core_mask'] is not None:
            apply_mask(self.core, setup['core_mask'])
        if setup['halo_mask'] is not None:
            apply_mask(self.halo, setup['halo_mask'])

    def plot(self, log=True, axes=None, **kwargs):
        """ Plot the halo and core cameras.

        The axes argument may be a tuple of axes onto which halo and core
        are plotted.
        """
        if axes is None:
            plt.figure(figsize=(12, 6))
            ax1 = plt.subplot(121)
            ax2 = plt.subplot(122)
        else:
            ax1, ax2 = axes

        plt.sca(ax1)
        self.halo.plot(log, **kwargs)
        plt.axhline(0, linestyle='--', color='C1')
        plt.axvline(0, linestyle='--', color='C1')
        plt.sca(ax2)
        self.core.plot(log, **kwargs)
        plt.axhline(0, linestyle='--', color='C1')
        plt.axvline(0, linestyle='--', color='C1')
