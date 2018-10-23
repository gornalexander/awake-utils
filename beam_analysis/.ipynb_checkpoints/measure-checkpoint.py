import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from skimage import measure as skm
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from .acquisition import Camera
from .geometry import polar_to_cart, cart_to_polar, image_in_polar
from .geometry import fit_circle, fit_ellipse, shift_radius
from .util import EventCorruptError

from scipy.signal import medfilt2d, medfilt
from scipy.ndimage import gaussian_filter


DEFAULT_LOG_FILTER = lambda log: medfilt2d(log, 7)
DEFAULT_NEG_FILTER = lambda neg: gaussian_filter(neg, 4)


def interp_smooth_radii(radii, angles=None, angle_count=360,
                        savgol_window=19, savgol_deg=1,
                        spline_s=None, spline_k=1):
    """ Interpolate missing values using spline and smooth using savgol. """
    if angles is None:
        angles = np.linspace(0, 1, len(radii))

    if np.ma.is_masked(radii):
        angles = angles[~radii.mask]
        radii = radii[~radii.mask]

    spline = UnivariateSpline(angles, radii, bbox=[0, 1], s=spline_s,
                              check_finite=False, k=spline_k)

    a = np.linspace(0, 1, angle_count)
    r = spline(a)
    return a, savgol_filter(r, savgol_window, savgol_deg, mode='wrap')


def find_contour(cam: Camera, level=2, noise=None):
    """ Compute the longest contour around an intensity of level * noise.

    If noise is None, it is computed via the average of pixel distant from
    the center. Note this only works if the halo is sufficiently small.
    """
    if noise is None:
        max_rad = cam.max_radius

        angles = np.linspace(0, 1, 2 * max_rad)
        radii = np.arange(max_rad * .9, max_rad)
        x, y = polar_to_cart(cam.center, *np.meshgrid(angles, radii))
        noise = np.mean(cam.data[y.astype(int), x.astype(int)])

    contours = skm.find_contours(cam.data, noise * level, 'high')
    border = max(contours, key=lambda c: c.shape[0])

    if np.ma.is_masked(cam.data):
        y, x = border.astype(int).transpose()
        m = cam.data.mask[y, x]

        border = np.ma.array(border, mask=np.stack((m, m)).transpose())

    return border


def contour_to_radii(center, border, angle_count=360, savgol_window=17):
    """ Transform contour into single radius per angle (linear spline). """
    angles, radii = cart_to_polar(center, border[:, 1], border[:, 0])

    if np.ma.is_masked(border):
        mask = border.mask[:, 0]

        angles = angles[~mask]
        radii = radii[~mask]
        is_blocked = np.any(mask)
    else:
        is_blocked = None

    df = pd.DataFrame({'a': angles, 'r': radii})
    df = df.groupby('a').mean()

    a, r = interp_smooth_radii(df.values, df.index, angle_count, savgol_window)
    return is_blocked, a, r


def radii_contour(cam: Camera, level=2, noise=None, angle_count=360):
    """ Compute radii using the contour method. """
    border = find_contour(cam, level, noise)
    return contour_to_radii(cam.center, border, angle_count)


def radii_gradient(cam: Camera, angle_count=360, neg_threshold=.9, min_rad=2,
                   log_filter=DEFAULT_LOG_FILTER,
                   neg_filter=DEFAULT_NEG_FILTER):
    """ Compute radii by computing the radial gradient. """
    # smooth log of image
    log_data = np.ma.log(cam.data.data)
    log_mask = log_data.mask
    log_smoothed = np.ma.array(log_filter(log_data.astype(np.float64)),
                               mask=log_mask)

    # gradient of log
    _, polar = image_in_polar(log_smoothed, cam.center, cam.max_radius,
                              angle_count, min_rad)
    # up-gradient for log-overflow (these are certainly outside of halo)
    polar[polar.mask] = 1

    grad = np.gradient(polar, axis=0)  # radial gradient

    # regions with negative (non-positive) gradient
    negativity = neg_filter((grad < 0) + .5 * (grad == 0))

    step = negativity > neg_threshold

    # find transition
    diffs = step[:-1] & ~step[1:]

    # find radii
    radii = []
    for i in range(angle_count):
        radii.append(np.where(diffs[1:, i])[0] + min_rad)

    # filter radii

    # starting position should have unambiguous radius
    init = int(np.argmin([len(rads) for rads in radii]))
    if len(radii[init]) == 0:
        raise EventCorruptError
    elif len(radii[init]) == 1:
        radii[init] = radii[init][0]
    else:
        init = int(np.argmin([len(rads) for rads in radii]))
        prompt_angle = 1 / (angle_count - 1) * init
        select = _prompt_init_selection(cam.data, radii[init], prompt_angle,
                                        cam.center, cam.max_radius)
        radii[init] = radii[init][select]

    for dist in range(1, len(radii)):
        keep = np.argmin(
            np.abs(radii[init - dist + 1] - radii[init - dist]))
        radii[init - dist] = radii[init - dist][keep]

    # plt.imshow(polar, origin='lower')
    # plt.plot(radii, '.', color='red')
    # plt.show()
    if np.ma.is_masked(cam.data):
        _, mask = image_in_polar(cam.data.mask, cam.center, cam.max_radius,
                                 angle_count, min_rad)
        rad_mask = mask[radii, np.arange(angle_count)]
        return np.ma.array(radii, mask=rad_mask)
    return np.array(radii)


def _prompt_init_selection(data: np.ndarray, choices, angle, center, rad_max):
    plt.figure(figsize=(15, 4))

    ax1 = plt.subplot(121)
    x, y = polar_to_cart(center, angle, np.arange(-rad_max, rad_max))
    strip = data[y.astype(int), x.astype(int)]
    ax1.semilogy(strip, label='data')

    ax2 = plt.subplot(122)
    ax2.pcolormesh(data, norm=LogNorm())
    ax2.set_aspect(aspect='equal')

    for i, choice in enumerate(choices):
        ax1.axvline(center[0] + choice, label='%d' % i, color='C%d' % i)
        x, y = polar_to_cart(center, angle, choice)
        ax2.plot(x, y, 'o', label='%d' % i, color='C%d' % i)

    ax1.legend()
    ax2.legend()
    ax1.grid()
    plt.show()

    choice = int(input('Choose the index: '))
    return choice


def measure_center_plate(cam, angle_count=200, threshold=300):
    _, pol = image_in_polar(cam.data, cam.center, cam.max_radius, angle_count)
    small = gaussian_filter(pol, 3) < threshold
    shift = small[:-1] & ~small[1:]
    plate_rad = []
    for rad in shift.transpose():
        plate_rad.append(np.where(rad)[0][0])
    plate_rad = medfilt(plate_rad, 3)

    angles = np.linspace(0, 1, angle_count)
    (r, x0, y0), _ = fit_circle(angles, plate_rad)
    angles, plate_rad = shift_radius(angles, plate_rad, x0, y0)
    popt, pcov = fit_ellipse(angles, plate_rad)
    eccentricity = np.sqrt(1 - min(popt[0] / popt[1], popt[1] / popt[0]) ** 2)
    return eccentricity, popt, pcov
