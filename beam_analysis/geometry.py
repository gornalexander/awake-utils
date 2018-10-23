import numpy as np
from scipy.optimize import curve_fit
import warnings


def polar_to_cart(center, angle, radius):
    """ Transform polar coordinates to cartesian. 
    
    The angle, as with other methods in this module, ranges from 0 to 1
    in a full circle. Center is expected to be in "image" coordinates,
    so the y-coordinate comes first.
    """
    y, x = (center[0] + radius * np.sin(2 * angle * np.pi),
            center[1] + radius * np.cos(2 * angle * np.pi))  # (y, x)
    return x, y


def cart_to_polar(center, x, y):
    """ Angles and radii given the cartesian coordinates and the center. """
    x0 = x - center[1]
    y0 = y - center[0]

    r = np.sqrt(x0**2 + y0**2)
    a = np.arctan2(y0, x0) / (2 * np.pi)

    # arctan2 returns values in [-pi, pi], this method returns values
    # corresponding to [0, 2 pi] (i.e. [-.5, .5], [0, 1] respectively)
    a[a < 0] = 1 + a[a < 0]

    return a, r


def image_in_polar(image, center, max_radius, angles=360, min_radius=1):
    """ Polar transformation of image array.

    The image is "unfolded" around the center, making the angles range over the
    x-axis, and radii range over the y-axis. Thus, the shape of the returned
    array is (max_radius-min_radius, angles).
    """
    # get mesh grid of polar coordinates
    polar_radius = np.arange(min_radius, max_radius)
    polar_angle = np.linspace(0, 1, angles)
    polar_mgrid = np.meshgrid(polar_angle, polar_radius)

    # get equivalent cartesian coordinates and evaluate the image there
    x, y = polar_to_cart(center, *polar_mgrid)
    flat_image = image[y.astype(int).flatten(), x.astype(int).flatten()]
    return (x, y), flat_image.reshape(x.shape)


def ellipse(angles, a, b, c):
    """ Radii of an ellipse with diameters a and b, rotated by c.

    The angle c ranges from 0 to 1 in a full circle, the range [0, 1/4]
    is therefore sufficient for all configurations (given a, b are free).
    """
    return a * b / np.sqrt(
        a ** 2 * np.sin(np.pi * 2 * (angles + c)) ** 2 +
        b ** 2 * np.cos(np.pi * 2 * (angles + c)) ** 2)


def shift_radius(angles, radii, x0, y0):
    """ Shift the radii observed at given angles to new observation point.

    If the radii correspond to a circle around x0, y0, the returned
    radii will all be an array with all entries equal. Note that the new
    angles are similar to the old angles only if x0, y0 are small relative
    to the radii.

    A simple example:
        >>> angles = np.array([0, .5])
        >>> radii = np.array([3., 1.])
        >>> new_angles, new_radii = shift_radius(angles, radii, 1, 0)
        >>> new_radii
        array([2., 2.])
        >>> new_angles
        array([0. , 0.5])
        >>> shift_radius(new_angles, new_radii, -1, 0)  # transform back
        (array([0. , 0.5]), array([3., 1.]))
    """
    new_radii = np.sqrt(
        x0 ** 2 + y0 ** 2 + radii ** 2 -
        2 * radii * (x0 * np.cos(2 * np.pi * angles) +
                     y0 * np.sin(2 * np.pi * angles)))

    new_cos = radii / new_radii * np.cos(2 * np.pi * angles) - x0 / new_radii
    # this try-except block makes the method more tolerant to numeric errors
    # the example in the docstring of fit_circle would otherwise yield
    # a runtime warning: invalid value in arccos.
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            new_angles = np.arccos(new_cos)
    except RuntimeWarning:
        new_angles = np.arccos(np.nextafter(new_cos, 0.))
    return new_angles / (2 * np.pi), new_radii


def circle(angles, r0, x0, y0):
    """ Get radius of a circle centered at (x0, y0) under given angle.

    Note that both the angle and the returned radius are relative to (0, 0)
    and not (x0, y0). Therefore, it is required x0**2 + y0**2 < r0**2.
    """
    return shift_radius(angles, r0, -x0, -y0)[1]


def fit_ellipse(angles, radii, r0=None, rmin=None, rmax=None):
    """ Compute optimal parameters for ellipse and the covariance matrix.

    The output is always ordered such that the long axis (a) is the first
    parameter.

    Example:
        >>> angles = np.array([0, .25, .5, .75])
        >>> radii = np.array([1., 2., 1., 2.])
        >>> popt, pcov = fit_ellipse(angles, radii)
        >>> '%.2f, %.2f, %.2f' % popt
        '2.00, 1.00, 0.25'
    """
    if rmin is None:
        rmin = np.min(radii)
    if rmax is None:
        rmax = np.max(radii)
    if r0 is None:
        r0 = (rmax + rmin) / 2
    popt, pcov = curve_fit(ellipse, angles, radii, p0=(r0, r0, 0),
                           bounds=([rmin, rmin, 0.], [rmax, rmax, .25]))

    if popt[0] < popt[1]:
        # reorder
        popt = (popt[1], popt[0], popt[2] + .25)
        new_order = [1, 0, 2]
        pcov = pcov[:, new_order][new_order]

    return popt, pcov


def fit_circle(angles, radii, r0=None, rmin=None, rmax=None):
    """ Fit circle to radii observed at given angle, allowing shift and scaling.

    Example:
        >>> angles = np.array([0, .5, .25, .75])
        >>> radii = np.array([3., 1., 2., 2.])
        >>> popt, pcov = fit_circle(angles, radii)
        >>> '%.f, %.f, %.f' % (*popt,)
        '2, 1, 0'
    """
    if rmin is None:
        rmin = np.min(radii)
    if rmax is None:
        rmax = np.max(radii)
    if r0 is None:
        r0 = (rmax + rmin) / 2
    return curve_fit(circle, angles, radii, p0=(r0, 0, 0),
                     bounds=([rmin, 0, 0], [rmax, rmax, rmax]))
