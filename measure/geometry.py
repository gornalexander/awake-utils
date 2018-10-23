import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def project(center, x, y, x_grad, y_grad):
    dx = x - center[1]
    dy = y - center[0]
    norm = np.sqrt(dx ** 2 + dy ** 2)
    norm[norm == 0] = 1
    return (x_grad * dx + y_grad * dy) / norm


def polar_to_x(center, angle, radius, to_int=True) -> tuple:
    y, x = (center[0] + radius * np.sin(2 * angle * np.pi),
            center[1] + radius * np.cos(2 * angle * np.pi))  # (y, x)
    if to_int:
        return y.astype(int), x.astype(int)
    else:
        return y, x


def image_in_polar(center, image, angle_count, max_radius, min_radius=0):
    polar_radius = np.arange(min_radius, max_radius)
    polar_angle = np.linspace(0, 1, angle_count)
    polar_mgrid = np.meshgrid(polar_angle, polar_radius)

    y, x = polar_to_x(center, *polar_mgrid)
    return image[y.flatten(), x.flatten()].reshape(x.shape)


def shift_radius(angles, radii, x0, y0):
    new_radii = np.sqrt(
        x0 ** 2 + y0 ** 2 + radii ** 2 -
        2 * radii * (x0 * np.cos(2 * np.pi * angles) +
                     y0 * np.sin(2 * np.pi * angles)))
    return new_radii


def ellipse_r(theta, a, b, c):
    return a * b / np.sqrt(
        a ** 2 * np.sin(np.pi * 2 * theta + c) ** 2 + b ** 2 * np.cos(
            np.pi * 2 * theta + c) ** 2)


def ellipse_rs(theta, a, b, c, x0, y0):
    return shift_radius(theta, ellipse_r(theta, a, b, c), x0, y0)


def fit_ellipse(angles, radii):
    r0 = np.mean(radii)
    popt, pcov = curve_fit(ellipse_r, angles, radii, p0=(r0, r0, 0))
    return popt


def fit_circle(angles, radii, r0=10.):
    popt, pcov = curve_fit(lambda a, r, x0, y0: shift_radius(a, r, x0, y0),
                           angles, radii, p0=(r0, 0, 0))
    return popt


def adjust_elliptic(radii, un_ellipse=False, plot=True):
    angles = np.linspace(0, 1, len(radii[0]))
    mean_rad = np.mean(radii, axis=0)
    r0: float = np.mean(mean_rad)

    r, x0, y0 = fit_circle(angles, mean_rad, r0)
    mean_rad_s = shift_radius(angles, mean_rad, -x0, -y0)
    a, b, c = fit_ellipse(angles, mean_rad_s)

    if plot:
        plt.figure(figsize=(15, 4))
        plt.subplot2grid((1, 3), (0, 0), colspan=2)
        plt.plot(angles * 360, mean_rad, ':', label='raw')
        plt.plot(angles * 360, mean_rad_s, '-', label='shifted')
        plt.plot(angles * 360, ellipse_r(angles, a, b, c), '--',
                 label='ellipse fit')
        plt.xlabel('angle')
        plt.ylabel('mean radius')
        plt.legend()
        plt.grid()

        plt.subplot2grid((1, 3), (0, 2))
        y, x = polar_to_x(np.zeros(2), angles, mean_rad, False)
        plt.plot(x, y, ':', label='raw')
        y, x = polar_to_x(np.zeros(2), angles, mean_rad_s, False)
        plt.plot(x, y, label='shifted')
        y, x = polar_to_x(np.zeros(2), angles, r0, False)
        plt.plot(x, y, '--', label='circle')
        plt.gca().set_aspect('equal')
        plt.legend()
        plt.grid()
        plt.show()

    adjusted = shift_radius(angles, radii, -x0, -y0)
    if un_ellipse:
        adjusted *= r0 / ellipse_r(angles, a, b, c)
    return adjusted, (a, b, c, x0, y0)
