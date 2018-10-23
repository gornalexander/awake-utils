import numpy as np
from ..util import cam_extract_bovwa, cam_extract_btv
from ..mask import Box, Circle
from datetime import datetime
from ..measure import radii_gradient, radii_contour, interp_smooth_radii


# BTV53 = {
#     'core': (cam_extract_btv, 'TT41.BTV.412353'),
#     #'halo': (cam_extract_bovwa, 'BOVWA.02TCC4.CAM10'),
#     'core_px_mm': .044,  # length of one pixel in mm
#     #'halo_px_mm': 1 / 30,
#     'core_trafo': None,
#     #'halo_trafo': lambda img: np.fliplr(img),
#     'shift': np.array([9.73046533, 9.96887321]),

#     'core_mask': None,
#     #'halo_mask': None
# }

IS1 = {
    'core': (cam_extract_bovwa, 'BOVWA.07TCC4.AWAKECAM07'),
    'halo': (cam_extract_bovwa, 'BOVWA.02TCC4.CAM10'),
    'core_px_mm': .044,  # length of one pixel in mm
    'halo_px_mm': 1 / 30,
    'core_trafo': None,
    'halo_trafo': lambda img: np.fliplr(img),
    'shift': np.array([9.73046533, 9.96887321]),

    'core_mask': None,
    'halo_mask': None
}

IS2 = {
    'core': (cam_extract_bovwa, 'BOVWA.02TCC4.AWAKECAM02'),
    'halo': (cam_extract_bovwa, 'BOVWA.04TCC4.CAM12'),
    'core_px_mm': 30 / 960,
    'halo_px_mm': .04,
    'core_trafo': lambda d: np.fliplr(np.rot90(d)),
    'halo_trafo': lambda d: np.flipud(d),
    'shift': np.array([169.33635394, 705.04329489]),

    'core_mask': None,
    'halo_mask': (
        # corners starting bottom left, clock-wise
        Box((550, 115), (545, 1086), (1357, 1152), (1367, 68)) -
        Box((572, 130), (570, 1067), (1329, 1134), (1341, 91)) +
        # center plate
        Circle(957, 594, 246 // 2) +
        # markings
        Box((513, 592), (560, 607)) + Box((529, 563), (554, 583)) +      # left
        Box((933, 31), (944, 95)) + Box((956, 98), (971, 77)) +          # top
        Box((1349, 609), (1398, 626)) + Box((1371, 558), (1392, 583)) +  # right
        Box((917, 1112), (937, 1274)))                                   # bot.
}

META = {
    # count in 1e10
    'proton_count': (
        ['AwakeEventData', 'TT41.BCTF.412340', 'Acquisition',
         'totalIntensityPreferred', 0],
        None,   # default
        None),  # transform the value
    # 1 sigma length in ns
    'bunch_len': (
        ['AwakeEventData', 'SPSBQMSPSv1', 'Acquisition', 'bunchLengths', 0],
        np.nan, lambda l: 1e9 / 4 * l),
    # in ps
    'laser_bunch_offset': (
        ['AwakeEventData', 'VTUAwake2Frep', 'NormalModeAcq', 'b_offline', 0],
        np.nan, lambda t: 1e9 / 400788645 * (t - 445551)),
    # timing = offset - offset_fine
    'laser_bunch_offset_fine': (
        ['AwakeEventData', 'AWAKEInjPhaseShifter',
         'PhaseReadback', 'phaseACurrentValue', 0],
        np.nan, lambda t: 0.006930779632685896 * (t - 40)),
    # 1st halo filter
    'filter_1': (['AwakeEventData', 'BTV.TT41.412426_FWHEEL',
                  'Acquisition', 'position', 0], None, None),
    # 2nd halo filter
    'filter_2': (['AwakeEventData', 'BTV.TT41.412442_FWHEEL',
                  'Acquisition', 'position', 0], None, None),
    # in mJ
    'laser_energy_1': (['AwakeEventData', 'EMETER03', 'Acq', 'value', 0],
                       np.nan, lambda e: 1e3 * e),
    # in mJ
    'laser_energy_2': (['AwakeEventData', 'EMETER04', 'Acq', 'value', 0],
                       np.nan, lambda e: 1e3 * e),

    'bct_time': (
        ['AwakeEventData', 'TT41.BCTF.412340', 'Acquisition', 'acqTime', 0],
        np.nan,
        lambda t: 60 * 60 * 2 * 10 ** 9 + int(1e9 * datetime.strptime(
            t.decode(), '%Y/%m/%d %H:%M:%S.%f').timestamp())),

    # gas gradient
    'density_timestamp': (['AwakeEventData', 'TSG41.AWAKE-DENSITY-DATA',
                           'ValueAcquisition', 'floatValue', 0], None, None),

    'density_ups': (['AwakeEventData', 'TSG41.AWAKE-DENSITY-DATA',
                     'ValueAcquisition', 'floatValue', 1], None, None),

    'density_downs': (['AwakeEventData', 'TSG41.AWAKE-DENSITY-DATA',
                       'ValueAcquisition', 'floatValue', 2], None, None),

    'density_grad': (['AwakeEventData', 'TSG41.AWAKE-DENSITY-DATA',
                      'ValueAcquisition', 'floatValue', 3], None, None),
}

#
# def get_radii(dual_camera):
#     # using default parameters
#     blocked, a, r = radii_contour(dual_camera.halo, level=3)
#     # angles (a) are only used for plotting but not saved!
#     return blocked, a, r


# ALTERNATIVE
def get_radii(dual_camera):
    r = radii_gradient(dual_camera.halo, 360, .7)

    a, smooth = interp_smooth_radii(r, savgol_window=11)

    if np.ma.is_masked(r):
        return np.any(r.mask), a, smooth
    return None, a, smooth
