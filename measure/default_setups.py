import numpy as np
from scipy.signal import medfilt2d
from scipy.ndimage import gaussian_filter
from datetime import datetime


def _try_get(ds, keys, default=np.nan):
    try:
        res = ds
        for key in keys:
            res = res[key]
        return res
    except (ValueError, KeyError):
        return default


SETUP_17_SEPT_9 = {
    'meta': {
        # count in 1e10
        'proton_count':
            lambda ds: (ds['AwakeEventData']['TT41.BCTF.412340']
                          ['Acquisition']['totalIntensityPreferred'][0]),
        # 1 sigma length in ns
        'bunch_len':
            lambda ds: 1e9 / 4 * _try_get(
                ds,
                ['AwakeEventData', 'SPSBQMSPSv1', 'Acquisition',
                 'bunchLengths', 0],
                np.nan),
        # in ps
        'laser_bunch_offset':
            lambda ds: 1e9 / 400788645 * (
                _try_get(ds, ['AwakeEventData', 'VTUAwake2Frep',
                              'NormalModeAcq', 'b_offline', 0], np.nan)
                - 445551),
        # timing = offset - offset_fine
        'laser_bunch_offset_fine':
            lambda ds: 0.006930779632685896 * (
                _try_get(ds, ['AwakeEventData', 'AWAKEInjPhaseShifter',
                              'PhaseReadback', 'phaseACurrentValue', 0], np.nan)
                - 40),
        # 1st halo filter
        'filter_1':
            lambda ds: (ds['AwakeEventData']['BTV.TT41.412426_FWHEEL']
                          ['Acquisition']['position'][0]),
        # 2nd halo filter
        'filter_2':
            lambda ds: (ds['AwakeEventData']['BTV.TT41.412442_FWHEEL']
                          ['Acquisition']['position'][0]),
        # in mJ
        'laser_energy_1':
            lambda ds: 1e3 * _try_get(
                ds, ['AwakeEventData', 'EMETER03', 'Acq', 'value', 0], np.nan),
        # in mJ
        'laser_energy_2':
            lambda ds: 1e3 * _try_get(
                ds, ['AwakeEventData', 'EMETER04', 'Acq', 'value', 0], np.nan),

        'bct_time':
            lambda ds:
                60 * 60 * 2 * 10 ** 9 + int(1e9 * datetime.strptime(
                    ds['AwakeEventData']['TT41.BCTF.412340']
                      ['Acquisition']['acqTime'][0].decode(),
                    '%Y/%m/%d %H:%M:%S.%f').timestamp()),

        # gas gradient
        'density_timestamp':
            lambda ds: (ds['AwakeEventData']['TSG41.AWAKE-DENSITY-DATA']
                          ['ValueAcquisition']['floatValue'][0]),

        'density_ups':
            lambda ds: (ds['AwakeEventData']['TSG41.AWAKE-DENSITY-DATA']
                          ['ValueAcquisition']['floatValue'][1]),

        'density_downs':
            lambda ds: (ds['AwakeEventData']['TSG41.AWAKE-DENSITY-DATA']
                          ['ValueAcquisition']['floatValue'][2]),

        'density_grad':
            lambda ds: (ds['AwakeEventData']['TSG41.AWAKE-DENSITY-DATA']
                          ['ValueAcquisition']['floatValue'][3]),

        # camera settings, do not change the following; set automatically
        'cam_time': None,
        'core_1_exposure': None, 'core_2_exposure': None,
        'halo_1_exposure': None, 'halo_2_exposure': None
    },

    'radius_processing': {
        'angle_count': 300,
        'above_noise': 600,  # values above 500 are likely not noise
        'log_filter': lambda log: medfilt2d(log, 7),
        'neg_filter': lambda neg: gaussian_filter(neg, 4),
        'neg_threshold': .99,  # minimal negativity
        'rad_filter': lambda radii: radii
    },

    'setup_1': {
        # camera, length of one pixel
        'core': ('BOVWA.07TCC4.AWAKECAM07', .044),
        'halo': ('BOVWA.02TCC4.CAM10', 1 / 30),
        'core_trafo': None,
        'halo_trafo': lambda d: np.fliplr(d),
        'shift': np.array([8, 10]) - np.array([-1.73046533,  0.03112679])
    },
    'setup_2': {
        # camera, length of one pixel
        'core': ('BOVWA.02TCC4.AWAKECAM02', 30 / 960),
        'halo': ('BOVWA.04TCC4.CAM12', .04),
        'core_trafo': lambda d: np.fliplr(np.rot90(d)),
        'halo_trafo': lambda d: np.flipud(d),
        'shift': np.array([190, 700]) - np.array([20.66364606,  -5.04329489])
    }
}
