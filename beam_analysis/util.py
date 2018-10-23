import numpy as np
import datetime
import os


class EventCorruptError(Exception):
    pass


def safe_join(base, relative):
    path = os.path.join(base, relative)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def cam_extract_bovwa(df, name):
    ext_img = df['AwakeEventData'][name]['ExtractionImage']
    if ext_img.attrs.get('exception'):
        raise EventCorruptError
    img = np.array(ext_img['imageRawData'])

    time = ext_img['imageTimeStamp'][0]
    exposure = df['AwakeEventData'][name]['PublishedSettings'][
        'exposureTime'][0]

    return img, (time, exposure)


def cam_extract_btv(df, name):
    ext_img = df['AwakeEventData'][name]['Image']
    if ext_img.attrs.get('exception'):
        raise EventCorruptError
    img = np.array(ext_img['imageSet'])
    img = img.reshape(ext_img['nbPtsInSet2'][0], ext_img['nbPtsInSet1'][0])
    x, y = ext_img['imagePositionSet1/'][0], ext_img['imagePositionSet2/'][0]
#     acq_time = ext_img['acqTime'][0]
#     time = datetime.datetime.strptime(
#         acq_time.decode(), '%Y/%m/%d %H:%M:%S.%f').timestamp()
#     time = int(time * 1e9) + 2 * 60 * 60 * 1e9  # utc vs local
#     exposure = None  # TODO: missing  
    
    return img, (x, y)


def mm_mgrid(px_width, shape, center=None):
    """ Get a mesh grid in mm coordinates. """
    if center is None:
        center = (0, 0)
    x = np.linspace(-center[1], shape[1] - center[1], shape[1]) * px_width
    y = np.linspace(-center[0], shape[0] - center[0], shape[0]) * px_width

    return np.meshgrid(x, y)
