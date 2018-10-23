import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import beam_analysis as ba
import h5py
import os
from pandas import DataFrame as df
from pandas import concat

btv  = ["TT41.BTV.412350", "TT41.BTV.412353", "TT41.BTV.412426"]

def Gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))
    
def BTV_extract_sigma(data, x_axe, y_axe):
    
    """ Find a reasonable Gauss fit to the BTV images """
    
    data = ba.measure.medfilt(data) # apply median filter
    
    Alpha = 1  #set a treshold for binary conversion
    Data_proc = np.where(data < Alpha * np.mean(data), 0.0, 1.0) # convert image to binary form
    
    kernel1 = np.ones((10, 10), np.uint8) #define a cernel for for opening filter 
    opening = cv2.morphologyEx(Data_proc, cv2.MORPH_OPEN, kernel1) # 0 outside center, 1 inside

    kernel2 = np.ones((10, 10), np.uint8) # kernel for dilution
    opening = cv2.dilate(opening,kernel2,iterations = 1)
    opening_inverse = 1 - opening  
    
    Clean_figure = np.mean(data*opening_inverse)*opening_inverse + opening*data # use average background + originall center
    
    sum_x = np.sum(Clean_figure, 0) #integrate out 
    sum_y = np.sum(Clean_figure, 1)
    
    sum_x = sum_x - np.min(sum_x) #shift to zero minimum
    sum_y = sum_y - np.min(sum_y)
    
    mean_x = sum(sum_x * x_axe) / sum(sum_x)
    sigma_x = np.sqrt(sum(sum_x * (x_axe - mean_x)**2) / sum(sum_x))
    popt_x, pcov_x = curve_fit(Gauss, x_axe, sum_x, p0=[max(sum_x), mean_x, sigma_x]) #optimal parameters x

    mean_y = sum(sum_y * y_axe) / sum(sum_y)
    sigma_y = np.sqrt(sum(sum_y * (y_axe - mean_y)**2) / sum(sum_y))
    popt_y, pcov_y = curve_fit(Gauss, y_axe, sum_y, p0=[max(sum_y), mean_y, sigma_y]) #optimal parameters y
    
    return ([sum_x, sum_y], popt_x, popt_y)


def data_fall(file_range, btv_name, path):
    h5files = os.listdir(path)
    first, last = file_range
    
    metas = df([])
    errors = []
    
    for i, name in enumerate(h5files[first:last]):
        print("\rloading %.2f %%" % (100*(i)/(last-first)), end='')
        eb = os.path.join(path, name)
        file = h5py.File(eb, 'r')

        pic_test, (x_axe, y_axe) = ba.util.cam_extract_btv(file, btv_name)
        meta = ba.acquisition.get_meta(file, ba.setups.sept_2017.META)
        meta = df([meta])
        try: 
            ([suum_x, suum_y], popt_xx, popt_yy) = BTV_extract_sigma(pic_test, x_axe, y_axe)
            meta['sigma_x'] = (popt_xx[2])
            meta['sigma_y'] = (popt_yy[2])
        except:
            errors.append(i)
            meta['sigma_x'] = (0.0)
            meta['sigma_y'] = (0.0)
            print("-")     
        metas = concat([metas, meta], ignore_index=True)
    print(" completed") 
    return(metas, errors)