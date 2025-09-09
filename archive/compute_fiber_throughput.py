'''
measure_display_mirror.py
Generic interface for mirror measurement and output
Take measurements using interferometer, compute surface and display
1/9/2025 warrenbfoster
'''

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize
from matplotlib import cm
from scipy import interpolate, ndimage
from scipy.optimize import minimize
import pickle
import cv2 as cv
from matplotlib.widgets import EllipseSelector
from General_zernike_matrix import *
from tec_helper import *
from LFAST_TEC_output import *
from LFAST_wavefront_utils import *
from hcipy import *
from interferometer_utils import *
import os
from matplotlib import patches as mpatches
import csv
from plotting_utils import *
#%%
#Mirror parameters
in_to_m = 25.4e-3

OD = 32*in_to_m #Outer mirror diameter (m)
ID = 3*in_to_m #Central obscuration diameter (m)
clear_aperture_outer = 0.5*OD
clear_aperture_inner = 0.5*ID
remove_coef = [0,1,2,4]
fiber_diameters = np.linspace(1e-6,100e-6,100)
#Set up the Zernike fitting matrix to process the h5 files
Z = General_zernike_matrix(44,int(clear_aperture_outer * 1e6),int(clear_aperture_inner * 1e6))

#%% Set up path to folder holding measurements
base_path = 'C:/Users/warrenbfoster/OneDrive - University of Arizona/Documents/LFAST/mirrors/'
list_of_paths = ['M10/20250729/1/','M19/20250826/1/']

test_holder = []
for path in list_of_paths:
    save_subfolder = base_path + path
    test = {'path': save_subfolder,
            'title':save_subfolder.split('/')[-4]}
    test_holder.append(test)

for test in test_holder:
    save_subfolder = test['path']
    data_holder = []
    coord_holder = []
    ID_holder = []
    for file in os.listdir(save_subfolder):
        if file.endswith(".h5"):
            print('Now processing ' + file)
            data, circle_coord, ID = measure_h5_circle(save_subfolder + file, use_optimizer=True)
            data_holder.append(data)
            coord_holder.append(circle_coord)
            ID_holder.append(ID)

    avg_circle_coord = np.mean(coord_holder, axis=0)
    avg_ID = np.mean(ID_holder)

    #Based on the defined pupil, process the measurements
    increased_ID_crop = 1.25

    wf_maps = []

    for data in data_holder:
        wf_maps.append(format_data_from_avg_circle(data, avg_circle_coord, clear_aperture_outer, clear_aperture_inner*increased_ID_crop, Z, normal_tip_tilt_power=True)[1])

    surface = np.flip(np.mean(wf_maps, 0), 0)

    test.update({'surface': surface})
    #np.save(fig_path + 'surface_v' + str(step_num) + '.npy', surface)

perfect_surface = surface.copy()
perfect_surface[~np.isnan(perfect_surface)] = 0
test = {'path': 'Imagination',
        'title': 'Diffraction limited',
        'surface': perfect_surface}
test_holder.append(test)
#%%
for test in test_holder:
    M,C = get_M_and_C(test['surface'], Z)

    updated_surface = remove_modes(M,C,Z,remove_coef)
    wf_foc, throughput, x_foc, y_foc = propagate_wavefront(updated_surface, clear_aperture_outer, clear_aperture_inner,
                                                       Z, use_best_focus=True, fiber_diameters = fiber_diameters)
    test.update({'throughput': throughput})

#%%
for test in test_holder:
    smoothed_throughput = ndimage.gaussian_filter1d(test['throughput'], sigma=5, mode='nearest')
    plt.plot(np.multiply(fiber_diameters,1e6), smoothed_throughput,label=test['title'])
plt.legend()
plt.xlabel('Fiber diameter (um)')
plt.ylabel('Normalized throughput')
plt.title('Throughput of different fiber diameters')
plt.show()

