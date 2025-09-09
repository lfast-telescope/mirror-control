from hcipy import *
import numpy as np
import matplotlib.pyplot as plt
from General_zernike_matrix import *
from LFAST_wavefront_utils import *

in_to_m = 25.4e-3
OD = 31.9*in_to_m #Outer mirror diameter (m)
ID = 3*in_to_m #Central obscuration diameter (m)
clear_aperture_outer = 0.47*OD
clear_aperture_inner = ID

output_foc_holder = []
throughput_holder = []
focal_length = 2.537
# Fiber parameters
fiber_radius = 18e-6 / 2
fiber_subtense = fiber_radius / focal_length
center_wavelength = 580e-9 #center wavelength for propagations
bw = np.multiply([1,2,5,10,20,50,100,200,500],1e-9) #FW bandwidth for broadband propagations
number_waves = [5,5,5,7,7,7,11,11,11]
distance = focal_length

defocus_positions = np.linspace(-100e-6,100e-6,21)

grid = make_pupil_grid(1024, OD)
focal_grid = make_focal_grid(15, 42, spatial_resolution=632e-9 / OD)
pixel_size = 2.4e-6
small_grid_diameter = np.max(focal_grid.x) - np.min(focal_grid.x)
number_pixels = small_grid_diameter/pixel_size
sensor_grid = make_pupil_grid(int(np.ceil(number_pixels)), small_grid_diameter)
prop = FraunhoferPropagator(grid, focal_grid, focal_length=focal_length)
eemask = Apodizer(evaluate_supersampled(make_circular_aperture(fiber_subtense * 2), focal_grid, 8))

fresnel = FresnelPropagator(focal_grid, distance)

#%%
base_path = 'C:/Users/warrenbfoster/OneDrive - University of Arizona/Documents/LFAST/'
array_of_paths = [base_path + 'mirrors/M8/20241122/', base_path + 'mirrors/M8/20241203/']

#%%
use_path = base_path + 'mirrors/M8/20241203/'
data_holder = []
coord_holder = []
ID_holder = []

for file in os.listdir(use_path):
    data,circle_coord, ID = measure_h5_circle(use_path + file)
    data_holder.append(data)
    coord_holder.append(circle_coord)
    ID_holder.append(ID)

avg_circle_coord = np.mean(coord_holder, axis=0)
avg_ID = np.mean(ID_holder)

#Based on the defined pupil, process the measurements
increased_ID_crop = 1.25

wf_maps = []
#%%
for data in data_holder:
    wf_maps.append(format_data_from_avg_circle(data, avg_circle_coord, clear_aperture_outer, clear_aperture_inner*increased_ID_crop, Z, normal_tip_tilt_power=True)[1])

updated_surface = np.flip(np.mean(wf_maps, 0), 0)

#%%
grid_for_h5 = make_pupil_grid(updated_surface.shape, OD)
inter_h5 = make_linear_interpolator_separated(updated_surface.ravel(),grid_for_h5)

#%%
surface_field = inter_h5(grid)
surface_field[np.isnan(surface_field)] = 0
opd = Field(surface_field * 1e-6, grid)
mirror = SurfaceApodizer(opd, 2)
output_figs = True

for num, bandwidth in enumerate(bw):
    wavelengths = np.linspace(center_wavelength-bandwidth/2,center_wavelength+bandwidth/2,number_waves[num])
    wavefront_holder = [[] for i in np.arange(len(wavelengths))]

    sensor_holder = []
    distance_holder = []

    for index, wave in enumerate(wavelengths):
        wf = Wavefront(make_obstructed_circular_aperture(OD*0.98, clear_aperture_inner / clear_aperture_outer)(grid),wave)
        wf.total_power = 1
        wf_opd = mirror.forward(wf)
        wf_foc = prop.forward(wf_opd)

        fresnel.distance = defocus_positions[0]
        wf_start = fresnel.forward(wf_foc)
        wavefront_holder[index] = wf_start.copy()

    for index, shift_distance in enumerate(np.hstack([0,np.diff(defocus_positions)])):
        fresnel.distance = shift_distance
        irradiance_at_this_shift = []
        for wf_num, wf in enumerate(wavefront_holder):
            wf_next = fresnel.forward(wf.copy())
            wavefront_holder[wf_num] = Wavefront(wf_next.electric_field,wavelength=wf_next.wavelength)
            sensor_inter = make_linear_interpolator_separated(wf_next.electric_field)
            sensor_wf = Wavefront(sensor_inter(sensor_grid), wf_next.wavelength)
            irradiance_at_this_shift.append(deravel(sensor_wf.power))
        broadband_irradiance = np.mean(irradiance_at_this_shift, axis=0)
        sensor_holder.append(broadband_irradiance)

        if output_figs:
            plt.imshow(broadband_irradiance)
            plt.title(str(round(bandwidth*1e9)) + 'nm bandwidth PSF at ' + str(np.round(defocus_positions[index]*1e6)) + 'um shift')
            plt.show()

    np.save(base_path + 'propagations/broadband/sensor_irradiance_bw=' + str(int(bandwidth*1e9)) + 'nm.npy', sensor_holder)

#%%
np.save(base_path + 'sensor_measurements.npy', sensor_holder)
np.save(base_path + 'fresnel_propagations.npy', wf_holder)
np.save(base_path + 'distances.npy', distance_holder)

#%%
yiyang_result = np.load('C:/Users/warrenbfoster/Downloads/phase_estimate.npy')
yiyang_scaled = -yiyang_result*0.8 / (2*np.pi)
yiyang_scaled[yiyang_scaled==0]=np.nan

vmin = np.min([np.nanmin(yiyang_scaled),np.nanmin(updated_surface)])
vmax = np.max([np.nanmax(yiyang_scaled),np.nanmax(updated_surface)])

fig,ax = plt.subplots(1,2,)
im0= ax[0].imshow(updated_surface,vmin=vmin,vmax=vmax)
im1= ax[1].imshow(yiyang_scaled,vmin=vmin,vmax=vmax)

for i in [0,1]:
    ax[i].set_xticks([])
    ax[i].set_yticks([])
    ax[i].set_xlabel('um',x=1.2)
    fig.colorbar(im0, ax=ax[i], shrink=0.5)

ax[0].set_title('Interferometer measurement')
ax[1].set_title('Phase diversity reconstruction')
plt.tight_layout()
fig.suptitle('Phase diversity reconstruction of simulated PSF',y=0.9)

plt.show()
