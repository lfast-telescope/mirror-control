import numpy as np
from scipy.optimize import minimize, minimize_scalar
import matplotlib.pyplot as plt
from shared.General_zernike_matrix import General_zernike_matrix
from hcipy import (Wavefront, make_pupil_grid, make_focal_grid, Field, Apodizer,
                   SurfaceApodizer, FraunhoferPropagator, evaluate_supersampled,
                   make_circular_aperture, make_obstructed_circular_aperture)

#%% Wavefront analysis and propagation routines

def add_defocus(avg_ref, Z, amplitude=1):
    #Adds an "amplitude" amount of power to surface map; useful for focus optimization
    power = (Z[1].transpose(2, 0, 1)[4]) * amplitude
    left = np.min(avg_ref)
    right = np.max(avg_ref)

    if False:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(avg_ref, vmin=left, vmax=right)
        ax[1].imshow(avg_ref + power, vmin=left, vmax=right)
        plt.show()
    return avg_ref + power  #return 1D flattened surface and 2D surface


def propagate_wavefront(avg_ref, clear_aperture_outer, clear_aperture_inner, Z=None, use_best_focus=False,
                        wavelengths=[632.8e-9], fiber_diameters = None):
    #Define measured surface as a wavefront and do Fraunhofer propagation to evaluate at focal plane

    prop_ref = avg_ref.copy()
    prop_ref[np.isnan(prop_ref)] = 0

    if use_best_focus:
        if Z == None:
            Z = General_zernike_matrix(36, int(clear_aperture_radius * 1e6), int(ID * 1e6))

        prop_ref = optimize_focus(prop_ref, Z, clear_aperture_outer, clear_aperture_inner, wavelength=[1e-6])

    focal_length = clear_aperture_outer * 3.5
    grid = make_pupil_grid(500, clear_aperture_outer)

    if fiber_diameters is None:
        #Fiber parameters
        fiber_radius = 18e-6/2
        fiber_subtense = fiber_radius / focal_length
        focal_grid = make_focal_grid(15, 15, spatial_resolution=632e-9 / clear_aperture_outer)
        eemask = Apodizer(evaluate_supersampled(make_circular_aperture(fiber_subtense * 2), focal_grid, 8))
        prop = FraunhoferPropagator(grid, focal_grid, focal_length=focal_length)
    else:
        focal_grid = make_focal_grid(30, 20, spatial_resolution=632e-9 / clear_aperture_outer)
        prop = FraunhoferPropagator(grid, focal_grid, focal_length=focal_length)

    output_foc_holder = []
    throughput_holder = []

    if type(wavelengths) != list and type(wavelengths) != np.ndarray:
        wavelengths = [wavelengths]
        
    for wavelength in wavelengths:    

        wf = Wavefront(make_obstructed_circular_aperture(clear_aperture_outer,clear_aperture_inner/clear_aperture_outer)(grid),wavelength)
        wf.total_power = 1

        opd = Field(prop_ref.ravel() * 1e-6, grid)
        mirror = SurfaceApodizer(opd, 2)
        wf_opd = mirror.forward(wf)
        wf_foc = prop.forward(wf_opd)
        if fiber_diameters is None:
            throughput_holder.append(eemask.forward(wf_foc).total_power)
        else:
            EE_holder = []
            for diameter in fiber_diameters:
                fiber_radius = diameter/2
                EE_holder.append(compute_fiber_throughput(wf_foc, fiber_radius, focal_length, focal_grid))
            throughput_holder.append(EE_holder)
        size_foc = [int(np.sqrt(wf_foc.power.size))] * 2
        output_foc_holder.append(np.reshape(wf_foc.power, size_foc))

    if fiber_diameters is None:
        throughput = np.mean(throughput_holder)
    else:
        throughput = np.mean(throughput_holder, 0)

    if len(wavelengths) == 1:
        output_foc = output_foc_holder[0]
    else:
        output_foc = np.mean(output_foc_holder, 0)

    grid_dims = [int(np.sqrt(wf_foc.power.size))] * 2
    x_foc = 206265 * np.reshape(wf_foc.grid.x, grid_dims)
    y_foc = 206265 * np.reshape(wf_foc.grid.y, grid_dims)
    return output_foc, throughput, x_foc, y_foc
#%%
def compute_fiber_throughput(wf_foc, fiber_radius, focal_length, focal_grid):
    """Compute energy coupled into a circular fiber of given radius."""
    fiber_subtense = fiber_radius / focal_length
    fiber_mask = Apodizer(
        evaluate_supersampled(make_circular_aperture(fiber_subtense * 2), focal_grid, 8)
    )
    return fiber_mask.forward(wf_foc).total_power

def deravel(field,dims=None):
    if not dims:
        dims = [np.sqrt(field.size).astype(int)]*2
    new_shape = np.reshape(field,dims)
    return np.array(new_shape)
#%%

def find_best_focus(output_ref, Z, centerpoint, scale, num_trials, clear_aperture_outer, clear_aperture_inner):
    #Dumb focus compensation algorithm: just evaluate PSF with different applied defocus
    defocus_range = np.linspace(centerpoint - scale, centerpoint + scale, num_trials)
    throughput_holder = []
    for amplitude in defocus_range:
        title = 'Adding ' + str(round(amplitude,2)) + ' focus '
        defocused_avg = add_defocus(output_ref,Z,amplitude)
        output_foc,throughput,x_foc,y_foc = propagate_wavefront(defocused_avg,clear_aperture_outer,clear_aperture_inner)
        throughput_holder.append(throughput)         
    if True:
        plt.plot(defocus_range, throughput_holder)
        plt.xlabel('Defocus')
        plt.ylabel('Throughput')
    best_focus = defocus_range[np.argmax(throughput_holder)]
    return best_focus


def optimize_focus(updated_surface, Z, clear_aperture_outer, clear_aperture_inner, wavelength):
    #Focus optimizer
    res = minimize_scalar(objective_function,method='bounded', bounds=[-1,1], args = (updated_surface,Z,clear_aperture_outer,clear_aperture_inner, wavelength))
    defocused_surf= add_defocus(updated_surface,Z,amplitude=res.x)
    defocused_surf[np.isnan(defocused_surf)] = 0 

    return defocused_surf
    
def objective_function(amplitude,output_ref,Z,clear_aperture_outer,clear_aperture_inner, wavelength): #takes input, applies operations, returns a single number
    #Optimization function for minimization optimization: returns negative throughput in range [0-1]
    defocused_avg = add_defocus(output_ref, Z, amplitude)
    output_foc, throughput, x_foc, y_foc = propagate_wavefront(defocused_avg, clear_aperture_outer,
                                                               clear_aperture_inner, wavelengths=wavelength)

    if False:
        print('Amplitude is ' + str(amplitude) + ' and throughput is ' + str(throughput*100))
    return -throughput
