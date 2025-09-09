"""
Utility functions for Zernike polynomial analysis
Ideally this should be agnostic of wavefront sensing method
"""
import numpy as np
import matplotlib.pyplot as plt

def return_coef(C,coef_array):
    #Print out the amplitudes of Zernike polynomials
    try:
        for coef in coef_array:
            print('Z' + str(coef) + ' is ' + str(round(C[2][coef] * 1000)) + 'nm')
    except:
        print('Z' + str(coef_array) + ' is ' + str(round(C[2][coef_array] * 1000)) + 'nm')

def return_zernike_nl(order, print_output = True):
    #Create list of n,m Zernike indicies
    n_holder = []
    l_holder = []
    coef = 0
    for n in np.arange(0,order+1):
        for l in np.arange(-n,n+1,2):
            if print_output:
                print('Z' + str(coef) + ': ' + str(n) + ', ' + str(l))
                coef += 1
            n_holder.append(n)
            l_holder.append(l)
            
    return n_holder,l_holder
   
def calculate_error_per_order(M,C,Z):
    n,l = return_zernike_nl(12,print_output = False)
    error = []
    
    remove_coef = [0,1,2,4]
    updated_surface = remove_modes(M,C,Z,remove_coef)
    
    vals = updated_surface[~np.isnan(updated_surface)]*1000
    rms = np.sqrt(np.sum(np.power(vals,2))/len(vals))

    coef = np.power(C[2]*1000,2)
    
    list_orders = np.arange(2,13)
    output_order = list_orders.copy()
    for order in list_orders:
        args = np.where(n==order)
        flag = np.where(args[0] < len(C[2]))
        if len(flag[0]) != 0:
            subset = coef[args]
            error.append(np.sqrt(np.sum(subset)))
            if False:
                print('Order ' + str(order) + ' has ' + str(error[-1]))
        elif len(output_order) == len(list_orders):
            output_order = np.arange(2,order)
                
    residual = np.sqrt(rms**2 - np.sum(np.power(error,2)))
    
    plt.bar(output_order,error)
    plt.bar(np.max(output_order+1),residual)
    plt.xlabel('Zernike order')
    plt.ylabel('rms wavefront error (nm)')
    plt.title('Zernike amplitude per order')
    plt.legend(['Fitted error', 'Higher order residual'])
    return error, residual

def get_M_and_C(avg_ref, Z):
    #Compute M and C surface height variables that are used for Zernike analysis
    #M is a flattened surface map; C is a list of Zernike coefficients
    M = avg_ref.flatten(), avg_ref
    C = Zernike_decomposition(Z, M, -1)  #Zernike fit
    return M, C

def Zernike_decomposition(Z,M,n):
    
    Z_processed = Z[0].copy()[:,0:n]
    Z_processed[np.isnan(Z_processed)] = 0  #replaces NaN's with 0's
    
    if type(M) == tuple:
        M_processed = M[0].copy()
    else:
        M_processed = M.copy().ravel()
    M_processed[np.isnan(M_processed)] = 0  #replaces NaN's with 0's
    
    Z_t = Z_processed.transpose() #
    
    A = np.dot(Z_t,Z_processed) #
    
    A_inv = np.linalg.inv(A) #
    
    B = np.dot(A_inv,Z_t)       #
    
    Zernike_coefficients = np.dot(B,M_processed) #Solves matrix equation:  Zerninke coefficients = ((Z_t*Z)^-1)*Z_t*M
    
    Surf = np.dot(Z[1][:,:,0:n],Zernike_coefficients) #Calculates best fit surface to Zernike modes
    
    return Surf.flatten(),Surf,Zernike_coefficients #returns the vector containing Zernike coefficients and the generated surface in tuple form       

def remove_modes(M,C,Z,remove_coef):
    #Remove Zernike modes from input surface map
    removal = M[1]*0
    for coef in remove_coef:
        term = (Z[1].transpose(2,0,1)[coef])*C[2][coef]
        removal += term
        
        if False: #Plot the removal terms for sanity
            plt.imshow(term)
            plt.title(coef+1)
            plt.show()
    Surf = M[1] - removal
    return Surf

def return_zernike_name(coef):
    #Return name of input Zernike polynomial; use numbering in "return_zernike_nl"
    if coef == 0:
        name = 'Piston'
    elif coef == 1:
        name = 'Tip'
    elif coef == 2:
        name = 'Tilt'
    elif coef == 3:
        name = 'Astigmatism'
    elif coef == 4:
        name = 'Defocus'
    elif coef == 5:
        name = 'Oblique Astigmatism'
    elif coef == 6:
        name = 'Trefoil'
    elif coef == 7:
        name = 'Vertical Coma'
    elif coef == 8:
        name = 'Horizontal Coma'
    elif coef == 9:
        name = 'Horizontal Trefoil'
    elif coef == 10:
        name = 'Quatrefoil'
    elif coef == 12:
        name = 'Primary spherical'
    elif coef == 14:
        name = 'Horizontal Quatrefoil'
    elif coef == 24:
        name = 'Secondary spherical'
    elif coef == 40:
        name = 'Tertiary spherical'
    elif coef == 60:
        name = 'Quatenary spherical'
    elif coef == 84:
        name = 'Quinary spherical'
    else:
        name = None
    return name
