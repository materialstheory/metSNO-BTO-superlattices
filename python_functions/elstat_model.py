# function to calculate the energy of the system
import numpy as np

# Constants
eps0 = 8.854e-12  # The permittivity of free space in C/V/m
e_charge = 1.6e-19  # The electronic charge in C

# default parameters 
d_sno_default = None
p_layer_default = 0.5
sigma_tf_default = 0
e_g_default = 0.0
epsr_default = 110
count_default = 0

def save_3d_array_to_txt(array, filename):
   
    # Open the file in write mode
    with open(filename, 'w') as f:
        # Write the shape of the array on the first line
        f.write(f"{array.shape[0]} {array.shape[1]} {array.shape[2]}\n")
        
        # Reshape the array to 2D
        reshaped_array = array.reshape(array.shape[0], -1)
        
        # Save the reshaped array data
        np.savetxt(f, reshaped_array)
    
    print(f"Array saved to {filename}")

def load_3d_array_from_txt(filename):
    # Open the file in read mode
    with open(filename, 'r') as f:
        # Read the shape information from the first line
        shape = tuple(map(int, f.readline().strip().split()))
        
        # Load the array data
        loaded_array = np.loadtxt(f).reshape(shape)
    
    return loaded_array

def getmin(energies):
    min = np.argmin(energies)
    min = np.unravel_index(min, energies.shape)
    return min



def energy_model_insulating(p_bto, a_bto, b_bto, p_sno,  a_sno, b_sno, d_bto, d_sno = d_sno_default, p_layer = p_layer_default ,sigma_tf = sigma_tf_default, e_g = e_g_default, epsr = epsr_default, count = count_default):

    '''
    
    :param p_bto: 
    :param P_bto: polarization of BaTiO3, in C/m^2 
    :param a: first parameter of the energy model, in eV.m^4.C^-2
    :param b: second parameter of the energy model, in eV.m^8.C^-4
    :param d: thickness of the layer, in m
    :param P_sno: polarization of SmNiO3, in C/m^2
    :param P_layer: layer polarization, in C/m^2
    :param sigma: screening charge, in C/m^2
    :return: total energy of the system, in eV
    '''


    
    # thickness
    if d_sno is None:
        d_sno = d_bto
        # print("BTO and SNO thickness are equal")
    
    # Calculate the energy
    
    # energy to form polarization 
    e_bto = a_bto * p_bto**2 + b_bto * p_bto**4
    e_sno = a_sno * p_sno**2 + b_sno * p_sno**4
    
    # print('p',e_bto, e_sno)
    # self screening in bto
    limit = (eps0 * epsr * e_g)/(p_layer - p_bto + p_sno)
    sigma_temp = (p_layer - p_bto + p_sno) - e_g * eps0 * epsr / d_bto
    if isinstance(d_bto, np.ndarray):
        sigma_gap = np.zeros(len(d_bto))
        inds = np.argwhere(np.logical_and(d_bto > limit, sigma_temp > 0))
        sigma_gap[inds] = sigma_temp[inds]
        # print('scr',sigma_gap)
    else:
        if d_bto > limit and sigma_temp > 0: # right thickness and positive screening charge
            count += 1
            sigma_gap = sigma_temp
            # print('scr', sigma_gap)
        else:
            sigma_gap = 0
    
    # electrostatic energy
    e_es = 1 / (2 * eps0 * epsr * e_charge) * (p_layer - p_bto + p_sno - (sigma_gap + sigma_tf))**2 * d_bto    

    # screening energy
    e_scr = sigma_gap / e_charge * e_g

    # total energy
    e_tot = e_bto*d_bto + e_sno*d_sno + e_es + e_scr

    # summary of energies
    energies = [e_bto*d_bto, e_sno*d_sno, e_es, e_scr]
    
    return e_tot, energies, count


# function to calculate the energy of the system
def energy_model_depol_field(p_bto, a_bto, b_bto, p_sno,  a_sno, b_sno, d_bto, lambda_dep, d_sno = d_sno_default, p_layer = p_layer_default, epsr = epsr_default, count = count_default):

    '''
    
    :param p_bto: 
    :param P_bto: polarization of BaTiO3, in C/m^2 
    :param a: first parameter of the energy model, in eV.m^4.C^-2
    :param b: second parameter of the energy model, in eV.m^8.C^-4
    :param d: thickness of the layer, in m
    :param P_sno: polarization of SmNiO3, in C/m^2
    :param P_layer: layer polarization, in C/m^2
    :param sigma: screening charge, in C/m^2
    :param d_sno: thickness of SmNiO3 layer, in m
    :param tilted: if the SmNiO3 layer is tilted
    :param epsr: relative permittivity of the system
    :param count: number of iterations
    :return: total energy of the system, in eV

    '''
    
    # thickness
    if d_sno is None:
        d_sno = d_bto
        # print("BTO and SNO thickness are equal")
    
    # Calculate the energy
    
    # energy to form polarization 
    e_bto = a_bto * p_bto**2 + b_bto * p_bto**4
    e_sno = a_sno * p_sno**2 + b_sno * p_sno**4
    
    
    # electrostatic energy
    e_es = 2* lambda_dep**2 / (eps0 * epsr * e_charge) * (p_layer - p_bto + p_sno)**2 / d_bto    

    # total energy
    e_tot = e_bto*d_bto + e_sno*d_sno + e_es
    # print('e_tot', e_tot)

    # separate energies
    energies = [e_bto*d_bto, e_sno*d_sno, e_es]
    
    return e_tot, energies, count

# function to calculate the energy of the system
def energy_model_depol_and_selfscreen(p_bto, a_bto, b_bto, p_sno,  a_sno, b_sno, d_bto, lambda_dep, d_sno = d_sno_default, p_layer = p_layer_default, e_g = e_g_default, epsr = epsr_default, count = count_default, gamma = None):

    '''
    param p_bto: polarization of BaTiO3, in C/m^2
    param a_bto: first parameter of the energy model of BaTiO3, in eV.m^4.C^-2
    param b_bto: second parameter of the energy model of BaTiO3, in eV.m^8.C^-4
    param p_sno: polarization of SmNiO3, in C/m^2
    param a_sno: first parameter of the energy model of SmNiO3, in eV.m^4.C^-2
    param b_sno: second parameter of the energy model of SmNiO3, in eV.m^8.C^-4
    param d_bto: thickness of BaTiO3, in m
    param lambda_dep: screening length, in m
    param d_sno: thickness of SmNiO3, in m
    param p_layer: layer polarization, in C/m^2
    param e_g: band gap, in eV
    param tilted: if the energy model is tilted
    param epsr: relative permittivity
    param count: counter for self screening
    param factor: factor instead of screening length
    '''



    
    # thickness
    if d_sno is None:
        d_sno = d_bto
        # print("BTO and SNO thickness are equal")

    # factor instead of screening length
    if gamma is None:
        gamma = 2 * lambda_dep / d_bto
    else:
        lambda_dep = gamma * d_bto / 2

    # self screening
    ptot = (p_layer - p_bto + p_sno)
    limit =  (eps0 * epsr * e_g) / (gamma**2 * ptot)
    # print(limit.shape, d_bto.shape)
    sigma_temp = ptot - eps0 * epsr * e_g * d_bto / 4 / lambda_dep**2
    if isinstance(d_bto, np.ndarray):
        sigma_gap = np.zeros(len(d_bto))
        inds = np.argwhere(np.logical_and(d_bto > limit, sigma_temp > 0))
        sigma_gap[inds] = sigma_temp[inds]
        count += len(inds)
    else:
        if d_bto >  limit and sigma_temp > 0:
            count += 1
            sigma_gap = sigma_temp
        else:
            sigma_gap = 0
    
    # Calculate the energy
    
    # energy to form polarization 
    e_bto = a_bto * p_bto**2 + b_bto * p_bto**4
    e_sno = a_sno * p_sno**2 + b_sno * p_sno**4
    
    
    # electrostatic energy
    e_es = 2 *lambda_dep**2 / (eps0 * epsr * e_charge) * (p_layer - p_bto + p_sno - sigma_gap)**2 / d_bto    


    # energy self screening
    e_scr = sigma_gap / e_charge * e_g

    # total energy
    e_tot = e_bto*d_bto + e_sno*d_sno + e_es + e_scr


    # separate energies
    energies = [e_bto*d_bto, e_sno*d_sno, e_es, e_scr]
    
    return e_tot, energies, count

def energy_model_vacuum_depol(p_bto, a_bto, b_bto, p_sno, a_sno, b_sno, d_bto, lambda_dep, d_sno=d_sno_default, p_layer=p_layer_default, p_vacuum=0, epsr=epsr_default, gamma=None):
    '''
    Calculate the energy model for a vacuum depolarization scenario.

    Parameters:
    p_bto (float): Polarization of BaTiO3, in C/m^2
    a_bto (float): First parameter of the energy model of BaTiO3, in eV.m^4.C^-2
    b_bto (float): Second parameter of the energy model of BaTiO3, in eV.m^8.C^-4
    p_sno (float): Polarization of SmNiO3, in C/m^2
    a_sno (float): First parameter of the energy model of SmNiO3, in eV.m^4.C^-2
    b_sno (float): Second parameter of the energy model of SmNiO3, in eV.m^8.C^-4
    d_bto (float): Thickness of BaTiO3, in m
    lambda_dep (float): Screening length, in m
    d_sno (float, optional): Thickness of SmNiO3, in m. Default is d_sno_default.
    p_layer (float, optional): Layer polarization, in C/m^2. Default is p_layer_default.
    p_vacuum (float, optional): Vacuum polarization, in C/m^2. Default is 0.
    epsr (float, optional): Relative permittivity. Default is epsr_default.
    gamma (float, optional): Factor instead of screening length. Default is None.

    Returns:
    tuple: Total energy, list of individual energies, and count
    '''
    
    # thickness
    if d_sno is None:
        d_sno = d_bto
        # print("BTO and SNO thickness are equal")

    # factor instead of screening length
    if gamma is None:
        gamma = 2 * lambda_dep / d_bto
    else:
        lambda_dep = gamma * d_bto / 2

    # total polarization
    ptot_inter = (p_layer - p_bto + p_sno)
    ptot_surf = (p_bto - p_vacuum)
    ptot = ptot_inter - ptot_surf
    
    # Calculate the energy
    
    # energy to form polarization 
    e_bto = a_bto * p_bto**2 + b_bto * p_bto**4
    e_sno = a_sno * p_sno**2 + b_sno * p_sno**4
    
    
    # electrostatic energy
    e_es = lambda_dep**2 / (eps0 * epsr * e_charge) * (ptot)**2 / d_bto    


    # total energy
    e_tot = e_bto*d_bto + e_sno*d_sno + e_es


    # separate energies
    energies = [e_bto*d_bto, e_sno*d_sno, e_es]
    
    return e_tot, energies