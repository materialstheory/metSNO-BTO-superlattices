import numpy as np
from scipy.constants import e


def myround(x, prec=1, base=.5):
  """
  round to 0.5 or 1

  last updated 10.03.2023
  """
  return np.around(base * np.round(x/base),prec)

def get_pol_from_charges(pos_pol, pos_unpol, born_charges, volume):
    """
    pos_pol, pos_unpol: polar and unpolar positions in direct coordinates (not fractional!), as matrix with xyz as columns and atoms as rows
    born_charges: born effective charges as matrix, shape (number atoms, 3, 3)
    volume: volume of cell
    returns polarization in units of muC/cm^2
    """

    delta = pos_pol - pos_unpol

    
    polarisation = []
    polarisation = [np.matmul(delta[i,:],born_charges[i,:]) for i in range(len(delta))] # matric multiplication of displacement* Born effective charges
    polarisation = np.sum(polarisation, axis = 0) # sum over all elements
    polarisation *= e/volume*1e16*1e6 # calc. polarization in muC/cm^2
    
    return polarisation


def pol_from_layers(het_struct, layer_het, a_atoms = ['Ba', 'Sm'], b_atoms = ['Ti', 'Ni'], o_atoms = ['O']):
    """
    calculates polarization per layer from formal charges
    last edited: 14.06.2023

    het_struct: structure
    layer_het:  array with indices of atoms per layer, shape: number of layers*atoms per layer
    figs: control figures (atoms in-plane per layer), True: they are plotted

    returns: dict for whole layer and for only SNO layers (2 layers = 1 unit cell), form: index of layer, volume, c-lattice, polarization, fractional dipole, dipole

    """

    '''initialize stuff'''

    # structure
    het_struct_unpolar = het_struct.copy()  # cell all layers
    c_long = het_struct.cell.cellpar()[2]  # c_axis of entire cell
    all_c_short = np.zeros(len(layer_het))  # to store all
    unpol_coords = het_struct_unpolar.get_scaled_positions()  # to store positions of unpolar reference structure
    pol_coords = het_struct.get_scaled_positions()  # to store positions of polar structure


    '''FORMAL CHARGES'''
    m_Ba = np.zeros([3, 3])
    np.fill_diagonal(m_Ba, 2)
    m_O = np.zeros([3, 3])
    np.fill_diagonal(m_O, -2)
    m_Ti = np.zeros([3, 3])
    np.fill_diagonal(m_Ti, 4)
    m_Ni = np.zeros([3, 3])
    np.fill_diagonal(m_Ni, 3)
    m_Sm = np.zeros([3, 3])
    np.fill_diagonal(m_Sm, 3)

    i_SNO = np.array(
        [i for i in range(len(layer_het)) if np.any(np.array(het_struct[layer_het[i]].get_chemical_symbols()) == a_atoms[1])])

    ################
    '''BTO LAYERS'''
    ################

    # len_res = len(i_BTO)
    len_res = len(layer_het)
    # To store results
    all_pol = np.full([len_res, 3, 3], np.nan)
    row_index = np.full(len_res, np.nan)
    all_V = np.zeros(len_res)
    all_c_short = np.zeros(len_res)  # to store all
    # plt.figure()

    for i, layer in enumerate(layer_het):
        # index = i_BTO[i]
        index = i
        # print(f'BTO {i}')
        '''SMALL UNIT CELL'''
        # LOWER END OF LAYER UNIT CELL
        if index == 0:  # first unit cell: lower end is on top of unit cell (negative)
            lower_end_bool = (np.array(het_struct_unpolar[layer_het[-1]].get_chemical_symbols()) == a_atoms[1]) | (
                    np.array(het_struct_unpolar[layer_het[-1]].get_chemical_symbols()) == a_atoms[0])
            # print(het_struct_unpolar[layer_het[-1]][lower_end_bool].get_positions()[0,2])
            if min(het_struct_unpolar[layer_het[-1]][lower_end_bool].get_positions()[:, 2]) < c_long/2:
                # closer to lower than upper end
                lower_end = min(het_struct_unpolar[layer_het[-1]][lower_end_bool].get_positions()[:, 2])
            else:
                lower_end = max(het_struct_unpolar[layer_het[-1]][lower_end_bool].get_positions()[:, 2]) - c_long
            # print(lower_end)
        else:  # lower end = position of Ba or Sm in unit cell below
            lower_end_bool = (np.array(het_struct_unpolar[layer_het[index - 1]].get_chemical_symbols()) == a_atoms[1]) | (
                    np.array(het_struct_unpolar[layer_het[index - 1]].get_chemical_symbols()) == a_atoms[0])
            lower_end = max(het_struct_unpolar[layer_het[index - 1]][lower_end_bool].get_positions()[:, 2])

        # UPPER END OF UNiT CELL: position of Ba or Sm
        upper_end_bool = (np.array(het_struct_unpolar[layer].get_chemical_symbols()) == a_atoms[1]) | (
                np.array(het_struct_unpolar[layer].get_chemical_symbols()) == a_atoms[0])
        upper_end = het_struct_unpolar[layer][upper_end_bool].get_positions()[0, 2]
        if upper_end <= c_long / 2 and index == len(layer_het) - 1:
            upper_end += c_long

        # calculate smaller c-lattice vector
        c_short = upper_end - lower_end
        all_c_short[i] = c_short

        '''UNPOLAR POSITIONS'''
        if index == 0:  # change positions to small (=layer) unit cell
            unpol_coords[layer, 2] = (het_struct_unpolar[layer].get_positions()[:, 2] - lower_end)
            unpol_coords[layer, 2] = unpol_coords[layer, 2] / c_short
        elif index == len(layer_het) - 1:
            unpol_coords[layer, 2] = (unpol_coords[layer, 2] * c_long)
            mask_i = np.nonzero(unpol_coords[layer, 2] <= c_long / 2)
            unpol_coords[layer[mask_i], 2] += c_long
            unpol_coords[layer, 2] = (unpol_coords[layer, 2] - lower_end) / c_short
        else:
            unpol_coords[layer, 2] = (unpol_coords[layer, 2] * c_long - lower_end) / c_short
        unpol_coords[layer] = myround(unpol_coords[layer], prec=2, base=0.25)


        # print(unpol_coords[layer])
        '''POLAR POSITIONS'''
        if index == 0:  # change positions to small unit cell
            pol_coords[layer, 2] = (het_struct_unpolar[layer].get_positions()[:, 2] - lower_end)
            pol_coords[layer, 2] = pol_coords[layer, 2] / c_short
        elif index == len(layer_het) - 1:
            pol_coords[layer, 2] = (pol_coords[layer, 2] * c_long)
            mask_i = pol_coords[layer, 2] <= c_long / 2
            pol_coords[layer[mask_i], 2] += c_long
            pol_coords[layer, 2] = (pol_coords[layer, 2] - lower_end) / c_short
        else:
            pol_coords[layer, 2] = (pol_coords[layer, 2] * c_long - lower_end) / c_short
        unpol_coords[layer, 2] = myround(pol_coords[layer, 2].copy(), base=0.5) # SOLUTION: makes sure along c no atoms are placed at 0.25 or 0.75 in centrosymm. structure

        # short cell
        cell = np.array([[het_struct.cell.cellpar()[0], 0, 0], [0, het_struct.cell.cellpar()[1], 0], [0, 0, c_short]])
        # volume
        all_V[i] = cell[0, 0] * cell[1, 1] * cell[2, 2]

        '''POLARIZATION'''
        chem_symbols = np.array(het_struct_unpolar[layer].get_chemical_symbols())

        # write formal charges in same order as structure
        charges = np.zeros([len(chem_symbols), 3, 3])
        if np.any(chem_symbols == a_atoms[0]):
            charges[chem_symbols == a_atoms[0]] = np.array([m_Ba for _ in charges[chem_symbols == a_atoms[0]]])
            charges[chem_symbols == b_atoms[0]] = np.array([m_Ti for _ in charges[chem_symbols == b_atoms[0]]])
            charges[chem_symbols == o_atoms[0]] = np.array([m_O for _ in charges[chem_symbols == o_atoms[0]]])
        else:
            charges[chem_symbols == b_atoms[1]] = np.array([m_Ni for _ in charges[chem_symbols == b_atoms[1]]])
            charges[chem_symbols == a_atoms[1]] = np.array([m_Sm for _ in charges[chem_symbols == a_atoms[1]]])
            charges[chem_symbols == o_atoms[0]] = np.array([m_O for _ in charges[chem_symbols == o_atoms[0]]])

        # absolute positions in small cell
        pos_polar = np.matmul(pol_coords[layer], cell)
        pos_unpolar = np.matmul(unpol_coords[layer], cell)

        # polarization in layer: CELL 1
        cell_1 = np.array([[1., 0.], [0, 0], [0, 1], [1, 1], [0.5, 0], [0.5, 1], [0.75, 0.25],
                           [0.75, 0.75]])  # unit cell 1 (always 2 BTO unit cells per layer)
        i_1 = np.any([(unpol_coords[layer][:, 0:2] == cell).all(1) for cell in cell_1], axis=0)
        all_pol[i, 0] = get_pol_from_charges(pos_polar[i_1], pos_unpolar[i_1], charges[i_1], all_V[i] / 2)


        # polarization in layer: CELL 2
        cell_2 = np.array([[0.5, 0.5], [1, 0.5], [0, 0.5], [0.25, 0.25], [0.25, 0.75]])
        i_2 = np.any([(unpol_coords[layer][:, 0:2] == cell).all(1) for cell in cell_2], axis=0)
        all_pol[i, 1] = get_pol_from_charges(pos_polar[i_2], pos_unpolar[i_2], charges[i_2], all_V[i] / 2)

        # TOTAL polarization layer
        all_pol[i, 2] = get_pol_from_charges(pos_polar, pos_unpolar, charges, all_V[i])

        row_index[i] = i

    dict_res_BTO = {"name": row_index,
                    "volume": all_V,
                    "c": all_c_short,
                    "pol": all_pol,
                    "pol coords": pol_coords,
                    "unpol coords": unpol_coords}

    ################
    '''SNO LAYERS'''
    ################

    # structure
    het_struct_unpolar = het_struct.copy()  # cell all layers
    c_long = het_struct.cell.cellpar()[2]  # c_axis of entire cell
    all_c_short = np.zeros(len(layer_het))  # to store all
    unpol_coords_sno = het_struct_unpolar.get_scaled_positions()  # to store positions of unpolar reference structure
    pol_coords_sno = het_struct.get_scaled_positions()  # to store positions of polar structure

    layer_SNO = layer_het[i_SNO].reshape(int(layer_het[i_SNO].shape[0] / 2), int(layer_het[i_SNO].shape[1] * 2))
    i_SNO = i_SNO.reshape(int(i_SNO.shape[0] / 2), 2)


    # results
    all_pol_sno = np.full([len(i_SNO), 3], np.nan)
    row_index_sno = np.full(len(i_SNO), np.nan)
    all_V_sno = np.zeros(len(i_SNO))
    all_c_short_sno = np.zeros(len(i_SNO))  # to store all
    for i, layer in enumerate(layer_SNO):
        del pos_polar, pos_unpolar
        i_low = i_SNO[i, 0]
        i_high = i_SNO[i, 1]

        '''SMALL UNIT CELL'''
        # LOWER END OF LAYER UNIT CELL
        if i_low == 0:  # first unit cell: lower end is on top of unit cell (negative)
            lower_end_bool = (np.array(het_struct_unpolar[layer_het[-1]].get_chemical_symbols()) == a_atoms[1]) | (
                    np.array(het_struct_unpolar[layer_het[-1]].get_chemical_symbols()) == a_atoms[0])
            # print(het_struct_unpolar[layer_het[-1]][lower_end_bool].get_positions()[0,2])
            if min(het_struct_unpolar[layer_het[-1]][lower_end_bool].get_positions()[:, 2]) < c_long / 2:
                # closer to lower than upper end
                lower_end = min(het_struct_unpolar[layer_het[-1]][lower_end_bool].get_positions()[:, 2])
            else:
                lower_end = max(het_struct_unpolar[layer_het[-1]][lower_end_bool].get_positions()[:, 2]) - c_long
        else:  # lower end = position of Ba or Sm in unit cell below
            lower_end_bool = (np.array(het_struct_unpolar[layer_het[i_low - 1]].get_chemical_symbols()) == a_atoms[1]) | (
                    np.array(het_struct_unpolar[layer_het[i_low - 1]].get_chemical_symbols()) == a_atoms[0])
            lower_end = max(het_struct_unpolar[layer_het[i_low - 1]][lower_end_bool].get_positions()[:, 2])
        # UPPER END OF UNiT CELL: position of Ba or Sm
        if i == len(i_SNO) - 1:
            upper_end_bool = (np.array(het_struct_unpolar[layer].get_chemical_symbols()) == a_atoms[1]) | (
                    np.array(het_struct_unpolar[layer].get_chemical_symbols()) == a_atoms[0])
            upper_end = min(het_struct_unpolar[layer][upper_end_bool].get_positions()[:, 2])
            if upper_end < c_long / 2:
                upper_end += c_long
            else:
                upper_end = max(het_struct_unpolar[layer][upper_end_bool].get_positions()[:, 2])
        else:
            upper_end_bool = (np.array(het_struct_unpolar[layer].get_chemical_symbols()) == a_atoms[1]) | (
                    np.array(het_struct_unpolar[layer].get_chemical_symbols()) == a_atoms[0])
            upper_end = max(het_struct_unpolar[layer][upper_end_bool].get_positions()[:, 2])
        # calculate smaller c-lattice vector
        c_short = upper_end - lower_end
        all_c_short_sno[i] = c_short

        '''UNPOLAR POSITIONS'''
        if i_low == 0:
            unpol_coords_sno[layer, 2] = (het_struct_unpolar[layer].get_positions()[:, 2] - lower_end)
            unpol_coords_sno[layer, 2] = unpol_coords_sno[layer, 2] / c_short
        elif i == len(i_SNO) - 1:
            unpol_coords_sno[layer, 2] = (unpol_coords_sno[layer, 2] * c_long)
            mask_i = unpol_coords_sno[layer, 2] <= c_long / 2
            unpol_coords_sno[layer[mask_i], 2] += c_long
            unpol_coords_sno[layer, 2] = (unpol_coords_sno[layer, 2] - lower_end) / c_short
        else:
            unpol_coords_sno[layer, 2] = (unpol_coords_sno[layer, 2] * c_long - lower_end) / c_short
        unpol_coords_sno[layer] = myround(unpol_coords_sno[layer], prec=2, base=0.25)

        '''POLAR POSITIONS'''
        if i_low == 0:
            pol_coords_sno[layer, 2] = (het_struct_unpolar[layer].get_positions()[:, 2] - lower_end)
            pol_coords_sno[layer, 2] = pol_coords_sno[layer, 2] / c_short
        elif i == len(i_SNO) - 1:
            pol_coords_sno[layer, 2] = (pol_coords_sno[layer, 2] * c_long)
            mask_i = pol_coords_sno[layer, 2] <= c_long / 2
            pol_coords_sno[layer[mask_i], 2] += c_long
            pol_coords_sno[layer, 2] = (pol_coords_sno[layer, 2] - lower_end) / c_short
        else:
            pol_coords_sno[layer, 2] = (pol_coords_sno[layer, 2] * c_long - lower_end) / c_short

        # short cell
        cell = np.array([[het_struct.cell.cellpar()[0], 0, 0], [0, het_struct.cell.cellpar()[1], 0], [0, 0, c_short]])
        # volume
        all_V_sno[i] = cell[0, 0] * cell[1, 1] * cell[2, 2]

        '''POLARIZATION'''
        # sort unpolar coords
        chem_symbols = np.array(het_struct_unpolar[layer].get_chemical_symbols())
        charges = np.zeros([len(chem_symbols), 3, 3])

        charges[chem_symbols == b_atoms[1]] = np.array([m_Ni for _ in charges[chem_symbols == b_atoms[1]]])
        charges[chem_symbols == a_atoms[1]] = np.array([m_Sm for _ in charges[chem_symbols == a_atoms[1]]])
        charges[chem_symbols == o_atoms[0]] = np.array([m_O for _ in charges[chem_symbols == o_atoms[0]]])

        # absolute positions in small cell

        pos_polar = np.matmul(pol_coords_sno[layer], cell)
        pos_unpolar = np.matmul(unpol_coords_sno[layer], cell)
        all_pol_sno[i] = get_pol_from_charges(pos_polar, pos_unpolar, charges, all_V_sno[i])

        row_index_sno[i] = i


    dict_res_SNO = {"name": row_index_sno,
                    "volume": all_V_sno,
                    "c": all_c_short_sno,
                    "pol": all_pol_sno,
                    "pol coords": pol_coords_sno,
                    "unpol coords": unpol_coords_sno}
    # plt.show()
    return dict_res_BTO, dict_res_SNO


def pol_from_layers_nt(het_struct, layer_het): 
    '''
    calculates polarization per layer from formal charges for structure without tilts
    last edited: 13.01.2024

    het_struct: structure
    layer_het:  array with indices of atoms per layer, shape: number of layers*atoms per layer

    returns: dict per layer, form: c-lattice, volume, polarization

    
    '''
    all_pols = np.zeros([len(layer_het), 3])
    all_vs = np.zeros(len(layer_het))
    row_index = []
    # initialize
    het_struct_unpolar = het_struct.copy() # cell all layers
    c_long = het_struct.cell.cellpar()[2] # c_axis of entire cell
    all_c_short = np.zeros(len(layer_het)) # to store all 
    unpol_coords =  het_struct_unpolar.get_scaled_positions() # to store positions of unpolar reference structure
    pol_coords = het_struct.get_scaled_positions() # to store positions of polar structure

    charges = np.zeros([len(layer_het[0]), 3, 3])
    m_ba = np.zeros([3, 3])
    np.fill_diagonal(m_ba, 2)
    m_o = np.zeros([3, 3])
    np.fill_diagonal(m_o, -2)
    m_ti = np.zeros([3, 3])
    np.fill_diagonal(m_ti, 4)
    m_sm = np.zeros([3, 3])
    np.fill_diagonal(m_sm, 3)
    m_ni= np.zeros([3, 3])
    np.fill_diagonal(m_ni, 3)

    # loop over layers
    for index, layer_i_het in enumerate(layer_het):
        # SMALL UNIT CELL
        # LOWER END OF LAYER UNIT CELL
        if index == 0: # first unit cell: lower end is on top of unit cell (negative)
            lower_end_bool = (np.array(het_struct_unpolar[layer_het[-1]].get_chemical_symbols())=='Sm') | (np.array(het_struct_unpolar[layer_het[-1]].get_chemical_symbols())=='Ba')
            lower_end = het_struct_unpolar[layer_het[-1]][lower_end_bool].get_positions()[:,2]-c_long
        else: # lower end = position of Ba or Sm in unit cell below
            lower_end_bool = (np.array(het_struct_unpolar[layer_het[index-1]].get_chemical_symbols())=='Sm') | (np.array(het_struct_unpolar[layer_het[index-1]].get_chemical_symbols())=='Ba')
            lower_end = het_struct_unpolar[layer_het[index-1]][lower_end_bool].get_positions()[:,2]

        # UPPER END OF UNiT CELL: position of Ba or Sm 
        upper_end_bool = (np.array(het_struct_unpolar[layer_i_het].get_chemical_symbols()) == 'Sm') | (np.array(het_struct_unpolar[layer_i_het].get_chemical_symbols()) == 'Ba')
        upper_end = het_struct_unpolar[layer_i_het][upper_end_bool].get_positions()[:, 2]

        # calculate smaller c-lattice vector
        c_short = upper_end-lower_end
        all_c_short[index] = c_short

        # UNPOLAR POSITIONS
        if index == 0:
            unpol_coords[layer_i_het,2] = (het_struct_unpolar[layer_i_het].get_positions()[:, 2] - lower_end)
            unpol_coords[layer_i_het,2] = myround(unpol_coords[layer_i_het,2] / c_short)
        else:
            unpol_coords[layer_i_het,2] = myround((unpol_coords[layer_i_het,2] * c_long - lower_end) / c_short)

        # POLAR POSITIONS
        if index == 0:
            pol_coords[layer_i_het,2] = (het_struct_unpolar[layer_i_het].get_positions()[:, 2] - lower_end)
            pol_coords[layer_i_het,2] = pol_coords[layer_i_het,2] / c_short
        else:
            pol_coords[layer_i_het,2] = (pol_coords[layer_i_het,2] * c_long - lower_end) / c_short

        # short cell
        cell = np.array([[het_struct.cell.cellpar()[0], 0, 0], [0, het_struct.cell.cellpar()[1], 0], [0, 0, c_short[0]]])
        # volume
        volume = cell[0, 0]*cell[1, 1]*cell[2, 2]

        chem_symbols = np.array(het_struct_unpolar[layer_i_het].get_chemical_symbols())
        # POLARIZATION
        if np.any(np.array(het_struct_unpolar[layer_i_het].get_chemical_symbols()) == 'Ba'): # only in Ba layers

            charges[chem_symbols=='Ba'] = np.array([m_ba for _ in charges[chem_symbols == 'Ba']])
            charges[chem_symbols=='Ti'] = np.array([m_ti for _ in charges[chem_symbols == 'Ti']])
            charges[chem_symbols=='O'] = np.array([m_o for _ in charges[chem_symbols == 'O']])

            row_index.append(f"{index}: BTO")
        else:
            row_index.append(f"{index}: SNO")
            charges[chem_symbols=='Sm'] = np.array([m_sm for _ in charges[chem_symbols == 'Sm']])
            charges[chem_symbols=='Ni'] = np.array([m_ni for _ in charges[chem_symbols == 'Ni']])
            charges[chem_symbols=='O'] = np.array([m_o for _ in charges[chem_symbols == 'O']])

        # absolute positions in small cell
        pos_polar = np.matmul(pol_coords[layer_i_het], cell)
        pos_unpolar = np.matmul(unpol_coords[layer_i_het], cell)

        # polarization in layer
        polarisation = get_pol_from_charges(pos_polar, pos_unpolar, charges, volume)

        all_vs[index] = volume
        all_pols[index] = polarisation


    df = {"c-axis": all_c_short,
          "volume": all_vs,
          "pol.": all_pols}
    return df