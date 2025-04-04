

import numpy as np
from pymatgen.core import Structure


def get_order_per_layer(struct, atoms_per_layer, d_0=0.005, nt=False, atoms_a = ['Ba', 'Sm'], atoms_b = ['Ni', 'Ti'], atoms_o = ['O']):
    '''
    function: gives array to sort by unit cell layers and c from a pymatgen structure (same function as read_het_structs without reading the structure and using pymatgen structure)
    last updated: 12.04.2024

    struct: pymatgen structure object
    atoms_per_layer: number of atoms per layer
    d_0: offset by which it still counts the atom to belong to the top part of the unit cell
    filetype: OUTCAR, CONTCAR, POSCAR, ...

    returns: structure and array with layer indices (shape: number of layers* atoms per layer)
    '''

    # get types of atoms
    chem_symbols = np.array([specie.value for specie in struct.species])
    # print(chem_symbols)
    all_A = np.nonzero((chem_symbols == atoms_a[0]) + (chem_symbols == atoms_a[1]))[0]
    all_B = np.nonzero((chem_symbols == atoms_b[0]) + (chem_symbols == atoms_b[1]))[0]
    all_O = np.nonzero(chem_symbols == atoms_o[0])[0]
    print(f'number of A: {len(all_A)}, number of B: {len(all_B)}, number of O: {len(all_O)}')

    # get coordinates
    coords = struct.frac_coords
    if nt == False:
        coords[coords[:, 2] <= d_0, 2] = coords[coords[:, 2] <= d_0, 2] + np.repeat(1,
                                                                                    len(coords[coords[:, 2] <= d_0, 2]))
    else:
        mask_i = coords[:, 2] >= 1 - d_0
        coords[mask_i, 2] = coords[mask_i, 2] - np.repeat(1, len(coords[mask_i, 2]))

    # sort
    sort_A = np.argsort(coords[all_A, 2])
    sort_B = np.argsort(coords[all_B, 2])
    sort_O = np.argsort(coords[all_O, 2])
    n_layers = int(len(struct) / atoms_per_layer)
    # print(chem_symbols[all_A[sort_A]])
    # print(chem_symbols[all_B[sort_B]])

    sorted_layers = np.zeros([int(n_layers), int(atoms_per_layer)], dtype=int)
    uc_per_layer = int(atoms_per_layer / 5)
    for i in range(n_layers):
        temp_layer = np.zeros(atoms_per_layer, dtype=int)
        temp_layer[0:uc_per_layer] = all_A[sort_A][uc_per_layer * i:(uc_per_layer * i + uc_per_layer)]
        temp_layer[uc_per_layer:2 * uc_per_layer] = all_B[sort_B][uc_per_layer * i:(uc_per_layer * i + uc_per_layer)]
        temp_layer[2 * uc_per_layer:(2 * uc_per_layer + 3 * uc_per_layer)] = all_O[sort_O][3 * uc_per_layer * i:(
                3 * uc_per_layer * i + 3 * uc_per_layer)]
        sorted_layers[i] = temp_layer
    
    if atoms_per_layer > 5:
        for i,layer in enumerate(sorted_layers):
            if chem_symbols[layer[0]] != chem_symbols[layer[1]]:
                print(f'WARNING: layer {i} has two different A atoms')
            if chem_symbols[layer[2]] != chem_symbols[layer[3]]:
                print(f'WARNING: layer {i} has two different B atoms')

    print(f"number of layers: {n_layers}")
    return sorted_layers