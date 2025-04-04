from scipy.ndimage import gaussian_filter1d
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_bandgap(dos, energy, cutoff=0.2, maxenergy=2.5):
    """
    Calculate the bandgap from the density of states (DOS) and energy values.

    Parameters:
    dos (numpy.ndarray): Array of density of states values.
    energy (numpy.ndarray): Array of energy values corresponding to the DOS.
    cutoff (float, optional): Threshold value below which the DOS is considered to be in the bandgap. Default is 0.2.
    maxenergy (float, optional): Maximum energy value to consider for the bandgap calculation. Default is 2.5.

    Returns:
    tuple: A tuple containing the bandgap value, the upper bound energy of the bandgap, and the lower bound energy of the bandgap.
           If no bandgap is found, returns (0, 0, 0).
    """

    # get the DOS values within the energy range
    dos_gap = dos[np.logical_and(energy < maxenergy, energy > -maxenergy)] 
    egap = energy[np.logical_and(energy< maxenergy, energy > -maxenergy)] 

    # find the energy values below the cutoff
    cut_energies = np.argwhere(dos_gap<cutoff)

    # only calculate bandgap if there are states below the cutoff
    if len(cut_energies) > 0:
        return egap[cut_energies[-1][0]]-egap[cut_energies[0][0]], egap[cut_energies[-1][0]], egap[cut_energies[0][0]]
    else:
        return 0, 0, 0

def get_summed_dos_for_layer(dos, layer, separate_spins = False, sigma = -1, orbital_keys = None):
    """
    Calculate the summed density of states (DOS) for a specified layer.

    Parameters:
    dos (dict): pymatgen DOS dictionary containing the density of states data. It should have keys 'energies' and 'pdos'.
    layer (list): A list of indices representing the layer for which the DOS should be summed.
    separate_spins (bool, optional): If True, the DOS for spin-up and spin-down will be calculated separately. Default is False.
    sigma (float, optional): If greater than 0, a Gaussian filter with this sigma value will be applied to the summed DOS. Default is -1.
    orbital_keys (list, optional): A list of orbital keys to consider for summing the DOS. If None, all orbitals will be considered. Default is None.

    Returns:
    numpy.ndarray: The summed DOS for the specified layer. If separate_spins is True, returns a 2D array with separate spin-up and spin-down DOS.
    """
    if separate_spins == False:
        summed_dos = np.zeros(len(dos['energies']))
    else:
        summed_dos = np.zeros((2, len(dos['energies'])))

    for i in layer:
        element_dos = dos["pdos"][i]

        if orbital_keys is None:
            orbital_keys = element_dos.keys()
        else:
            orbital_keys = orbital_keys

        for key in orbital_keys:
            if separate_spins == False:
                summed_dos += element_dos[key]["densities"]["1"]
                try:
                    summed_dos += element_dos[key]["densities"]["-1"]
                except:
                    pass
            else:
                summed_dos[0] += element_dos[key]["densities"]["1"]
                summed_dos[1] += element_dos[key]["densities"]["-1"]

    if sigma > 0:
        diff = [dos['energies'][i + 1] - dos['energies'][i] for i in range(len(dos['energies']) - 1)]
        avg_diff = sum(diff) / len(diff)
        summed_dos = gaussian_filter1d(summed_dos, sigma / avg_diff) 
    return summed_dos

def plot_dos_per_layer(fig_list, dos_list, labels, layer, sigma = 0.07, xlim = [-5, 5], ylim = [0, 15], name_start=' ', colors = sns.color_palette(["#355F84",  "#F8B56B", "#A90E3D"]), orbitals = None):
    """
    Plot the density of states (DOS) per layer.

    Parameters:
    fig_list (list): A list containing the figure and axes objects.
    dos_list (list): A list of pymatgen dictionaries containing DOS data. 
    labels (list): A list of labels for the DOS data.
    layer (list): A list of layers to plot.
    sigma (float, optional): The standard deviation for Gaussian smearing. Default is 0.07.
    xlim (list, optional): The x-axis limits for the plot. Default is [-5, 5].
    ylim (list, optional): The y-axis limits for the plot. Default is [0, 15].
    name_start (str, optional): A prefix for the labels in the legend. Default is ' '.
    colors (list, optional): A list of colors for the plot lines. Default is a seaborn color palette.
    orbitals (list, optional): A list of orbitals to include in the DOS calculation. Default is None.

    Returns:
    tuple: A tuple containing the figure and axes objects.
    """
    nlayers = len(layer)
    plt.rcParams.update({'font.size': 18})
    # fig, ax = plt.subplots(nrows = nlayers, sharex = True, subplot_kw={'aspect': 'auto'})

    fig = fig_list[0]
    ax = fig_list[1]

    fig.subplots_adjust(hspace=0)
    # fig.set_size_inches([7*cm, 17*cm])

    for i in (range(nlayers)):

        for_legend = []
        names = []

        for j, dosdict in enumerate(dos_list):
            dos_layer_i = get_summed_dos_for_layer(dosdict, layer[i], False, sigma=sigma, orbital_keys=orbitals) # sum dos for one layer
            nr = ax[nlayers-1-i].plot(np.array(dosdict['energies'])-dosdict["efermi"], dos_layer_i, '-', color = colors[j]) # add dos to plot in correct layer
            for_legend.append(nr)
            names.append(f'{name_start}{labels[j]}')

        ax[i].axvline(0, color='lightgray', linestyle='--')

    if fig.legends:
        # If a legend already exists, add the new lines to the existing legend
        handles = fig.legends[0].legendHandles
        labels  = [text.get_text() for text in fig.legends[0].texts]
        handles.extend([i[0] for i in for_legend])
        labels.extend(names)
        fig.legends[0].remove()
        lgd = fig.legend(handles, labels, loc='lower center', bbox_to_anchor=[0.55, 0.95], frameon=False, bbox_transform=ax[0].transAxes)
    else:
        # If no legend exists, create a new one
        lgd = fig.legend([i[0] for i in for_legend], names, loc='lower center', bbox_to_anchor=[0.55, 0.95], frameon=False, bbox_transform=ax[0].transAxes)

    ax[-1].set_xlabel('$E-E_f$ (eV)')
    ax[int(nlayers/2)].set_ylabel('DOS (arb. unit)')

    # TURN OFF Y AXIS LABELS
    [ax_i.yaxis.set_tick_params(labelleft=False) for ax_i in ax]
    [ax_i.set_yticks([]) for ax_i in ax]


    [ax_i.set_xlim(xlim) for ax_i in ax]
    [ax_i.set_ylim(ylim) for ax_i in ax]

    return fig, ax