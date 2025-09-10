import pandas as pd
import numpy as np
from reinvent.utils.Geometry.StructureGenerator import StructureGenerator
import os
import shutil
import time
from time import time as tm
from pathlib import Path
import subprocess
import numpy as np  # summation
import matplotlib.pyplot as plt  # plots
from scipy.signal import find_peaks  # peak detection
from rdkit import Chem
def excited_states( smiles: str,
                    dirname: str,
                    path_to_xtb: str,
                    path_to_stda: str,
                    delete_existing: bool = False,
                    maximum_waiting_time: int = 180,
                    use_stddft: bool = False,
                    use_gfnff: bool = False, #Default is gnf2
                    threshold: float = 0.00001,
                    intensity_at: int = 400
                    ):
    '''

    :param smiles:
    :param dirname: This directory gets created for temporary file creation. Should be a non existing directory in a temporary directory that exists.
    :param delete_existing: If True, the directory will be deleted if it exists. Caution!!
    :param path_to_xtb: Path to the xtb executable
    :param path_to_stda: Path to the xtb4stda, stda executable and to the parameters files for xtb4stda
    :param maximum_waiting_time: After this time in seconds, the calculation will be aborted
    :param use_stddft: Whether to use stddft or stda (Grimme proposes to use stda)
    :param use_gfnff: Whether to use gfnff or gfn2 for the geometry optimization
    :param intensity_at: Return intensity at this wavelength. Used for the suitabilty of a compound for a certain wavelength.
    :return:
    '''

    if os.path.isdir(dirname) and delete_existing == False:
        print(
            f"Warning: {dirname} exists already. Please set remove to true to delete it or choose another name for the directory.")
        print('Something')
        #return None, None
    if os.path.isdir(dirname) and delete_existing == True:
        print(
            f"Warning: Deleting existing directory {dirname}, as flag delete_existing is set to True. If you don't want this, you have 10 seconds to abort the process.")
        time.sleep(10)
        shutil.rmtree(dirname)

    mol = Chem.MolFromSmiles(smiles)
    charge = Chem.GetFormalCharge(mol)

    os.mkdir(dirname)
    os.chdir(dirname)

    try:
        if Chem.MolFromSmiles(smiles) is None:
            print(f"Error: {smiles} is not a valid SMILES string.")
            return np.nan, np.nan, np.nan, np.nan
        _, _ = StructureGenerator().get_structure(smiles, write_xyz=True)
        geometry_optimized = False
        start_time = tm()
        if use_gfnff:
            os.system(f"{path_to_xtb} input.xyz --opt --gfnff --chrg {charge} --verbose > xtb_output.log")
        else:
            os.system(f"{path_to_xtb} input.xyz --opt --gfn2 --chrg {charge} --verbose > xtb_output.log")

        while geometry_optimized == False and tm() - start_time < maximum_waiting_time:
            files = os.listdir(dirname)
            if 'xtbopt.xyz' in files:
                geometry_optimized = True
            if tm() - start_time > maximum_waiting_time:
                os.chdir('..')
                shutil.rmtree(dirname)
                print(f"Geometry optimization took longer than {maximum_waiting_time} seconds. Aborting.")
                #return None, None
        os.environ['XTB4STDAHOME'] = path_to_stda
        os.system(f"{path_to_stda}/xtb4stda xtbopt.xyz --chrg {charge} --verbose > xtb_output_xtb4stda.log")
        if use_stddft:
            os.system(f"{path_to_stda}/stda -xtb -e 7 -rpa --chrg {charge} > output_stda.out")
        else:
            os.system(f"{path_to_stda}/stda -xtb -e 7 --chrg {charge} > output_stda.out")

        os.system("grep -A1 state output_stda.out | tail -1 > S1Ex.dat")
        with open('S1Ex.dat') as data:
            datastr = data.read()
            temp = datastr.split()
            try:
                e_s1 = float(temp[1])
            except:
                e_s1 = np.nan

        peaks, height, intensity = get_uv_data('output_stda.out', threshold=threshold, intensity_at=intensity_at)
        os.chdir('..')
        shutil.rmtree(dirname)

        return peaks, height, intensity, e_s1
    except:
        os.chdir('..')
        shutil.rmtree(dirname)
        print(f"Error in the calculation of {smiles}.")
        return np.nan, np.nan, np.nan, np.nan

def roundup(x, nm_plot):
    # round to next 10 or 100
    if nm_plot:
        return x if x % 10 == 0 else x + 10 - x % 10
    else:
        return x if x % 100 == 0 else x + 100 - x % 100
def rounddown(x, nm_plot):
    # round to next 10 or 100
    if nm_plot:
        return x if x % 10 == 0 else x - 10 - x % 10
    else:
        return x if x % 100 == 0 else x - 100 - x % 100
def gauss(a, m, x, w):
    # calculation of the Gaussian line shape
    # a = amplitude (max y, intensity)
    # x = position
    # m = maximum/meadian (stick position in x, wave number)
    # w = line width, FWHM
    return a * np.exp(-(np.log(2) * ((m - x) / w) ** 2))
def get_uv_data(filename: str='output.out', threshold=0.00001, intensity_at=400):

    # global constants
    found_uv_section = False  # check for uv data in out


    specstring_start = 'state    eV      nm       fL        Rv(corr)'

    export_delim = " "  # delimiter for data export
    save_spectrum = False

    startx = 100
    endx = 600
    export_spectrum = False
    nm_plot = False
    w_nm = 20
    w_wn = 1000

    # plot config section - configure here
    nm_plot = True  # wavelength plot /nm if True, if False wave number plot /cm-1
    show_single_gauss = False  # show single gauss functions if True
    show_single_gauss_area = False  # show single gauss functions - area plot if True
    show_conv_spectrum = True  # show the convoluted spectra if True (if False peak labels will not be shown)
    show_sticks = True  # show the stick spectra if True
    label_peaks = True  # show peak labels if True
    minor_ticks = True  # show minor ticks if True
    spectrum_title = "Absorption spectrum"  # title
    spectrum_title_weight = "bold"  # weight of the title font: 'normal' | 'bold' | 'heavy' | 'light' | 'ultrabold' | 'ultralight'
    y_label = "intensity"  # label of y-axis
    x_label_wn = r'energy /cm$^{-1}$'  # label of the x-axis - wave number
    x_label_nm = r'$\lambda$ /nm'  # label of the x-axis - nm

    # global lists
    statelist = list()  # mode
    energylist = list()  # energy cm-1
    intenslist = list()  # fosc
    gauss_sum = list()  # list for the sum of single gaussian spectra = the convoluted spectrum for cm-1

    # open a file
    # check existence
    try:
        with open(filename, "r") as input_file:
            for line in input_file:
                # start exctract text
                if specstring_start in line:
                    # found UV data in orca.out
                    found_uv_section = True
                    continue
                if found_uv_section:
                    try:
                        statelist.append(int(line.strip().split()[0]))
                        energylist.append(float(line.strip().split()[1])*8065.54429)
                        intenslist.append(float(line.strip().split()[3]))
                    except IndexError:
                        break

    # file not found -> exit here
    except:
        return np.nan, np.nan, np.nan

    # no UV data in orca.out -> exit here
    if found_uv_section == False:
        print(f"'{specstring_start}'" + "not found in" + f"'{filename}'")
        return np.nan, np.nan, np.nan

    if nm_plot:
        # convert wave number to nm for nm plot
        energylist = [1 / wn * 10 ** 7 for wn in energylist]
        w = w_nm  # use line width for nm axis
    else:
        w = w_wn  # use line width for wave number axis

    # plotrange must start at 0 for peak detection
    plt_range_x = np.arange(0, max(energylist) + w * 3, 1)

    # plot single gauss function for every frequency freq
    # generate summation of single gauss functions
    for index, wn in enumerate(energylist):
        gauss_sum.append(gauss(intenslist[index], plt_range_x, wn, w))
    # y values of the gauss summation /cm-1
    plt_range_gauss_sum_y = np.sum(gauss_sum, axis=0)
    if intensity_at < len(plt_range_gauss_sum_y):
        intensity = plt_range_gauss_sum_y[intensity_at]
    else:
        intensity = 0
    # find peaks scipy function, change height for level of detection
    peaks, peak_heights = find_peaks(plt_range_gauss_sum_y, height=threshold)
    return peaks, peak_heights, intensity
