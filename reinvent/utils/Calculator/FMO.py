from reinvent.utils.Geometry.StructureGenerator import StructureGenerator
import os
import shutil
import time
from time import time as tm
from pathlib import Path
import subprocess
from rdkit import Chem

def extract_homo_lumo_indices(logfile='xtb_output.log'):
    homo_idx, lumo_idx = None, None
    with open(logfile, 'r') as file:
        lines = file.readlines()
        found_it = False

        for idx, line in enumerate(lines):
            # Look for HOMO and LUMO lines in xtb output
            if '* Orbital Energies and Occupations' in line:
                found_it = True
                start_of_occupation = idx
                continue
            if found_it == True and idx > start_of_occupation+5:
                parts = line.split()
                if parts[-1] == '(HOMO)':
                    homo_idx = int(parts[0])
                if parts[-1] == '(LUMO)':
                    lumo_idx = int(parts[0])
                    break

    return homo_idx, lumo_idx
def generate_multiwfn_input(homo_idx, lumo_idx, dir=None):
    if dir:
        mwfn_inp_path = Path.joinpath(dir/'multiwfn_input.txt')
    else:
        mwfn_inp_path = 'multiwfn_input.txt'
    with open(mwfn_inp_path, 'w') as file:
        file.write(f"100\n")           # Orbital analysis based on overlap
        file.write(f"11\n")            # Select overlap centroid calculation
        file.write(f"{homo_idx}\n")    # HOMO index
        file.write(f"{lumo_idx}\n")    # LUMO index
        file.write("n\n")
        file.write("0,0\n")
        file.write("0\n")               # Finish analysis
        file.write("q\n")              # Quit
    print("Multiwfn input script generated.")
def run_multiwfn(multiwfn_path):
    try:
        os.system(f'{multiwfn_path} molden.input -silent < multiwfn_input.txt > mwfn.out')

        with open('mwfn.out', 'r') as f:

            centroid_distance = None
            overlap_norm = None
            overlap_square = None

            for line in f.readlines():
                if "Centroid distance between the two orbitals" in line:
                    centroid_distance = float(line.split()[-2])
                elif "Overlap integral of norm" in line:
                    overlap_norm = float(line.split()[-1])
                elif "Overlap integral of square" in line:
                    overlap_square = float(line.split()[-1])
                    return centroid_distance, overlap_norm, overlap_square
    except Exception as e:
        print(f"Error during Multiwfn execution: {e}")
def homo_lumo_mwfn(
        smiles: str,
        path_to_xtb: str,
        path_to_multiwfn: str,
        dirname: str,
        additional_geometry_opt: bool = True, # if set to true, optimization engine is gfnff, Also implicit solvation is considered with alpb acetonitrile
        use_gfn2=False,
        maximum_waiting_time: int = 180,
        delete_existing: bool = False
):
    '''
    Runs xtb and Multiwfn from commandline. Important: As the dirname will be removed, make sure this is not an important directory by mistake!
    This function will not optimise a triplet structure but will compute s0 and t1 homo lumo from the s0 geometry to speed things up.

    :param smiles:
    :param dirname:
    :param path_to_xtb:
    :param path_to_multiwfn:
    :return:
    '''

    mol = Chem.MolFromSmiles(smiles)
    charge = Chem.GetFormalCharge(mol)

    if os.path.isdir(dirname) and delete_existing == False:
        print(
            f"Warning: {dirname} exists already. Please set remove to true to delete it or choose another name for the directory.")
        print('Something')
        return None, None, None, None
    if os.path.isdir(dirname) and delete_existing == True:
        print(
            f"Warning: Deleting existing directory {dirname}, as flag delete_existing is set to True. If you don't want this, you have 10 seconds to abort the process.")
        time.sleep(10)
        shutil.rmtree(dirname)

    dirname = Path(dirname)
    os.mkdir(str(dirname))
    s1_dirname = str(dirname / 's1')
    t1_dirname = str(dirname / 't1')
    os.mkdir(s1_dirname)
    os.mkdir(t1_dirname)
    os.chdir(s1_dirname)

    if use_gfn2:
        xtb_method = '--gfn 2'
    else:
        xtb_method = '--gfnff'

    try:
        atoms, coordinates = StructureGenerator().get_structure(smiles, write_xyz=True)
        if additional_geometry_opt:
            geometry_optimized = False
            start_time = tm()
            os.system(f"{path_to_xtb} input.xyz --opt {xtb_method} --chrg {charge} --verbose> xtb_output.log")

            while geometry_optimized == False and tm() - start_time < maximum_waiting_time:
                files = os.listdir(s1_dirname)
                if 'xtbopt.xyz' in files:
                    geometry_optimized = True
                if tm() - start_time > maximum_waiting_time:
                    os.chdir('..')
                    shutil.rmtree(dirname)
                    print(f"Geometry optimization took longer than {maximum_waiting_time} seconds. Aborting.")
                    return None, None, None, None
            os.rename('xtbopt.xyz', 'input.xyz')

        shutil.copyfile('input.xyz', str(Path(t1_dirname) / 'input.xyz'))
        os.system(f"{path_to_xtb} input.xyz --molden --chrg {charge} --verbose > xtb_output.log")
        homo_idx, lumo_idx = extract_homo_lumo_indices()
        generate_multiwfn_input(homo_idx, lumo_idx)
        dist_s1, hl_norm_s1, hl_square_s1 = run_multiwfn(path_to_multiwfn)
        os.chdir('..')

        os.chdir(t1_dirname)
        shutil.rmtree(s1_dirname)
        os.system(f"{path_to_xtb} input.xyz --molden --chrg {charge} --uhf 2 --verbose > xtb_output.log")
        homo_idx, lumo_idx = extract_homo_lumo_indices()
        generate_multiwfn_input(homo_idx, lumo_idx)
        dist_t1, hl_norm_t1, hl_square_t1 = run_multiwfn(path_to_multiwfn)
        os.chdir('..')
        shutil.rmtree(t1_dirname)
        os.chdir('..')
        shutil.rmtree(dirname)
        return hl_norm_s1, dist_s1, hl_norm_t1, dist_t1