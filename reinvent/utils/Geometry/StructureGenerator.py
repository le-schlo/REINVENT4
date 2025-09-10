from rdkit import Chem
from rdkit.Chem import AllChem
from morfeus.conformer import ConformerEnsemble

class StructureGenerator():
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def get_structure(self, smiles: str, filename: str='input.xyz', write_xyz: bool=False):
        ensemble = ConformerEnsemble.from_rdkit(smiles, self.kwargs)
        ensemble.prune_rmsd()
        ensemble.sort()
        if write_xyz:
            if filename.split('.')[-1] != 'xyz':
                filename += '.xyz'
            ensemble.write_xyz(filename, ids=[1])

        lowest_conformer = ensemble[0]
        atoms = lowest_conformer.elements
        coordinates = lowest_conformer.coordinates
        return atoms, coordinates