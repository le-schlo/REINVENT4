from __future__ import annotations

__all__ = ["rigid"]

from typing import List
import logging
from pydantic.dataclasses import dataclass
from collections import OrderedDict
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from .component_results import ComponentResults
from .add_tag import add_tag
from ..normalize import normalize_smiles

logger = logging.getLogger("reinvent")

@add_tag("__parameters")
@dataclass
class Parameters:
    """Parameters for the scoring component

    Note that all parameters are always lists because components can have
    multiple endpoints and so all the parameters from each endpoint is
    collected into a list.  This is also true in cases where there is only one
    endpoint.
    """
    n_confs: List[int]
    upper_energy: List[float]
    rms: List[float]


'''
Implementation according to https://doi.org/10.1021/acs.jcim.6b00565
'''

@add_tag("__component")
class rigid:
    def __init__(self, params: Parameters):
        self.smiles_type = "rdkit_smiles"
        self.n_confs = params.n_confs[0]
        self.upper_energy = params.upper_energy[0]
        self.rms = params.rms[0]

    @normalize_smiles
    def __call__(self, smilies: List[str]) -> np.array:
        scores = []
        for i, smi in enumerate(smilies):
            try:
                mol = Chem.MolFromSmiles(smi)
                molecule = Chem.AddHs(mol)
                conformerIntegers = []
                # Embed and optimise the conformers with MMFF94
                conformers = AllChem.EmbedMultipleConfs(molecule, self.n_confs, numThreads=8, randomSeed=42)
                optimized_and_energies = AllChem.MMFFOptimizeMoleculeConfs(molecule, maxIters=600, nonBondedThresh=100.0)

                EnergyDictionaryWithIDAsKey = {}
                FinalConformersToUse = {}

                # Only keep the conformers which were successfully fully optimised
                for conformer in conformers:
                    optimised, energy = optimized_and_energies[conformer]
                    if optimised == 0:
                        EnergyDictionaryWithIDAsKey[conformer] = energy
                        conformerIntegers.append(conformer)

                # Keep the lowest energy conformer
                lowestenergy = min(EnergyDictionaryWithIDAsKey.values())

                for k, v in EnergyDictionaryWithIDAsKey.items():
                    if v == lowestenergy:
                        lowestEnergyConformerID = k

                FinalConformersToUse[lowestEnergyConformerID] = lowestenergy

                molecule = AllChem.RemoveHs(molecule)
                matches = molecule.GetSubstructMatches(molecule, uniquify=False)

                maps = [list(enumerate(match)) for match in matches]

                # Loop over conformers other than the lowest energy one
                for conformerID in EnergyDictionaryWithIDAsKey.keys():
                    okayToAdd = True

                    # Loop over reference conformers already added to list
                    for finalconformerID in FinalConformersToUse.keys():

                        # Calculate the best RMS of this conformer with the reference conformer in the list
                        RMS = AllChem.GetBestRMS(molecule, molecule, finalconformerID, conformerID, maps)
                        # Do not add if a match is found with a reference conformer
                        # (i.e., heavy-atom RMS within 1.0 Å)
                        if RMS < self.rms:
                            okayToAdd = False
                            break

                    # Add the conformer if the RMS is greater than 1.0 for every reference conformer
                    if okayToAdd:
                        FinalConformersToUse[conformerID] = EnergyDictionaryWithIDAsKey[conformerID]
                        # Chem.MolToXYZFile(molecule, filename=f'/tmp/libinvent/workdir/flexibility/used_confs/mol_{conformerID}', confId=conformerID)

                # Sort the conformers by energy
                sortedDictionary = OrderedDict(sorted(FinalConformersToUse.items(), key=lambda t: t[1]))
                energyList = [val for val in sortedDictionary.values()]

                descriptor = 0
                # Subtract the lowest energy found in the ordered list
                relativeEnergies = np.array(energyList) - energyList[0]

                # Only look at the energies of conformers other than the global minimum
                for energy in relativeEnergies[1:]:
                    # Optimized lower and upper energy limits for conformer energy
                    if 0 <= energy < self.upper_energy:
                        descriptor += 1

                scores.append(descriptor)
            except:
                scores.append(999)
        return ComponentResults([np.array(scores, dtype=float)])
