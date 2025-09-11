"""Compute scores with RDKit's functions"""

__all__ = ["conjugationdegree"]
from typing import List

import numpy as np
from rdkit import Chem
from pydantic.dataclasses import dataclass

from .component_results import ComponentResults
from ..normalize import normalize_smiles
from .add_tag import add_tag
from rdkit.Chem import rdMolDescriptors

@add_tag("__parameters")
@dataclass
class Parameters:
    mode: List[str] #Options are "fraction", or "largest_conjugated_fragment", default is "fraction"
    exclude_split_system: List[bool] #If True, exclude molecules that contain more than one conjugated system

@add_tag("__component")
class conjugationdegree:
    def __init__(self, params: Parameters):
        self.smiles_type = "rdkit_smiles"
        self.mode = params.mode[0]
        self.exclude_split_system = params.exclude_split_system[0]
    @normalize_smiles
    def __call__(self, smilies: List[str]) -> np.array:

        scores=[]
        for smi in smilies:
            mol = Chem.MolFromSmiles(smi)
            if not mol:
                conj_deg = 0
            else:
                conj_deg, conj_systems = self.calculate_conjugation_fraction(mol)

            if self.mode == "fraction":
                score = conj_deg
            elif self.mode == "largest_conjugated_fragment":
                if len(conj_systems) > 0:
                    score = max(conj_systems)
                else:
                    score = max(conj_systems)
            else:
                raise ValueError("Invalid mode")

            if self.exclude_split_system and len(conj_systems) > 1:
                score = 0

            scores.append(score)

        return ComponentResults([np.array(scores, dtype=float)])

    def get_conjugated_systems(self, mol):
        """
        Identify and quantify distinct conjugated systems in a molecule.
        """
        visited = set()
        conjugated_systems = []

        def dfs(atom_idx, system):
            """Depth-first search to find conjugated atoms."""
            visited.add(atom_idx)
            system.add(atom_idx)
            atom = mol.GetAtomWithIdx(atom_idx)
            for bond in atom.GetBonds():
                if bond.GetIsConjugated():
                    neighbor_idx = bond.GetOtherAtomIdx(atom_idx)
                    if neighbor_idx not in visited:
                        dfs(neighbor_idx, system)

        # Traverse all atoms to find conjugated systems
        for atom_idx in range(mol.GetNumAtoms()):
            if atom_idx not in visited:
                atom = mol.GetAtomWithIdx(atom_idx)
                if any(bond.GetIsConjugated() for bond in atom.GetBonds()):
                    system = set()
                    dfs(atom_idx, system)
                    conjugated_systems.append(system)

        # Measure size of each conjugated system
        system_sizes = []
        for system in conjugated_systems:
            system_bonds = set()
            for atom_idx in system:
                atom = mol.GetAtomWithIdx(atom_idx)
                for bond in atom.GetBonds():
                    if bond.GetIsConjugated():
                        begin_idx, end_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                        if begin_idx in system and end_idx in system:
                            system_bonds.add(bond.GetIdx())
            system_sizes.append(len(system_bonds))

        return system_sizes

    def calculate_conjugation_fraction(self, mol):
        """
        Calculate the fraction of conjugated bonds and the number of conjugated systems in a molecule.
        """
        if mol is None:
            raise ValueError("Invalid SMILES string")

        Chem.SanitizeMol(mol)
        total_bonds = mol.GetNumBonds()
        conjugated_systems = self.get_conjugated_systems(mol)

        total_conjugated_bonds = sum(conjugated_systems)
        fraction_conjugated = total_conjugated_bonds / total_bonds if total_bonds > 0 else 0

        return fraction_conjugated, conjugated_systems