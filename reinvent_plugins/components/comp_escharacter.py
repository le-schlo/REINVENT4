from __future__ import annotations

__all__ = ["escharacter"]

from typing import List
import logging
import sys

# sys.path.append('/tmp/InvEnT_container/') #ToDo: Automatically detect top level dir!
from reinvent.utils.Calculator.FMO import homo_lumo_mwfn, homo_lumo_mwfn_quick

import numpy as np
from pydantic.dataclasses import dataclass

from .component_results import ComponentResults
from .add_tag import add_tag
from reinvent.scoring.utils import suppress_output
import contextlib
from ..normalize import normalize_smiles
import os

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
    calculation_mode: List[str]
    aggregation_mode: List[str]  # formula or multiwfn_thresholds
    dir4tempfiles: List[str]
    use_gfn2: List[bool]
    path_to_xtb: List[str]
    path_to_multiwfn: List[str]

    # total expression fo multiwfn_quick is:
    # score= Singlet_param * [(S_overlap * s1_overlap) + (S_distance * s1_distance)] + Triplet_param * [(T_overlap * t1_overlap) + (T_distance * t1_distance)]
    # or in multiwfn_thresholds
    # score = [1*Singlet_param if s1_overlap < S_overlap else 0] + [1*Triplet_param if t1_overlap > T_overlap else 0]

    Singlet_param: List[float]
    S_overlap: List[float]
    S_distance: List[float]
    Triplet_param: List[float]
    T_overlap: List[float]
    T_distance: List[float]


@add_tag("__component")
class escharacter:
    def __init__(self, params: Parameters):
        self.smiles_type = "rdkit_smiles"
        self.run_mode = params.calculation_mode[0]
        self.aggregation_mode = params.aggregation_mode[0]
        self.tmp_dir = params.dir4tempfiles[0]
        self.path_to_xtb = params.path_to_xtb[0]
        self.path_to_multiwfn = params.path_to_multiwfn[0]
        self.use_gfn2 = params.use_gfn2[0]
        self.Singlet_param = params.Singlet_param[0]
        self.S_overlap = params.S_overlap[0]
        self.S_distance = params.S_distance[0]
        self.Triplet_param = params.Triplet_param[0]
        self.T_overlap = params.T_overlap[0]
        self.T_distance = params.T_distance[0]
        logger.info(f"Running HOMO-LUMO in {self.run_mode} mode")

    @normalize_smiles
    def __call__(self, smilies: List[str]) -> np.array:

        # ToDo: Parallelise but pay attention that each parallel job gets it's own working directory as
        result = []
        for smi in smilies:
            if self.run_mode == 'multiwfn':
                with contextlib.redirect_stdout(None):
                    overlap_s1, dist_s1, overlap_t1, dist_t1 = None, None, None, None
                    for t in [False, True]:
                        if t:
                            overlap_t1, dist_t1 = homo_lumo_mwfn(smi, triplet=t, use_gfn2=self.use_gfn2,
                                                                 dirname=self.tmp_dir, path_to_multiwfn=self.path_to_multiwfn,
                                                                 path_to_xtb=self.path_to_xtb)
                        else:
                            overlap_s1, dist_s1 = homo_lumo_mwfn(smi, triplet=t, use_gfn2=self.use_gfn2,
                                                                 dirname=self.tmp_dir, path_to_multiwfn=self.path_to_multiwfn,
                                                                 path_to_xtb=self.path_to_xtb)

            elif self.run_mode == 'multiwfn_quick':
                with contextlib.redirect_stdout(None):
                    overlap_s1, dist_s1, overlap_t1, dist_t1 = homo_lumo_mwfn_quick(smi, use_gfn2=self.use_gfn2,
                                                                                    dirname=self.tmp_dir, path_to_multiwfn=self.path_to_multiwfn,
                                                                                    path_to_xtb=self.path_to_xtb)

            elif self.run_mode == 'tblite':
                raise NotImplemented

            else:
                raise NotImplemented

            if self.aggregation_mode == 'formula':
                s_part = self.Singlet_param * ((self.S_overlap * overlap_s1) + (self.S_distance * dist_s1))
                t_part = self.Triplet_param * ((self.T_overlap * overlap_t1) + (self.T_distance * dist_t1))
                final_score = s_part + t_part

            elif self.aggregation_mode == 'threshold':
                s_part = 1 if overlap_s1 < self.S_overlap else 0  # CT character
                t_part = 1 if overlap_t1 > self.T_overlap else 0  # LE character
                final_score = (self.Singlet_param * s_part) + (self.Triplet_param * t_part)

            else:
                raise NotImplemented

            result.append(final_score)

        scores = result
        return ComponentResults([np.array(scores, dtype=float)])