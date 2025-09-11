from __future__ import annotations

__all__ = ["excitedstate"]

from typing import List
import logging
import sys

import pandas as pd
# sys.path.append('/tmp/InvEnT_container/') #ToDo: Automatically detect top level dir!
from reinvent.utils.Calculator.STDA import excited_states

import numpy as np
from pydantic.dataclasses import dataclass

from .component_results import ComponentResults
from .add_tag import add_tag
from ..normalize import normalize_smiles
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

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
    tmp_dir: List[str]
    path_to_stda: List[str]
    path_to_xtb: List[str]
    maximum_waiting_time: List[int] #180,
    use_stddft: List[bool] #False,
    use_gfnff: List[bool] #False,  # Default is gnf2
    target_property: List[str] # 'lambda_max'

@add_tag("__component")
class excitedstate:
    def __init__(self, params: Parameters):
        self.smiles_type = "rdkit_smiles"
        self.tmp_dir = params.tmp_dir[0]
        self.path_to_stda = params.path_to_stda[0]
        self.path_to_xtb = params.path_to_xtb[0]
        self.maximum_waiting_time = params.maximum_waiting_time[0]
        self.use_stddft = params.use_stddft[0]
        self.use_gfnff = params.use_gfnff[0]
        self.target_property = params.target_property
        self.num_endpoints = len(self.target_property)

        logger.info(f"Running excited state calculation with sTDA by Grimme et al.")

    @normalize_smiles
    def __call__(self, smilies: List[str]) -> np.array:

        results=[]
        s1_results = []
        s1t1_results = []
        lambda_max_results = []
        intensity_results = []

        for i, smi in enumerate(smilies):

            peaks, height, intensity, e_s1 = excited_states(smiles=smilies[i],
                                                      dirname=self.tmp_dir,
                                                      use_gfnff=self.use_gfnff,
                                                      path_to_stda=self.path_to_stda,
                                                      path_to_xtb=self.path_to_xtb,
                                                      maximum_waiting_time=self.maximum_waiting_time,
                                                      use_stddft=self.use_stddft,
                                                      threshold=0.00001,
                                                      intensity_at=400)

            lambda_max = np.max(peaks)
            lambda_max_results.append(lambda_max)

        if 'lambda_max' in self.target_property:
            results.append(np.array(lambda_max_results, dtype=float))

        scores = results
        return ComponentResults(scores)
