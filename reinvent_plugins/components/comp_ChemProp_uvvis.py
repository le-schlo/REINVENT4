"""Compute scores with ChemProp

[[component.ChemProp.endpoint]]
name = "ChemProp Score"
weight = 0.7

# component specific parameters
param.checkpoint_dir = "ChemProp/3CLPro_6w63"
param.rdkit_2d_normalized = true
param.target_column = "dG"

# transform
transform.type = "reverse_sigmoid"
transform.high = -5.0
transform.low = -35.0
transform.k = 0.4

# In case of multiclass models add endpoints as needed and set the target_column
"""

from __future__ import annotations

__all__ = ["ChemPropuvvis"]
from typing import List
import logging

import chemprop
import numpy as np
from pydantic.dataclasses import dataclass

from .component_results import ComponentResults
from .add_tag import add_tag
from reinvent.scoring.utils import suppress_output
from ..normalize import normalize_smiles
import pandas as pd
import os
import contextlib

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

    checkpoint_dir: List[str]
    checkpoint4featuregen: List[str]
    tmp_dir: List[str]


@add_tag("__component")
class ChemPropuvvis:
    def __init__(self, params: Parameters):
        logger.info(f"Using ChemProp version {chemprop.__version__}")

        # needed in the normalize_smiles decorator
        # FIXME: really needs to be configurable for each model separately
        self.smiles_type = "rdkit_smiles"
        self.parameterset = params
        self.keeps = np.array([0])
        self.number_of_endpoints = len(self.keeps)

    @normalize_smiles
    def __call__(self, smilies: List[str]) -> np.array:
        df = pd.DataFrame()
        df['smiles'] = smilies
        df['solvent'] = ['CC#N']*len(smilies)

        testpath = self.parameterset.tmp_dir[0] + '/batch_chempropuvvis.csv'
        predpath = self.parameterset.tmp_dir[0] + '/test_tddft_preds.csv'
        featpath = self.parameterset.tmp_dir[0] + '/features_test.csv'
        model4featuregen = self.parameterset.checkpoint4featuregen[0]

        df.to_csv(testpath, index=False)
        arguments = [
            '--test_path', testpath,
            '--preds_path', predpath,
            '--checkpoint_dir', model4featuregen,        ###Needs to be the path to the feature generator model
            '--number_of_molecules', '1'
        ]

        args = chemprop.args.PredictArgs().parse_args(arguments)
        _ = chemprop.train.make_predictions(
            args=args)

        tddft_preds_file_name = predpath
        if os.path.exists(tddft_preds_file_name):
            df = pd.read_csv(tddft_preds_file_name)
            df.rename(columns={'energy_max_osc': 'peakwavs_max_tddft_pred'}, inplace=True)
            if smilies[0] != df['smiles'][0]:
                raise Exception('smiles mismatch')
            feature_cols = [x for x in df.columns if x not in ['smiles', 'solvent', 'peakwavs_max']]
            df['peakwavs_max_tddft_pred'] = 1240 / df['peakwavs_max_tddft_pred']  # convert to nm

            df[feature_cols].to_csv(featpath, index=False)
        else:
            raise Exception('TD-DFT predictions failed')

        with contextlib.redirect_stdout(None):
            arguments = [
                '--test_path', testpath,
                '--preds_path', '/dev/null',
                '--checkpoint_dir', self.parameterset.checkpoint_dir[0],
                '--number_of_molecules', '2',
                '--features_path', featpath
            ]

            args = chemprop.args.PredictArgs().parse_args(arguments)
            preds = chemprop.train.make_predictions(args=args, return_invalid_smiles=True)

        scores = np.array(preds).transpose()[self.keeps]
        scores[scores == "Invalid SMILES"] = np.nan

        os.remove(testpath)
        os.remove(predpath)
        os.remove(featpath)

        return ComponentResults(list(scores.astype(float)))