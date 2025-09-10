import pandas as pd
import numpy as np
from pygmo import hypervolume
from reinvent.scoring.aggregators.pareto_front import get_pareto_frontier
from reinvent import config_parse
import os
import argparse

'''
Sketchy way of retrievhing the scores of te previously generated compounds. The scores of these molecules are used to compute the curent Pareto front.
The newly generated molecules are then evaluated based on their hypervolume improvement compared to the current Pareto front.
'''

class HypervolumeCalculator:
    def __init__(self, scores,
                 mode: str = "hypervolume_to_dynamic_reference",
                 ):
        self.mode = mode
        self.scores = scores
        self.num_refs = 5
        self.num_scores = scores.shape[0]
        self.score_history = self.load_scores(_get_report_filename())

    def load_scores(self, report_file):
        try:
            score_history = pd.read_csv(report_file)
            cols = score_history.columns
            components = []
            for component in cols:
                if '(raw)' in component and 'SMARTS' not in component:
                    components.append(component[:-6])
            score_history = score_history[components]

        except:
            score_history = pd.DataFrame(np.random.uniform(0,0.5, (self.num_refs, self.num_scores)))
        return score_history

    def compute_hypervolume(self):
        if self.mode == "hypervolume_to_dynamic_reference":
            pareto_front = get_pareto_frontier(self.score_history.values.tolist(), direction="maximize", min_points=self.num_refs)
            pareto_front = np.negative(pareto_front)
            arbitrary_reference_points = np.full(self.num_scores, 1e-8)
            ref_hv = hypervolume(pareto_front).compute(ref_point=arbitrary_reference_points)

            hv_improvements = []
            for point in self.scores.transpose():
                point *=-1
                hv_value = hypervolume(np.vstack((pareto_front, point))).compute(ref_point=arbitrary_reference_points)
                hv_improvement = hv_value - ref_hv
                hv_improvements.append(hv_improvement)
            return hv_improvements


        elif self.mode == "hypervolume_to_fixed_reference":
            return NotImplementedError
        else:
            raise ValueError(f"Invalid mode '{self.mode}'")


def _get_report_filename():
    '''
    Get the report file name. If multiple stages are run, make sure that the most recent stage is taken
    '''
    args = _parse_command_line()
    reader = getattr(config_parse, f"read_{args.config_format}")
    input_config = reader(args.config_filename)
    report_file_basename = (input_config["parameters"]['summary_csv_prefix'])
    files_in_dir = os.listdir()
    stage_number = 0
    for file in files_in_dir:
        if file.startswith(report_file_basename) and file.endswith(".csv"):
            #stage_number = int(file.split("_")[-1].split(".")[0])
            stage_number += 1
    report_file = report_file_basename + f"_{stage_number}.csv"
    return report_file

def _parse_command_line():
    parser = argparse.ArgumentParser(
        description=f"4: a molecular design "
        f"tool for de novo design, "
        "scaffold hopping, R-group replacement, linker design, molecule "
        "optimization, and others",
        epilog=f"4",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "config_filename",
        nargs="?",
        metavar="FILE",
        type=os.path.abspath,
        help="Input configuration file with runtime parameters",
    )

    parser.add_argument(
        "-f",
        "--config-format",
        metavar="FORMAT",
        default="toml"
    )

    parser.add_argument(
        "-d",
        "--device",
        metavar="DEV",
        default=None
    )

    parser.add_argument(
        "-l",
        "--log-filename",
        metavar="FILE",
        default=None,
        type=os.path.abspath,
        help=f"File for logging information, otherwise writes to stderr.",
    )

    parser.add_argument(
        "--log-level",
        metavar="LEVEL",
        default="info"
    )

    parser.add_argument(
        "-s",
        "--seed",
        metavar="N",
        type=int,
        default=None,
        help="Sets the random seeds for reproducibility",
    )

    parser.add_argument(
        "--dotenv-filename",
        metavar="FILE",
        default=None,
        type=os.path.abspath,
        help=f"Dotenv file with environment setup needed for some scoring components. "
        "By default the one from the installation directory will be loaded.",
    )

    parser.add_argument(
        "--enable-rdkit-log-levels",
        metavar="LEVEL",
        nargs="+"
    )

    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"4"
    )
    return parser.parse_args()