import os
import tempfile
from shutil import copyfile, rmtree

from pastml.acr import pastml_pipeline
from pastml.file import get_pastml_parameter_file
from pastml.ml import MPPA, F81

if '__main__' == __name__:
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--trees', required=True, type=str, nargs='+')
    parser.add_argument('--data', required=True, type=str)
    parser.add_argument('--htmls', required=True, type=str, nargs='+')
    parser.add_argument('--columns', required=True, type=str,  nargs='+')
    parser.add_argument('--name_column', required=False, type=str,  default=None)
    parser.add_argument('--model', required=False, type=str, default=F81)
    parser.add_argument('--prediction_method', required=False, type=str, default=MPPA)
    parser.add_argument('--date_column', required=False, type=str, default=None)
    parser.add_argument('--threshold', required=False, type=int, default=15)
    parser.add_argument('--parameters', required=False, type=str, default=None, nargs='*')
    parser.add_argument('--out_parameters', required=False, type=str, default=None, nargs='*')
    parser.add_argument('--out_data', required=False, type=str, default=None)
    parser.add_argument('--work_dir', required=False, type=str, default=None)
    parser.add_argument('--verbose', action='store_true', help="print information on the progress of the analysis")
    parser.add_argument('--joint_option', choices=['forced_joint', 'no_forced_joint'],
                        help="whether to force the addition of the JOINT state to the MPPA prediction")
    params = parser.parse_args()

    for tree, html in zip(params.trees, params.htmls):
        work_dir = tempfile.mkdtemp() if not params.work_dir else params.work_dir
        pastml_pipeline(data=params.data, tree=tree, html_compressed=html,
                        prediction_method=params.prediction_method, model=params.model,
                        columns=params.columns, name_column=params.name_column, date_column=params.date_column,
                        tip_size_threshold=params.threshold,
                        parameters=params.parameters, out_data=params.out_data, work_dir=work_dir,
                        verbose=params.verbose, no_forced_joint=params.joint_option == 'no_forced_joint')
        if params.out_parameters:
            for column, out_parameters in zip(params.columns, params.out_parameters):
                pastml_out_pars = \
                    get_pastml_parameter_file(method=params.prediction_method, model=params.model, column=column)
                if pastml_out_pars:
                    copyfile(os.path.join(work_dir, pastml_out_pars), out_parameters)
        if not params.work_dir:
            rmtree(work_dir)
