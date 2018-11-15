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
    parser.add_argument('--metadata', required=True, type=str)
    parser.add_argument('--htmls', required=True, type=str, nargs='+')
    parser.add_argument('--col', required=True, type=str,  nargs='+')
    parser.add_argument('--model', required=False, type=str, default=F81)
    parser.add_argument('--prediction_method', required=False, type=str, default=MPPA)
    parser.add_argument('--date_col', required=False, type=str, default=None)
    parser.add_argument('--threshold', required=False, type=int, default=15)
    parser.add_argument('--in_pars', required=False, type=str, default=None, nargs='*')
    parser.add_argument('--out_pars', required=False, type=str, default=None, nargs='*')
    parser.add_argument('--out_data', required=False, type=str, default=None)
    parser.add_argument('--work_dir', required=False, type=str, default=None)
    parser.add_argument('--verbose', action='store_true', help="print information on the progress of the analysis")
    params = parser.parse_args()

    for tree, html in zip(params.trees, params.htmls):
        work_dir = tempfile.mkdtemp() if not params.work_dir else params.work_dir
        pastml_pipeline(data=params.metadata, tree=tree,
                        html_compressed=html,
                        model=params.model, columns=params.col,
                        name_column=params.col[0],
                        prediction_method=params.prediction_method, date_column=params.date_col,
                        tip_size_threshold=params.threshold,
                        column2parameters=dict(zip(params.col, params.in_pars)) if params.in_pars else None,
                        out_data=params.out_data, work_dir=work_dir, verbose=params.verbose)
        if params.out_pars:
            for column, out_pars in zip(params.col, params.out_pars):
                pastml_out_pars = \
                    get_pastml_parameter_file(method=params.prediction_method, model=params.model, column=column)
                if pastml_out_pars:
                    copyfile(os.path.join(work_dir, pastml_out_pars), out_pars)
        if not params.work_dir:
            rmtree(work_dir)
