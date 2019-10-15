import os

from pastml.parsimony import is_meta_mp, get_default_mp_method
from pastml import col_name2cat, get_personalized_feature_name
from pastml.ml import is_ml, is_marginal, is_meta_ml, get_default_ml_method

PASTML_WORK_DIR = '{tree}_pastml'

COMBINED_ANCESTRAL_STATE_TAB = 'combined_ancestral_states.tab'
NAMED_TREE_NWK = 'named.tree_{tree}'

PASTML_ML_PARAMS_TAB = 'params.character_{state}.method_{method}.model_{model}.tab'
PASTML_MP_PARAMS_TAB = 'params.character_{state}.method_{method}.tab'
PASTML_MARGINAL_PROBS_TAB = 'marginal_probabilities.character_{state}.model_{model}.tab'


def get_column_method(column, method):
    column = col_name2cat(column)
    if is_meta_ml(method):
        method = get_default_ml_method()
    elif is_meta_mp(method):
        method = get_default_mp_method()
    else:
        return column, method
    return get_personalized_feature_name(column, method), method


def get_pastml_parameter_file(method, model, column):
    """
    Get the filename where the PastML parameters are saved
    (for non-ML methods and input parameters will be None, as they have no parameters).
    This file is inside the work_dir that can be specified for the pastml_pipeline method.

    :param method: str, the ancestral state prediction method used by PASTML.
    :param model: str, the state evolution model used by PASTML.
    :param column: str, the column for which ancestral states are reconstructed with PASTML.
    :return: str, filename or None for non-ML methods
    """
    ml = is_ml(method)
    template = PASTML_ML_PARAMS_TAB if ml else PASTML_MP_PARAMS_TAB
    column, method = get_column_method(column, method)
    return template.format(state=column, method=method, model=model)


def get_combined_ancestral_state_file():
    """
    Get the filename where the combined ancestral states are saved (for one or several columns).
    This file is inside the work_dir that can be specified for the pastml_pipeline method.

    :return: str, filename
    """
    return COMBINED_ANCESTRAL_STATE_TAB


def get_pastml_work_dir(tree):
    """
    Get the pastml work dir path.

    :param tree: str, path to the input tree.
    :return: str, filename
    """
    return PASTML_WORK_DIR.format(tree=os.path.splitext(tree)[0])


def get_named_tree_file(tree):
    """
    Get the filename where the PastML tree (input tree but named and with collapsed zero branches) is saved.
    This file is inside the work_dir that can be specified for the pastml_pipeline method.

    :param tree: str, the input tree in newick format.
    :return: str, filename
    """
    return NAMED_TREE_NWK.format(tree=os.path.basename(tree))


def get_pastml_marginal_prob_file(method, model, column):
    """
    Get the filename where the PastML marginal probabilities of node states are saved (will be None for non-marginal methods).
    This file is inside the work_dir that can be specified for the pastml_pipeline method.

    :param method: str, the ancestral state prediction method used by PASTML.
    :param model: str, the state evolution model used by PASTML.
    :param column: str, the column for which ancestral states are reconstructed with PASTML.
    :return: str, filename or None if the method is not marginal.
    """
    if not is_marginal(method):
        return None
    column, method = get_column_method(column, method)
    return PASTML_MARGINAL_PROBS_TAB.format(state=column, model=model)
