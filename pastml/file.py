import os

from pastml import col_name2cat
from pastml.ml import is_ml, is_marginal

COMBINED_ANCESTRAL_STATE_TAB = 'combined_ancestral_states.characters_{states}.tab'
NAMED_TREE_NWK = 'named.tree_{tree}'

PASTML_ML_PARAMS_TAB = 'params.character_{state}.method_{method}.model_{model}.tab'
PASTML_MP_PARAMS_TAB = 'params.character_{state}.method_{method}.tab'
PASTML_MARGINAL_PROBS_TAB = 'marginal_probabilities.character_{state}.model_{model}.tab'


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
    return template.format(state=col_name2cat(column), method=method, model=model)


def get_combined_ancestral_state_file(columns):
    """
    Get the filename where the combined ancestral states are saved (for one or several columns).
    This file is inside the work_dir that can be specified for the pastml_pipeline method.

    :param columns: list of str, the column(s) for which ancestral states are reconstructed with PASTML.
    :return: str, filename
    """
    template = COMBINED_ANCESTRAL_STATE_TAB
    return template.format(states='_'.join(col_name2cat(column) for column in columns))


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
    return PASTML_MARGINAL_PROBS_TAB.format(state=col_name2cat(column), model=model)
