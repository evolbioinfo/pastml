import os

from pastml import col_name2cat

PASTML_WORK_DIR = '{tree}_pastml'

COMBINED_ANCESTRAL_STATE_TAB = 'combined_ancestral_states.tab'
ANCESTRAL_STATE_TAB = 'ancestral_states.character_{state}.tab'
NAMED_TREE_NWK = 'named.tree_{tree}.nwk'

PASTML_STATS_TAB = 'stats.character_{state}.{acr}.tab'
PASTML_PARAMS_TAB = 'params.character_{state}.model_{model}.tab'
PASTML_COLOUR_TAB = 'colours.character_{state}.tab'
PASTML_MARGINAL_PROBS_TAB = 'marginal_probabilities.character_{state}.model_{model}.tab'


def get_acr_string(method, model=None):
    """
    Returns a string with ACR settings (method and, for ML methods, model) to be placed in file names.
    :param method: ACR method
    :param model: character state evolution model (for ML methods only)
    :return: a string with ACR settings
    """
    return 'method_{method}.model_{model}'.format(method=method, model=model) if model \
        else 'method_{method}'.format(method=method)


def get_pastml_parameter_file(model, column):
    """
    Get the filename where the PastML model parameters are saved (for max likelihood methods).

    :param model: str, the state evolution model used by PASTML.
    :param column: str, the column for which ancestral states are reconstructed with PASTML.
    :return: str, filename
    """
    return PASTML_PARAMS_TAB.format(state=col_name2cat(column), model=model)


def get_pastml_stats_file(method, model, column):
    """
    Get the filename where the statistics about the ACR are saved.

    :param method: str, the ACR method used by PASTML.
    :param model: str, the state evolution model used by PASTML (for max likelihood methods only).
    :param column: str, the column for which ancestral states are reconstructed with PASTML.
    :return: str, filename
    """
    return PASTML_STATS_TAB.format(state=col_name2cat(column), acr=get_acr_string(method=method, model=model))


def get_pastml_colour_file(column):
    """
    Get the filename where the PastML colours used for visualisation are saved.
    This file is inside the work_dir that can be specified for the pastml_pipeline method.

    :param column: str, the column for which ancestral states are reconstructed with PASTML.
    :return: str, filename
    """
    template = PASTML_COLOUR_TAB
    return template.format(state=column)


def get_combined_ancestral_state_file():
    """
    Get the filename where the combined ancestral states are saved (for one or several columns).
    This file is inside the work_dir that can be specified for the pastml_pipeline method.

    :return: str, filename
    """
    return COMBINED_ANCESTRAL_STATE_TAB


def get_ancestral_state_file(character):
    """
    Get the filename where the ancestral states for a given character are saved.
    This file is inside the work_dir that can be specified for the pastml_pipeline method.

    :return: str, filename
    """
    return ANCESTRAL_STATE_TAB.format(state=character)


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
    tree_name = os.path.splitext(os.path.basename(tree))[0]
    return NAMED_TREE_NWK.format(tree=tree_name if tree_name else 'tree')


def get_pastml_marginal_prob_file(model, column):
    """
    Get the filename where the PastML marginal probabilities of node states are saved.

    :param model: str, the state evolution model used by PASTML.
    :param column: str, the column for which ancestral states are reconstructed with PASTML.
    :return: str, filename or None if the method is not marginal.
    """
    return PASTML_MARGINAL_PROBS_TAB.format(state=col_name2cat(column), model=model)
