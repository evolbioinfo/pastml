
METHOD = 'method'
STATES = 'states'
CHARACTER = 'character'

NUM_SCENARIOS = 'num_scenarios'
NUM_UNRESOLVED_NODES = 'num_unresolved_nodes'
NUM_STATES_PER_NODE = 'num_states_per_node_avg'
PERC_UNRESOLVED = 'percentage_of_unresolved_nodes'
NUM_NODES = 'num_nodes'
NUM_TIPS = 'num_tips'


def col_name2cat(column):
    """
    Reformats the column string to make sure it contains only numerical, letter characters or underscore.

    :param column: column name to be reformatted
    :type column: str
    :return: column name with illegal characters removed
    :rtype: str
    """
    column_string = ''.join(s for s in column.replace(' ', '_') if s.isalnum() or '_' == s)
    return column_string


def get_personalized_feature_name(character, feature):
    """
    Precedes the feature name by the character name
    (useful when likelihoods for different characters are calculated in parallel).

    :param character: str, character name
    :param feature: str, feature to be personalized
    :return: str, the personalized feature
    """
    return '{}_{}'.format(character, feature)


def value2list(n, value, default_value):
    # create a variable for n columns
    # specifying the default value if nothing was chosen
    if value is None:
        value = default_value
    # and propagating the chosen value to all columns
    if not isinstance(value, list):
        value = [value] * n
    elif len(value) == 1:
        value = value * n
    # or making sure that the default value is chosen for the columns for which the value was not specified
    else:
        value += [default_value] * (n - len(value))
    return value
