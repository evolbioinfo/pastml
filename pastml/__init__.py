
METHOD = 'method'
STATES = 'states'
CHARACTER = 'character'

NUM_SCENARIOS = 'num_scenarios'
NUM_UNRESOLVED_NODES = 'num_unresolved_nodes'
NUM_NODES = 'num_nodes'
NUM_TIPS = 'num_tips'


def col_name2cat(column):
    """
    Reformats the column string to make sure it contains only numerical or letter characters.
    :param column: str, column name to be reformatted
    :return: str, the column name with illegal characters removed
    """
    column_string = ''.join(s for s in column if s.isalnum())
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
    # or making sure that the default value if chosen for the columns for which the value was not specified
    else:
        value += [default_value] * (n - len(value))
    return value
