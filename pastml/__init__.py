import logging
from datetime import datetime

PASTML_VERSION = '1.9.49'

METHOD = 'method'
STATES = 'states'
CHARACTER = 'character'

NUM_SCENARIOS = 'num_scenarios'
NUM_UNRESOLVED_NODES = 'num_unresolved_nodes'
NUM_STATES_PER_NODE = 'num_states_per_node_avg'
PERC_UNRESOLVED = 'percentage_of_unresolved_nodes'
NUM_NODES = 'num_nodes'
NUM_TIPS = 'num_tips'


def datetime2numeric(d):
    """
    Converts a datetime date to numeric format.
    For example: 2016-12-31 -> 2016.9972677595629; 2016-1-1 -> 2016.0
    :param d: a date to be converted
    :type d: np.datetime
    :return: numeric representation of the date
    :rtype: float
    """
    first_jan_this_year = datetime(year=d.year, month=1, day=1)
    day_of_this_year = d - first_jan_this_year
    first_jan_next_year = datetime(year=d.year + 1, month=1, day=1)
    days_in_this_year = first_jan_next_year - first_jan_this_year
    return d.year + day_of_this_year / days_in_this_year


def numeric2datetime(d):
    """
    Converts a numeric date to  datetime format.
    For example: 2016.9972677595629 -> 2016-12-31; 2016.0 ->  2016-1-1
    :param d: numeric representation of a date to be converted
    :type d: float
    :return: the converted date
    :rtype: np.datetime
    """
    year = int(d)
    first_jan_this_year = datetime(year=year, month=1, day=1)
    first_jan_next_year = datetime(year=year + 1, month=1, day=1)
    days_in_this_year = first_jan_next_year - first_jan_this_year
    day_of_this_year = int(round(days_in_this_year.days * (d % 1), 6)) + 1
    for m in range(1, 13):
        days_in_m = (datetime(year=year if m < 12 else (year + 1), month=m % 12 + 1, day=1)
                     - datetime(year=year, month=m, day=1)).days
        if days_in_m >= day_of_this_year:
            return datetime(year=year, month=m, day=day_of_this_year)
        day_of_this_year -= days_in_m


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


def _set_up_pastml_logger(verbose):
    logger = logging.getLogger('pastml')
    logger.setLevel(level=logging.DEBUG if verbose else logging.ERROR)
    logger.propagate = False
    if not logger.hasHandlers():
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(name)s:%(levelname)s:%(asctime)s %(message)s', datefmt="%H:%M:%S")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger
