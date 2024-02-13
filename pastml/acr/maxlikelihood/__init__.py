import numpy as np

LOG_LIKELIHOOD = 'log_likelihood'
RESTRICTED_LOG_LIKELIHOOD_FORMAT_STR = '{}_restricted_{{}}'.format(LOG_LIKELIHOOD)

JOINT = 'JOINT'
MPPA = 'MPPA'
MAP = 'MAP'

MARGINAL_PROBABILITIES = 'marginal_probabilities'

MODEL = 'model'

MIN_VALUE = np.log10(np.finfo(np.float64).eps)
MAX_VALUE = np.log10(np.finfo(np.float64).max)

MARGINAL_ML_METHODS = {MPPA, MAP}
ML_METHODS = MARGINAL_ML_METHODS | {JOINT}

AIC = 'AIC'


def calculate_AIC(loglh, k):
    """
    Calculates the Akaike information criterion (AIC) for the given model likelihood and parameter number.

    :param loglh: model loglikelihood
    :type loglh: float
    :param k: number of free model parameters
    :type k: int
    :return: AIC value
    :rtype: float
    """
    return 2 * k - 2 * loglh


def is_marginal(method):
    """
    Checks if the method is marginal, i.e. MAP, MPPA, or one of the meta-methods (ALL, ML).

    :param method: prediction method
    :type method: str
    :return: bool
    """
    return method in MARGINAL_ML_METHODS


def is_ml(method):
    """
    Checks if the method is max likelihood, i.e. JOINT or one of the marginal ones.

    :param method: prediction method
    :type method: str
    :return: bool
    """
    return method in ML_METHODS


def get_default_ml_method():
    return MPPA
