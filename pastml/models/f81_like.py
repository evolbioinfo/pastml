import numpy as np

JC = 'JC'
F81 = 'F81'
EFT = 'EFT'


def is_f81_like(model):
    """
    Checks if the evolutionary model is F81 or its simplification, e.g. JC or EFT.

    :param model: evolutionary model
    :type model: str
    :rtype: bool
    """
    return model in {F81, JC, EFT}


def get_mu(frequencies):
    """
    Calculates the mutation rate for F81 (and JC that is a simplification of it),
    as \mu = 1 / (1 - sum_i \pi_i^2). This way the overall rate of mutation -\mu trace(\Pi Q) is 1.
    See [Gascuel "Mathematics of Evolution and Phylogeny" 2005] for further details.

    :param frequencies: numpy array of state frequencies \pi_i
    :return: mutation rate \mu = 1 / (1 - sum_i \pi_i^2)
    """
    return 1. / (1. - frequencies.dot(frequencies))


def get_f81_pij(t, frequencies, mu):
    """
    Calculate the probability of substitution i->j over time t, given the mutation rate mu:
    For F81 (and JC which is a simpler version of it)
    Pij(t) = \pi_j (1 - exp(-mu t)) + exp(-mu t), if i == j, \pi_j (1 - exp(-mu t)), otherwise
    [Gascuel "Mathematics of Evolution and Phylogeny" 2005].

    :param frequencies: array of state frequencies \pi_i
    :type frequencies: numpy.array
    :param mu: mutation rate: \mu = 1 / (1 - sum_i \pi_i^2)
    :type mu: float
    :param t: time t
    :type t: float
    :param sf: scaling factor by which t should be multiplied.
    :type sf: float
    :return: probability matrix
    :rtype: numpy.ndarray
    """

    # if mu == inf (e.g. just one state) and t == 0, we should prioritise mu
    exp_mu_t = 0. if (mu == np.inf) else np.exp(-mu * t)
    return (1 - exp_mu_t) * frequencies + np.eye(len(frequencies)) * exp_mu_t