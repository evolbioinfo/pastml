import numpy as np

HKY = 'HKY'

KAPPA = 'kappa'

HKY_STATES = np.array(['A', 'C', 'G', 'T'])
A, C, G, T = 0, 1, 2, 3


def get_hky_pij(t, frequencies, kappa):
    """
    Calculates the probability matrix of substitutions i->j over time t,
    with HKY model [Hasegawa, Kishino and Yano 1985], given state frequencies and kappa.


    :param t: time
    :type t: float
    :param kappa: kappa parameter for HKY model
    :type kappa: float
    :param frequencies: array of state frequencies \pi_i
    :type frequencies: numpy.array
    :return: probability matrix
    :rtype: numpy.ndarray
    """
    pi_a, pi_c, pi_g, pi_t = frequencies
    pi_ag = pi_a + pi_g
    pi_ct = pi_c + pi_t
    beta = .5 / (pi_ag * pi_ct + kappa * (pi_a * pi_g + pi_c * pi_t))

    exp_min_beta_t = np.exp(-beta * t)
    exp_ct = np.exp(-beta * t * (1. + pi_ct * (kappa - 1.))) / pi_ct
    exp_ag = np.exp(-beta * t * (1. + pi_ag * (kappa - 1.))) / pi_ag
    ct_sum = (pi_ct + pi_ag * exp_min_beta_t) / pi_ct
    ag_sum = (pi_ag + pi_ct * exp_min_beta_t) / pi_ag
    p = np.ones((4, 4), dtype=np.float64) * (1 - exp_min_beta_t)
    p *= frequencies

    p[T, T] = pi_t * ct_sum + pi_c * exp_ct
    p[T, C] = pi_c * ct_sum - pi_c * exp_ct

    p[C, T] = pi_t * ct_sum - pi_t * exp_ct
    p[C, C] = pi_c * ct_sum + pi_t * exp_ct

    p[A, A] = pi_a * ag_sum + pi_g * exp_ag
    p[A, G] = pi_g * ag_sum - pi_g * exp_ag

    p[G, A] = pi_a * ag_sum - pi_a * exp_ag
    p[G, G] = pi_g * ag_sum + pi_a * exp_ag

    return p

