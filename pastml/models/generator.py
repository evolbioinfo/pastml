import numpy as np


def get_diagonalisation(frequencies, rate_matrix=None):
    """
    Normalises and diagonalises the rate matrix.

    :param frequencies: character state frequencies.
    :type frequencies: numpy.array
    :param rate_matrix: (optional) rate matrix (by default an all-equal-rate matrix is used)
    :type rate_matrix: numpy.ndarray
    :return: matrix diagonalisation (d, A, A^{-1})
        such that A.dot(np.diag(d))).dot(A^{-1}) = 1/mu Q (normalised generator)
    :rtype: tuple
    """
    Q = get_normalised_generator(frequencies, rate_matrix)
    d, A = np.linalg.eig(Q)
    return d, A, np.linalg.inv(A)


def get_normalised_generator(frequencies, rate_matrix=None):
    """
    Calculates the normalised generator from the rate matrix and character state frequencies.

    :param frequencies: character state frequencies.
    :type frequencies: numpy.array
    :param rate_matrix: (optional) rate matrix (by default an all-equal-rate matrix is used)
    :type rate_matrix: numpy.ndarray
    :return: normalised generator 1/mu Q
    :rtype: numpy.ndarray
    """
    if rate_matrix is None:
        n = len(frequencies)
        rate_matrix = np.ones(shape=(n, n), dtype=np.float64) - np.eye(n)
    generator = rate_matrix * frequencies
    generator -= np.diag(generator.sum(axis=1))
    mu = -generator.diagonal().dot(frequencies)
    generator /= mu
    return generator


def get_pij_matrix(t, diag, A, A_inv):
    """
    Calculates the probability matrix of substitutions i->j over time t,
    given the normalised generator diagonalisation.


    :param t: time
    :type t: float
    :return: probability matrix
    :rtype: numpy.ndarray
    """
    return A.dot(np.diag(np.exp(diag * t))).dot(A_inv)

