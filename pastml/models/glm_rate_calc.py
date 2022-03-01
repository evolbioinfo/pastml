import logging

import numpy as np

from pastml.models.generator import get_pij_matrix, get_diagonalisation

CUSTOM_RATES = 'CUSTOM_RATES'
GLM = 'GLM'


def glm_rate_calc(glm_dict, coefficients=None):
    """
    Returns a rate matrix calculated from the GLM matrices with each factor assigned a coefficient. If no coefficients are given
    the coefficients are calculated as being equal proportionally to 1.
    """

    #THINK ABOUT WHAT IS THE BEST WAY TO CHECK WEIGHTS AND IMPORT WEIGHTS THROUGH PASTML?

    #get dictionary with just the name of factor and the matrices
    char_dict = list(glm_dict.values())[0]
    num_matrix = len(char_dict)

    #When coefficients are not given, create a new matrix that has factor name with the coefficients as equal proportionally to 1
    if coefficients==None:
        coefficients = {key: 1/num_matrix for key in char_dict.keys()}
    print(coefficients)


    rate_matrix1 = list(char_dict.values())[0]
    target_shape = rate_matrix1.shape

    final = np.zeros(shape=target_shape, dtype=float)
    print(final)

    for key in char_dict.keys():
        value = char_dict[key]
        coefficient = coefficients[key]
        final += value*coefficient
    return final
