import logging

import numpy as np

from pastml.models.generator import get_pij_matrix, get_diagonalisation

CUSTOM_RATES = 'CUSTOM_RATES'
GLM = 'GLM'


def glm_rate_calc(glm_dict, weights=None):
    """
    Returns a rate matrix calculated from the GLM matrices with each factor assigned a weight. If no weights are given
    the weights are calculated as being equal proportionally to 1.
    """

    #THINK ABOUT WHAT IS THE BEST WAY TO CHECK WEIGHTS AND IMPORT WEIGHTS THROUGH PASTML?

    #get dictionary with just the name of factor and the matrices
    char_dict = list(glm_dict.values())[0]
    num_matrix = len(char_dict)

    #When weights are not given, create a new matrix that has factor name with the weights as equal proportionally to 1
    if weights==None:
        weights = {key: 1/num_matrix for key in char_dict.keys()}
        #weights= char_dict.copy()
        #weight_value = 1 / num_matrix
        #weights = weights.fromkeys(weights, weight_value)
    print(weights)

    #FUTURE WORK: PASS DICTIONARY WHERE KEY IS NAME OF MATRIX AND VALUE IS THE WEIGHT


    rate_matrix1 = list(char_dict.values())[0]
    #rate_matrix = rate_matrix1.copy()

    #rate_matrix[:] = 0
    #final = rate_matrix

    target_shape = rate_matrix1.shape

    final = np.zeros(shape=target_shape, dtype=float)
    print(final)

    for key in char_dict.keys():
        value = char_dict[key]
        weight = weights[key]
        final = final + value*weight
    return final
